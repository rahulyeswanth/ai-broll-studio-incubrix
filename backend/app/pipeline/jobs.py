
"""File-backed job store + background thread runner (no Redis needed).
Each job lives in <storage_root>/jobs/<job_id>/ with a job.json state
file next to its outputs.
"""
from __future__ import annotations

import asyncio
import dataclasses
import datetime as dt
import json
import logging
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .config import PipelineSettings
from app.core.errors import public_error
from .diagnostics import friendly_error
from .errors import PipelineError
from .multi import run_multi_ratio
from .orchestrator import Deps, JobRequest, run_pipeline

log = logging.getLogger(__name__)

JOB_ID_RE = re.compile(r"^[a-f0-9]{12}$")
_RATIO_FILE_RE = re.compile(r"(video|timeline|manifest)_(16x9|9x16|1x1)\.(mp4|json)")
DOWNLOADABLE = {
    "video.mp4",
    "captions.srt",
    "timeline.json",
    "manifest.json",
    "job.json",
}


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def downloadable_name(job_dir: Path, name: str) -> Optional[Path]:
    """Whitelist lookup; never builds paths from raw user input."""
    m = _RATIO_FILE_RE.fullmatch(name or "")
    if (name in DOWNLOADABLE) or (m and (m.group(1) == "video") == (m.group(3) == "mp4")) or re.fullmatch(
        r"narration\.(mp3|wav|m4a|aac|ogg|flac)", name
    ) or re.fullmatch(r"thumb_scene-\d{1,4}\.jpg", name):
        path = job_dir / name
        return path if path.is_file() else None

    return None


class JobStore:
    def __init__(self, settings: PipelineSettings):
        self.settings = settings
        self.root = settings.jobs_dir
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def new_id(self) -> str:
        return uuid.uuid4().hex[:12]

    def job_dir(self, job_id: str) -> Path:
        if not JOB_ID_RE.fullmatch(job_id or ""):
            raise ValueError("invalid job id")
        return self.root / job_id

    def write(self, job_id: str, **fields: Any) -> Dict[str, Any]:
        """Atomically update job state using a unique temporary file.

        The lock protects writes through this JobStore instance.
        A bounded retry handles transient Windows file replacement errors.
        """
        tmp_path: Optional[Path] = None

        with self._lock:
            directory = self.job_dir(job_id)
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / "job.json"

            if path.is_file():
                state = json.loads(path.read_text(encoding="utf-8"))
            else:
                state = {
                    "job_id": job_id,
                    "created_at": _now(),
                }

            state.update(fields)
            state["updated_at"] = _now()

            try:
                # Unique temporary file in the destination directory.
                fd, tmp_name = tempfile.mkstemp(
                    prefix=".job-",
                    suffix=".tmp",
                    dir=str(directory),
                )
                tmp_path = Path(tmp_name)

                # Write and flush the complete state before replacement.
                with os.fdopen(
                    fd, "w", encoding="utf-8", newline="\n"
                ) as file:
                    json.dump(state, file, indent=2)
                    file.flush()
                    os.fsync(file.fileno())

                # Retry transient Windows sharing/permission failures.
                for attempt in range(5):
                    try:
                        os.replace(str(tmp_path), str(path))
                        tmp_path = None
                        break
                    except PermissionError:
                        if attempt == 4:
                            raise

                        delay = 0.1 * (attempt + 1)
                        log.warning(
                            "Temporary PermissionError replacing job state "
                            "for job %s; retry %s/5 in %.1f seconds",
                            job_id,
                            attempt + 1,
                            delay,
                        )
                        time.sleep(delay)

                return state

            finally:
                # Remove only this write's temporary file.
                if tmp_path is not None:
                    try:
                        tmp_path.unlink(missing_ok=True)
                    except OSError:
                        log.warning(
                            "Could not remove temporary job state file "
                            "for job %s: %s",
                            job_id,
                            tmp_path,
                        )

    def read(self, job_id: str) -> Optional[Dict[str, Any]]:
        try:
            path = self.job_dir(job_id) / "job.json"
        except ValueError:
            return None

        if not path.is_file():
            return None

        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None

    def recover_interrupted(self) -> int:
        count = 0

        for directory in self.root.iterdir() if self.root.is_dir() else []:
            if not directory.is_dir() or not JOB_ID_RE.fullmatch(
                directory.name
            ):
                continue

            state = self.read(directory.name)

            if state and state.get("status") in ("queued", "processing"):
                self.write(
                    directory.name,
                    status="failed",
                    error={
                        "code": "interrupted",
                        "message": (
                            "The server restarted while this job was running."
                        ),
                        "details": {},
                    },
                )
                count += 1

        return count

    def cleanup(
        self,
        max_tmp_age_hours: float = 24.0,
        retention_days: Optional[float] = None,
    ) -> Dict[str, int]:
        """Delete abandoned tmp directories and optionally old jobs."""
        removed_tmp = 0
        removed_jobs = 0
        now = time.time()

        for directory in self.root.iterdir() if self.root.is_dir() else []:
            if (
                not directory.is_dir()
                or not JOB_ID_RE.fullmatch(directory.name)
            ):
                continue

            tmp_dir = directory / "tmp"

            if (
                tmp_dir.is_dir()
                and now - tmp_dir.stat().st_mtime
                > max_tmp_age_hours * 3600
            ):
                shutil.rmtree(tmp_dir, ignore_errors=True)
                removed_tmp += 1

            if (
                retention_days is not None
                and now - directory.stat().st_mtime
                > retention_days * 86400
            ):
                shutil.rmtree(directory, ignore_errors=True)
                removed_jobs += 1

        return {
            "tmp_dirs": removed_tmp,
            "jobs": removed_jobs,
        }


class JobRunner:
    def __init__(
        self,
        store: JobStore,
        deps_factory: Callable[[JobRequest], Deps],
        max_workers: int = 2,
    ):
        self.store = store
        self.deps_factory = deps_factory
        self.pool = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="broll-job",
        )
        self._active: Dict[str, str] = {}      # request fingerprint -> job id (queued/processing only)
        self._active_lock = threading.Lock()

    def find_active(self, fingerprint: Optional[str]) -> Optional[str]:
        """Job id of an identical request that is still queued/processing (double-click / retry safety)."""
        if not fingerprint:
            return None
        with self._active_lock:
            jid = self._active.get(fingerprint)
        if jid:
            st = self.store.read(jid)
            if st and st.get("status") in ("queued", "processing"):
                return jid
            with self._active_lock:
                self._active.pop(fingerprint, None)
        return None

    def submit(
        self,
        job_id: str,
        req: JobRequest,
        input_type: str,
        fingerprint: Optional[str] = None,
    ) -> None:
        if fingerprint:
            with self._active_lock:
                self._active[fingerprint] = job_id
        self.store.write(
            job_id,
            status="queued",
            progress=0,
            step="Queued",
            input_type=input_type,
            aspect_ratio=req.aspect_ratio,
            error=None,
            request_fingerprint=fingerprint,
            footage_provider=req.footage_provider,
        )

        self.pool.submit(self._run, job_id, req, fingerprint)

    def _run(self, job_id: str, req: JobRequest, fingerprint: Optional[str] = None) -> None:
        try:
            self._run_inner(job_id, req)
        finally:
            if fingerprint:
                with self._active_lock:
                    if self._active.get(fingerprint) == job_id:
                        self._active.pop(fingerprint, None)

    def _run_inner(self, job_id: str, req: JobRequest) -> None:
        from app.core.logging import job_id_ctx
        job_id_ctx.set(job_id)
        try:
            self._run_core(job_id, req)
        finally:
            st = self.store.read(job_id) or {}
            if st.get("status") == "failed":
                self._notify(job_id)

    def _run_core(self, job_id: str, req: JobRequest) -> None:
        store = self.store

        def progress(pct: float, message: str):
            store.write(
                job_id,
                status="processing",
                progress=round(float(pct), 1),
                step=message,
            )

        try:
            store.write(
                job_id,
                status="processing",
                progress=1,
                step="Starting",
            )

            deps = self.deps_factory(req)

            attempts = max(1, int(os.environ.get("JOB_MAX_ATTEMPTS", "2")))
            for attempt in range(1, attempts + 1):
                try:
                    runner_fn = run_multi_ratio if len(req.aspect_ratios or []) > 1 else run_pipeline
                    if runner_fn is run_pipeline and req.aspect_ratios:
                        req = dataclasses.replace(req, aspect_ratio=req.aspect_ratios[0], aspect_ratios=[])
                    manifest = asyncio.run(runner_fn(req, deps, store.job_dir(job_id), job_id, progress))
                    break
                except PipelineError as exc:
                    if not exc.retryable or attempt == attempts:
                        raise
                    log.warning("Job %s attempt %s/%s failed (%s); retrying", job_id, attempt, attempts, exc.code)
                    store.write(job_id, step=f"Retrying after {exc.code} (attempt {attempt + 1}/{attempts})", attempts=attempt + 1)

            store.write(
                job_id,
                status="completed",
                progress=100,
                step="Done",
                error=None,
                outputs=manifest["outputs"],
                warnings=manifest["warnings"],
                manifest_status=manifest["status"],
                summary=(manifest.get("diagnostics") or {}).get("summary", []),
                counts=(manifest.get("diagnostics") or {}).get("counts", {}),
            )
            self._publish_outputs(job_id)
            self._notify(job_id)

        except PipelineError as exc:
            log.warning("Job %s failed: %s", job_id, exc.message)

            try:
                store.write(
                    job_id,
                    status="failed",
                    step="Failed",
                    error={**exc.to_dict(), "user_message": friendly_error(exc.code, exc.message)},
                    outputs=self._partial(job_id),
                )
            except Exception:
                log.exception(
                    "Could not persist failure state for job %s",
                    job_id,
                )

        except Exception as exc:  # noqa: BLE001
            log.exception("Job %s crashed", job_id)

            try:
                store.write(
                    job_id,
                    status="failed",
                    step="Failed",
                    outputs=self._partial(job_id),
                    error={
                        "code": "internal_error",
                        "user_message": friendly_error("internal_error"),
                        "message": (
                            f"Unexpected error: {type(exc).__name__}. "
                            "See server logs."
                        ),
                        "details": {},
                    },
                )
            except Exception:
                log.exception(
                    "Could not persist crash state for job %s",
                    job_id,
                )

    def _publish_outputs(self, job_id: str) -> None:
        """Upload deliverables to MinIO/S3 when STORAGE_TYPE=minio; records presigned download URLs in job.json."""
        if os.environ.get("STORAGE_TYPE", "local").lower() != "minio":
            return
        try:
            from app.storage.minio_adapter import MinioStorageAdapter
            ad = MinioStorageAdapter(os.environ.get("MINIO_ENDPOINT", "localhost:9000"),
                                     os.environ.get("MINIO_ACCESS_KEY", ""), os.environ.get("MINIO_SECRET_KEY", ""),
                                     os.environ.get("MINIO_BUCKET_NAME", "broll-assets"),
                                     os.environ.get("MINIO_SECURE", "false").lower() == "true",
                                     public_endpoint=os.environ.get("MINIO_PUBLIC_ENDPOINT") or None)
            d = self.store.job_dir(job_id)

            async def _push():
                await ad.check_health()          # creates the bucket when missing
                found = {}
                for p in sorted(d.iterdir()):
                    if p.is_file() and p.name != "job.json" and downloadable_name(d, p.name):
                        ctype = "video/mp4" if p.suffix == ".mp4" else "application/json" if p.suffix == ".json" else "text/plain"
                        await ad.put_file(f"jobs/{job_id}/{p.name}", p, ctype)
                        found[p.name] = await ad.get_download_url(f"jobs/{job_id}/{p.name}", 7 * 24 * 3600)
                return found

            urls = asyncio.run(_push())
            self.store.write(job_id, storage={"type": "minio", "bucket": os.environ.get("MINIO_BUCKET_NAME", "broll-assets")},
                             download_urls=urls)
        except Exception as exc:  # noqa: BLE001 - local files stay available; surface the failure instead of hiding it
            log.exception("Output upload failed for job %s", job_id)
            st = self.store.read(job_id) or {}
            self.store.write(job_id, warnings=list(st.get("warnings", [])) + [f"Output upload to object storage failed ({type(exc).__name__}); files remain downloadable from the API."])

    def _notify(self, job_id: str) -> None:
        """Optional HMAC-SHA256 signed completion callback (X-IncuBrix-Signature: sha256=<hex>)."""
        st = self.store.read(job_id) or {}
        url, secret = st.get("callback_url"), os.environ.get("CALLBACK_SECRET", "")
        if not url:
            return
        try:
            import httpx
            from app.core.security import assert_public_url, sign_body
            assert_public_url(url)
            body = json.dumps({"job_id": job_id, "status": st.get("status"), "state": "succeeded" if st.get("status") == "completed" else "failed",
                               "progress": st.get("progress"), "outputs": st.get("outputs"), "error": public_error(st.get("error"))},
                              sort_keys=True).encode()
            headers = {"Content-Type": "application/json", "X-IncuBrix-Job-Id": job_id}
            if secret:
                headers["X-IncuBrix-Signature"] = sign_body(secret, body)
            ok = False
            for attempt in range(3):
                try:
                    r = httpx.post(url, content=body, headers=headers, timeout=10, follow_redirects=False)
                    if r.status_code < 300:
                        ok = True
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(2 ** attempt)
            self.store.write(job_id, callback_delivered=ok)
        except Exception as exc:  # noqa: BLE001
            log.warning("Callback for job %s not delivered: %s", job_id, type(exc).__name__)
            self.store.write(job_id, callback_delivered=False)

    def _partial(self, job_id: str) -> Dict[str, Any]:
        directory = self.store.job_dir(job_id)

        return {
            name: True
            for name in (
                "video.mp4",
                "captions.srt",
                "timeline.json",
                "manifest.json",
            )
            if (directory / name).is_file()
        }
