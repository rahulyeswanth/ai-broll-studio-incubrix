import hashlib
import json
import logging
import shutil
import threading
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.core.security import UnsafeURLError, current_owner, fetch_to_file
from app.pipeline.errors import InputValidationError, PipelineError
from app.pipeline.media import validate_audio_file
from app.pipeline.orchestrator import JobRequest, validate_script
from app.pipeline.renderer import DIMENSIONS

from app.providers.footage import get_footage_provider
from app.schemas.domain import BrollMatchRequest, BrollMatchResponse, SceneSegment
from app.services.matcher import BrollMatchingService
from app.services.segmentation import ScriptSegmentationService

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/match", response_model=BrollMatchResponse, status_code=status.HTTP_200_OK,
    summary="Match scenes with suitable B-roll footage",
    description="Accepts pre-segmented scenes or a raw script and selects suitable footage from the configured provider.",
)
async def match_broll(payload: BrollMatchRequest) -> BrollMatchResponse:
    provider_name = (payload.provider or "").strip().lower() or None
    try:
        footage_provider = get_footage_provider(provider_name)
    except (ValueError, NotImplementedError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    if payload.scenes:
        scenes_to_match: List[SceneSegment] = payload.scenes
    elif payload.script:
        try:
            scenes_to_match = ScriptSegmentationService().segment_script(
                script_text=payload.script.text, scene_hints=payload.script.scene_hints
            )
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    else:
        scenes_to_match = []

    if not scenes_to_match:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="No scenes available to match.")

    resolved_provider_name = provider_name or __import__("app.core.config", fromlist=["get_settings"]).get_settings().FOOTAGE_PROVIDER
    results, warnings = await BrollMatchingService(footage_provider).match_scenes(
        scenes=scenes_to_match, target_aspect_ratio=payload.target_aspect_ratio
    )
    enriched_scenes = [item.scene for item in results]
    matched = sum(item.match_status == "matched" for item in results)
    return BrollMatchResponse(
        target_aspect_ratio=payload.target_aspect_ratio, provider_used=resolved_provider_name,
        total_scenes=len(results), matched_scenes=matched, unmatched_scenes=len(results)-matched,
        results=results, scenes=enriched_scenes, warnings=warnings,
    )


# --------------------------------------------------------------------------------------------------------------------
# POST /api/v1/broll/jobs  (IncuBrix Assessment 3 contract)
# --------------------------------------------------------------------------------------------------------------------
class SceneIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=2000)
    visual_hint: Optional[str] = Field(default=None, max_length=300)


class CaptionOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    burn_in: bool = True
    srt: bool = True


class BrollJobIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    audio_url: Optional[str] = Field(default=None, max_length=2048)
    script: Optional[str] = None
    scenes: Optional[List[SceneIn]] = Field(default=None, max_length=200)
    voice: Optional[str] = Field(default=None, max_length=80)
    aspect_ratios: List[Literal["16:9", "9:16", "1:1"]] = Field(default_factory=lambda: ["16:9"], min_length=1, max_length=3)
    style: Optional[str] = Field(default=None, max_length=300)
    avoid: List[str] = Field(default_factory=list, max_length=20)
    captions: CaptionOptions = Field(default_factory=CaptionOptions)
    callback_url: Optional[str] = Field(default=None, max_length=2048)
    language: Optional[str] = Field(default=None, max_length=10)
    footage_provider: Optional[str] = Field(default=None, max_length=40)

    @field_validator("avoid")
    @classmethod
    def _avoid_len(cls, v):
        return [x.strip()[:80] for x in v if x and x.strip()]


_idem_lock = threading.Lock()


def _idem_path(store) -> Path:
    return store.root / "_idempotency.json"


def _idem_lookup(store, key: str, body_hash: str):
    with _idem_lock:
        p = _idem_path(store)
        data = json.loads(p.read_text()) if p.is_file() else {}
    rec = data.get(key)
    if rec is None:
        return None
    if rec["hash"] != body_hash:
        raise HTTPException(409, detail={"code": "idempotency_conflict", "details": {},
                                         "message": "This Idempotency-Key was already used with a different request."})
    return rec["job_id"]


def _idem_store(store, key: str, body_hash: str, job_id: str) -> None:
    with _idem_lock:
        p = _idem_path(store)
        data = json.loads(p.read_text()) if p.is_file() else {}
        data[key] = {"hash": body_hash, "job_id": job_id}
        p.write_text(json.dumps(data))


def _split_list(v: Any) -> List[str]:
    if v is None or v == "":
        return []
    if isinstance(v, list):
        return v
    v = str(v).strip()
    if v.startswith("["):
        return json.loads(v)
    return [x.strip() for x in v.split(",") if x.strip()]


def _bool(v: Any, default: bool) -> bool:
    return default if v in (None, "") else str(v).lower() in ("1", "true", "yes", "on")


@router.post("/jobs", status_code=202, summary="Create a script/voiceover to B-roll video job (background, returns 202)")
async def create_broll_job(request: Request, owner: Optional[str] = Depends(current_owner)):
    from app.api.v1.endpoints.jobs import _services     # lazy: avoids an import cycle with the jobs router
    settings, store, runner = _services()
    ctype = request.headers.get("content-type", "")
    upload = None
    try:
        if ctype.startswith("application/json"):
            try:
                raw = await request.json()
            except ValueError:
                raise HTTPException(422, detail={"code": "invalid_request", "message": "Body is not valid JSON.", "details": {}})
            if not isinstance(raw, dict):
                raise HTTPException(422, detail={"code": "invalid_request", "message": "Body must be a JSON object.", "details": {}})
        elif ctype.startswith("multipart/form-data"):
            form = await request.form()
            upload = form.get("audio")
            raw: Dict[str, Any] = {"audio_url": form.get("audio_url") or None, "script": form.get("script") or None,
                                   "voice": form.get("voice") or None, "style": form.get("style") or None,
                                   "language": form.get("language") or None, "callback_url": form.get("callback_url") or None,
                                   "footage_provider": form.get("footage_provider") or None}
            raw["scenes"] = json.loads(form["scenes"]) if form.get("scenes") else None
            raw["aspect_ratios"] = _split_list(form.get("aspect_ratios")) or ["16:9"]
            raw["avoid"] = _split_list(form.get("avoid"))
            raw["captions"] = (json.loads(form["captions"]) if form.get("captions") else
                               {"burn_in": _bool(form.get("burn_in"), True), "srt": _bool(form.get("srt"), True)})
        else:
            raise HTTPException(415, detail={"code": "invalid_request", "details": {},
                                             "message": "Send application/json or multipart/form-data."})
        body = BrollJobIn.model_validate(raw)
    except (ValidationError, json.JSONDecodeError) as exc:
        details = {"errors": [{"field": ".".join(map(str, e["loc"])), "problem": e["msg"]} for e in exc.errors()]} \
            if isinstance(exc, ValidationError) else {}
        raise HTTPException(422, detail={"code": "invalid_request", "message": "The request did not pass validation.", "details": details})

    has_upload = upload is not None and bool(getattr(upload, "filename", ""))
    script = (body.script or "").strip()
    scene_texts = [sc.text.strip() for sc in (body.scenes or []) if sc.text.strip()]
    sources = [bool(body.audio_url), has_upload]
    if all(sources):
        raise HTTPException(422, detail={"code": "invalid_request", "details": {},
                                         "message": "Provide either audio_url or an uploaded audio file, not both."})
    if script and scene_texts:
        raise HTTPException(422, detail={"code": "invalid_request", "details": {},
                                         "message": "Provide either script or scenes, not both. (With audio, either may accompany it.)"})
    if not (any(sources) or script or scene_texts):
        raise HTTPException(422, detail={"code": "invalid_input", "details": {},
                                         "message": "Provide audio (audio_url or upload), a script, or scenes."})
    text = script or "\n".join(scene_texts)
    hints = [sc.visual_hint for sc in (body.scenes or []) if sc.text.strip()]
    if body.callback_url:
        try:
            from app.core.security import assert_public_url
            assert_public_url(body.callback_url)
        except UnsafeURLError as exc:
            raise HTTPException(422, detail={"code": "invalid_request", "details": {}, "message": f"callback_url rejected: {exc}"})
    try:
        if text:
            validate_script(text, settings)
    except PipelineError as exc:
        raise HTTPException(exc.http_status, detail=exc.to_dict())

    job_id = store.new_id()
    job_dir = store.job_dir(job_id)
    audio_path = None
    original = None
    audio_hash = hashlib.sha256()
    limit = settings.max_audio_mb * 1024 * 1024
    try:
        if has_upload or body.audio_url:
            (job_dir / "input").mkdir(parents=True, exist_ok=True)
        if has_upload:
            original = Path(upload.filename).name
            ext = Path(original).suffix.lower()
            if ext not in settings.allowed_audio_extensions:
                raise HTTPException(422, detail={"code": "invalid_audio", "details": {},
                                                 "message": f"Unsupported audio format '{ext}'. Allowed: {', '.join(settings.allowed_audio_extensions)}."})
            audio_path = job_dir / "input" / f"upload{ext}"
            written = 0
            with audio_path.open("wb") as fh:
                while True:
                    chunk = await upload.read(1024 * 1024)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > limit:
                        raise HTTPException(413, detail={"code": "payload_too_large", "details": {},
                                                         "message": f"Audio file exceeds the {settings.max_audio_mb} MB limit."})
                    fh.write(chunk)
        elif body.audio_url:
            from urllib.parse import urlparse
            ext = Path(urlparse(body.audio_url).path).suffix.lower()
            if ext not in settings.allowed_audio_extensions:
                raise HTTPException(422, detail={"code": "invalid_audio_url", "details": {},
                                                 "message": f"audio_url must end in one of: {', '.join(settings.allowed_audio_extensions)}."})
            original = Path(urlparse(body.audio_url).path).name or f"remote{ext}"
            audio_path = job_dir / "input" / f"remote{ext}"
            try:
                await run_in_threadpool(fetch_to_file, body.audio_url, audio_path, limit)
            except OverflowError:
                raise HTTPException(413, detail={"code": "payload_too_large", "details": {},
                                                 "message": f"Remote audio exceeds the {settings.max_audio_mb} MB limit."})
            except (UnsafeURLError, Exception) as exc:  # noqa: BLE001 - do not leak network internals
                logger.warning("audio_url fetch failed: %s", type(exc).__name__)
                msg = str(exc) if isinstance(exc, UnsafeURLError) else "The audio URL could not be downloaded."
                raise HTTPException(422, detail={"code": "invalid_audio_url", "details": {}, "message": msg})
        if audio_path:
            audio_hash.update(audio_path.read_bytes() if audio_path.stat().st_size <= limit else b"")
            try:
                await run_in_threadpool(validate_audio_file, audio_path, settings, original_name=original)
            except PipelineError as exc:
                raise HTTPException(exc.http_status, detail=exc.to_dict())

        canonical = json.dumps({**body.model_dump(), "audio_sha": audio_hash.hexdigest(), "owner": owner}, sort_keys=True)
        body_hash = hashlib.sha256(canonical.encode()).hexdigest()
        idem_key = request.headers.get("idempotency-key")
        if idem_key:
            if not (1 <= len(idem_key) <= 128):
                raise HTTPException(422, detail={"code": "invalid_request", "details": {}, "message": "Idempotency-Key must be 1-128 characters."})
            scoped = f"{owner or '-'}:{idem_key}"
            existing = _idem_lookup(store, scoped, body_hash)
            if existing:
                shutil.rmtree(job_dir, ignore_errors=True)
                st = store.read(existing) or {}
                return JSONResponse(status_code=202, content={"job_id": existing, "state": _state(st), "status": st.get("status"),
                                                              "idempotent_replay": True,
                                                              "status_url": f"/api/v1/jobs/{existing}"},
                                    headers={"Idempotent-Replay": "true"})
    except BaseException:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise

    ratios = list(dict.fromkeys(body.aspect_ratios))
    req = JobRequest(script_text=text or None, audio_path=audio_path, audio_original_name=original, aspect_ratio=ratios[0],
                     aspect_ratios=ratios, voice=body.voice, language=body.language,
                     captions_enabled=body.captions.srt or body.captions.burn_in,
                     burn_captions=body.captions.burn_in, footage_provider=body.footage_provider,
                     scene_hints=hints if any(hints) else [], style=body.style, avoid=body.avoid)
    runner.submit(job_id, req, "uploaded_audio" if audio_path else "script_only", None)
    store.write(job_id, owner=owner, aspect_ratios=ratios, callback_url=body.callback_url, idempotency_key=bool(idem_key),
                request={"style": body.style, "avoid": body.avoid, "captions": body.captions.model_dump(), "voice": body.voice})
    if idem_key:
        _idem_store(store, f"{owner or '-'}:{idem_key}", body_hash, job_id)
    return JSONResponse(status_code=202, content={"job_id": job_id, "state": "queued", "status": "queued", "progress": 0,
                                                  "status_url": f"/api/v1/jobs/{job_id}"},
                        headers={"Location": f"/api/v1/jobs/{job_id}"})


class SwapIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: Optional[str] = Field(default=None, max_length=200, description="Optional visual hint steering the new clip, e.g. 'rainforest canopy'.")
    aspect_ratio: Optional[Literal["16:9", "9:16", "1:1"]] = Field(default=None, description="Ratio to edit (default: the first ratio of the job).")


@router.post("/jobs/{job_id}/scenes/{scene_id}/swap", status_code=200,
             summary="Swap the footage of one scene and re-render only that scene",
             description="Finds a different acceptable clip for the scene (excluding the current clip and clips used by other scenes), "
                         "re-encodes only that scene's segment, re-stitches the video with stream copy and rewrites that scene in "
                         "timeline.json. The response includes measured timings and the original full-render time.")
async def swap_scene_clip(job_id: str, scene_id: str, body: SwapIn, owner: Optional[str] = Depends(current_owner)):
    import re
    from app.pipeline.swap import job_lock, swap_scene
    from app.pipeline.orchestrator import JobRequest as _JR
    _, store, runner = _services_tuple()
    st = store.read(job_id) if re.fullmatch(r"[a-f0-9]{12}", job_id) else None
    if st is None or (owner is not None and st.get("owner") not in (None, owner)):
        raise HTTPException(404, detail={"code": "not_found", "message": "Unknown job id.", "details": {}})
    if st.get("status") != "completed":
        raise HTTPException(409, detail={"code": "invalid_request", "details": {},
                                         "message": "Only a completed job can be edited."})
    if not re.fullmatch(r"scene-\d{1,4}", scene_id):
        raise HTTPException(404, detail={"code": "not_found", "message": "Unknown scene.", "details": {}})
    job_dir = store.job_dir(job_id)
    ratios = st.get("aspect_ratios") or [None]
    ratio = body.aspect_ratio or ratios[0]
    if ratio is None:
        rd = job_dir
    else:
        rd = job_dir / f"r_{ratio.replace(':', 'x')}"
        if not rd.is_dir():
            rd = job_dir
            if ratio not in ([None] + list(ratios)) or len(ratios) > 1:
                raise HTTPException(422, detail={"code": "invalid_request", "details": {},
                                                 "message": f"This job has no editable {ratio} render."})
    if not (rd / "render_plan.json").is_file():
        raise HTTPException(409, detail={"code": "invalid_request", "details": {},
                                         "message": "This job was created before per-scene editing existed and cannot be swapped."})
    lock = job_lock(f"{job_id}:{rd.name}")
    if not lock.acquire(blocking=False):
        raise HTTPException(409, detail={"code": "invalid_request", "details": {}, "retryable": True,
                                         "message": "Another edit of this job is in progress."})
    try:
        tl = json.loads((rd / "timeline.json").read_text(encoding="utf-8"))
        deps = runner.deps_factory(_JR(footage_provider=tl.get("footage_provider")))
        try:
            full = json.loads((rd / "manifest.json").read_text(encoding="utf-8")).get("processing_durations_seconds", {}).get("render")
        except (OSError, ValueError):
            full = None
        try:
            result = await swap_scene(rd, scene_id, deps, job_id, query=body.query, full_render_seconds=full)
        except PipelineError as exc:
            raise HTTPException(exc.http_status, detail=exc.to_dict())
        if rd != job_dir:                     # multi-ratio: refresh the public copies of this ratio
            slug = rd.name[2:]
            shutil.copy2(rd / "video.mp4", job_dir / f"video_{slug}.mp4")
            shutil.copy2(rd / "timeline.json", job_dir / f"timeline_{slug}.json")
            if (ratios[0] or "").replace(":", "x") == slug:
                shutil.copy2(rd / "timeline.json", job_dir / "timeline.json")
        return result
    finally:
        lock.release()


def _services_tuple():
    from app.api.v1.endpoints.jobs import _services
    return _services()


def _state(st: Dict[str, Any]) -> str:
    return {"queued": "queued", "processing": "running", "completed": "succeeded", "failed": "failed"}.get(
        st.get("status", ""), "running" if st.get("status") else "queued")
