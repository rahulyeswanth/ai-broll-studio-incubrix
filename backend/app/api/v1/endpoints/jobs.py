"""Single-call video generation: ``POST /api/v1/jobs`` (script and/or audio) -> poll -> download."""
import hashlib
import logging
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.core.security import current_owner
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse

from app.pipeline.config import PipelineSettings, load_settings
from app.core.errors import public_error
from app.pipeline.errors import InputValidationError, PipelineError
from app.pipeline.factory import build_deps
from app.pipeline.jobs import JobRunner, JobStore, downloadable_name
from app.pipeline.media import validate_audio_file
from app.pipeline.narration import VOICE_PRESETS
from app.pipeline.orchestrator import JobRequest, validate_script
from app.pipeline.renderer import DIMENSIONS

router = APIRouter()
logger = logging.getLogger(__name__)


@lru_cache()
def _services():
    settings = load_settings()
    store = JobStore(settings)
    store.recover_interrupted()
    store.cleanup()
    return settings, store, JobRunner(store, lambda req: build_deps(settings, req))


def _err(exc: PipelineError) -> HTTPException:
    return HTTPException(status_code=exc.http_status, detail=exc.to_dict())


@router.get("/voices", summary="Narration voice presets (any Edge TTS voice short name also works)")
def list_voices():
    settings, _, _ = _services()
    return {"default": settings.default_voice, "voices": VOICE_PRESETS, "tts_provider": settings.tts_provider}


@router.get("/config", summary="Non-secret pipeline configuration summary")
def pipeline_config():
    s, _, _ = _services()
    return {"tts_provider": s.tts_provider, "transcription_provider": s.transcription_provider,
            "footage_provider": s.footage_provider, "allow_mock_providers": s.allow_mock_providers,
            "visual_verifier": s.visual_verifier, "match_threshold": s.match_threshold,
            "review_floor": s.review_floor, "fallback_policy": s.fallback_policy,
            "allowed_audio_extensions": list(s.allowed_audio_extensions), "max_audio_mb": s.max_audio_mb,
            "pixabay_key_configured": bool(s.pixabay_api_key), "pexels_key_configured": bool(s.pexels_api_key)}


@router.post("", status_code=202, summary="Generate a video from a script and/or uploaded audio")
async def create_job(
    script: Optional[str] = Form(default=None),
    audio: Optional[UploadFile] = File(default=None),
    aspect_ratio: str = Form(default="16:9"),
    voice: Optional[str] = Form(default=None),
    language: Optional[str] = Form(default=None),
    captions: bool = Form(default=True),
    burn_captions: bool = Form(default=True),
    footage_provider: Optional[str] = Form(default=None),
):
    settings, store, runner = _services()
    has_script = bool((script or "").strip())
    has_audio = audio is not None and bool(audio.filename)
    try:
        if aspect_ratio not in DIMENSIONS:
            raise InputValidationError("Unsupported aspect ratio. Use 16:9, 9:16 or 1:1.")
        if not has_script and not has_audio:
            raise InputValidationError("Provide a script, an audio file, or both.")
        if has_script:
            validate_script(script, settings)
    except PipelineError as exc:
        raise _err(exc)

    fp_parts = [script.strip() if has_script else "", aspect_ratio, voice or "", language or "", str(captions),
                str(burn_captions), footage_provider or ""]
    fingerprint = None
    if not has_audio:       # audio requests are fingerprinted after the upload has been read (below)
        fingerprint = hashlib.sha256("\x1f".join(fp_parts).encode("utf-8")).hexdigest()
        existing = runner.find_active(fingerprint)
        if existing:
            return JSONResponse(status_code=202, content={"job_id": existing, "status": "processing", "deduplicated": True,
                                                          "status_url": f"/jobs/{existing}"})

    job_id = store.new_id()
    job_dir = store.job_dir(job_id)
    audio_path = None
    original = None
    if has_audio:
        original = Path(audio.filename).name
        ext = Path(original).suffix.lower()
        if ext not in settings.allowed_audio_extensions:
            raise HTTPException(422, detail={"code": "invalid_audio", "details": {},
                                             "message": f"Unsupported audio format '{ext}'. Allowed: {', '.join(settings.allowed_audio_extensions)}."})
        (job_dir / "input").mkdir(parents=True, exist_ok=True)
        audio_path = job_dir / "input" / f"upload{ext}"
        limit = settings.max_audio_mb * 1024 * 1024
        written = 0
        audio_hash = hashlib.sha256()
        with audio_path.open("wb") as fh:
            while True:
                chunk = await audio.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > limit:
                    fh.close()
                    shutil.rmtree(job_dir, ignore_errors=True)
                    raise HTTPException(413, detail={"code": "invalid_audio", "details": {},
                                                     "message": f"Audio file exceeds the {settings.max_audio_mb} MB limit."})
                fh.write(chunk)
                audio_hash.update(chunk)
        fingerprint = hashlib.sha256(("\x1f".join(fp_parts) + audio_hash.hexdigest()).encode("utf-8")).hexdigest()
        existing = runner.find_active(fingerprint)
        if existing:
            shutil.rmtree(job_dir, ignore_errors=True)
            return JSONResponse(status_code=202, content={"job_id": existing, "status": "processing", "deduplicated": True,
                                                          "status_url": f"/jobs/{existing}"})
        try:
            await run_in_threadpool(validate_audio_file, audio_path, settings, original_name=original)
        except PipelineError as exc:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise _err(exc)

    req = JobRequest(script_text=script if has_script else None, audio_path=audio_path, audio_original_name=original,
                     aspect_ratio=aspect_ratio, voice=voice or None, language=language or None,
                     captions_enabled=captions, burn_captions=burn_captions and captions,
                     footage_provider=(footage_provider or None))
    runner.submit(job_id, req, "uploaded_audio" if has_audio else "script_only", fingerprint)
    return {"job_id": job_id, "status": "queued", "status_url": f"/jobs/{job_id}"}


@router.get("/{job_id}", summary="Job status, progress and (when finished) output links")
def get_job(job_id: str, owner: Optional[str] = Depends(current_owner)):
    _, store, _ = _services()
    state = store.read(job_id)
    if state is None or (owner is not None and state.get("owner") not in (None, owner)):
        raise HTTPException(404, detail={"code": "not_found", "message": "Unknown job id.", "details": {}})
    state.pop("owner", None)
    state.pop("callback_url", None)
    if state.get("error"):
        state["error"] = public_error(state["error"])
    state["state"] = {"queued": "queued", "processing": "running", "completed": "succeeded", "failed": "failed"}.get(
        state.get("status", ""), "running")
    d = store.job_dir(job_id)
    state["files"] = {n: f"/jobs/{job_id}/files/{n}" for n in
                      [p.name for p in d.iterdir() if downloadable_name(d, p.name) and p.name != "job.json"]} if d.is_dir() else {}
    return state


@router.get("/{job_id}/files/{name}", summary="Download a job output (video, narration, captions, timeline, manifest)")
def get_job_file(job_id: str, name: str, owner: Optional[str] = Depends(current_owner)):
    _, store, _ = _services()
    st = store.read(job_id)
    if st is None or (owner is not None and st.get("owner") not in (None, owner)):
        raise HTTPException(404, detail={"code": "not_found", "message": "Unknown job id.", "details": {}})
    try:
        d = store.job_dir(job_id)
    except ValueError:
        raise HTTPException(404, detail={"code": "not_found", "message": "Unknown job id.", "details": {}})
    path = downloadable_name(d, name)
    if path is None:
        raise HTTPException(404, detail={"code": "not_found", "message": "File not available.", "details": {}})
    media = {".mp4": "video/mp4", ".mp3": "audio/mpeg", ".wav": "audio/wav", ".m4a": "audio/mp4", ".srt": "application/x-subrip",
             ".json": "application/json", ".jpg": "image/jpeg"}.get(path.suffix.lower(), "application/octet-stream")
    return FileResponse(path, media_type=media, filename=f"{job_id}_{name}")
