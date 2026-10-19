"""API-level tests for POST /api/v1/jobs (FastAPI + httpx). Footage/TTS/ASR are test doubles; FFmpeg is real.

NOTE: these need the project's normal dependencies (fastapi, httpx, pytest-asyncio). They are written to run
with `pytest -v` from backend/ but were NOT executed in the authoring sandbox (no network, packages not installable)."""
import asyncio
import tempfile
import time
from functools import lru_cache
from pathlib import Path

import pytest

from app.api.v1.endpoints import jobs as jobs_endpoint
from app.pipeline.jobs import JobRunner, JobStore
from app.pipeline.narration import PlaceholderToneTTS
from app.pipeline.orchestrator import Deps

from tests.pipeline.helpers import (CatalogSearch, FakeTranscriber, LocalFileServer, asset, make_clip, make_settings,
                                    make_speechlike_audio)


@pytest.fixture
def services(monkeypatch):
    root = Path(tempfile.mkdtemp())
    s = make_settings(root)
    srv_dir = root / "srv"; srv_dir.mkdir()
    make_clip(srv_dir / "turtle.mp4", 4)
    server = LocalFileServer(srv_dir).__enter__()
    entries = [asset("turtle", "sea, turtle, turtles, ocean, swimming", server.url("turtle.mp4"))]
    store = JobStore(s)
    runner = JobRunner(store, lambda req: Deps(settings=s, search=CatalogSearch(entries), footage_provider_name="pixabay",
                                               get_tts=lambda: PlaceholderToneTTS(),
                                               get_transcriber=lambda: FakeTranscriber("Sea turtles swim through the ocean today.")))
    monkeypatch.setattr(jobs_endpoint, "_services", lambda: (s, store, runner))
    yield root, s
    server.__exit__(None, None, None)


async def wait_done(client, job_id, timeout=240):
    end = time.time() + timeout
    while time.time() < end:
        body = (await client.get(f"/api/v1/jobs/{job_id}")).json()
        if body["status"] in ("completed", "failed"):
            return body
        await asyncio.sleep(0.5)
    raise AssertionError("job timed out")


@pytest.mark.asyncio
async def test_script_only_job_end_to_end(async_client, services):
    r = await async_client.post("/api/v1/jobs", data={"script": "Sea turtles swim through the ocean.", "aspect_ratio": "9:16"})
    assert r.status_code == 202, r.text
    body = await wait_done(async_client, r.json()["job_id"])
    assert body["status"] == "completed", body
    assert {"video.mp4", "captions.srt", "timeline.json", "manifest.json"} <= set(body["files"])
    video = await async_client.get(body["files"]["video.mp4"].replace("/jobs", "/api/v1/jobs", 1))
    assert video.status_code == 200 and video.headers["content-type"] == "video/mp4" and len(video.content) > 1000


@pytest.mark.asyncio
async def test_audio_upload_job(async_client, services):
    root, _ = services
    audio = make_speechlike_audio(root / "a.mp3", 8)
    r = await async_client.post("/api/v1/jobs", files={"audio": ("a.mp3", audio.read_bytes(), "audio/mpeg")}, data={"aspect_ratio": "16:9"})
    assert r.status_code == 202, r.text
    body = await wait_done(async_client, r.json()["job_id"])
    assert body["status"] == "completed", body
    assert body["input_type"] == "uploaded_audio"


@pytest.mark.asyncio
async def test_validation_errors_are_structured(async_client, services):
    r = await async_client.post("/api/v1/jobs", data={"script": "   "})
    assert r.status_code == 422 and r.json()["detail"]["code"] == "invalid_input"
    r = await async_client.post("/api/v1/jobs", data={})
    assert r.status_code == 422
    r = await async_client.post("/api/v1/jobs", data={"script": "Hello world there.", "aspect_ratio": "4:3"})
    assert r.status_code == 422
    r = await async_client.post("/api/v1/jobs", files={"audio": ("x.exe", b"abc", "application/octet-stream")})
    assert r.status_code == 422 and r.json()["detail"]["code"] == "invalid_audio"
    r = await async_client.post("/api/v1/jobs", files={"audio": ("bad.mp3", b"not audio" * 100, "audio/mpeg")})
    assert r.status_code == 422 and r.json()["detail"]["code"] == "invalid_audio"


@pytest.mark.asyncio
async def test_unknown_job_and_path_traversal(async_client, services):
    assert (await async_client.get("/api/v1/jobs/aaaaaaaaaaaa")).status_code == 404
    assert (await async_client.get("/api/v1/jobs/aaaaaaaaaaaa/files/video.mp4")).status_code == 404
    assert (await async_client.get("/api/v1/jobs/..%2f..%2fetc/files/passwd")).status_code in (404, 422)


@pytest.mark.asyncio
async def test_voices_and_config_endpoints_do_not_leak_keys(async_client, services):
    v = (await async_client.get("/api/v1/jobs/voices")).json()
    assert any(x["id"] == "en-US-AriaNeural" for x in v["voices"])
    cfg = (await async_client.get("/api/v1/jobs/config")).text
    assert "api_key" not in cfg.lower().replace("_key_configured", "")
