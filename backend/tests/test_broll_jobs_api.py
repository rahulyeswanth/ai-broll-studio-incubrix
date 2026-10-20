"""POST /api/v1/broll/jobs: contract, multi-ratio output, idempotency, error envelope, SSRF, auth, callbacks."""
import asyncio
import hashlib
import hmac
import json

import pytest

from app.core.security import UnsafeURLError, assert_public_url, sign_body
from tests.test_jobs_api import services, wait_done  # noqa: F401
from tests.pipeline.helpers import make_speechlike_audio

URL = "/api/v1/broll/jobs"
SCRIPT = "Sea turtles swim through the ocean."


async def _wait(client, jid, headers=None, timeout=240):
    import time
    end = time.time() + timeout
    while time.time() < end:
        body = (await client.get(f"/api/v1/jobs/{jid}", headers=headers)).json()
        if body["state"] in ("succeeded", "failed"):
            return body
        await asyncio.sleep(0.5)
    raise AssertionError("timeout")


@pytest.mark.asyncio
async def test_multi_ratio_json_job_produces_contract_files(async_client, services):
    r = await async_client.post(URL, json={"script": SCRIPT, "aspect_ratios": ["16:9", "9:16", "1:1"],
                                           "avoid": ["alcohol"], "captions": {"burn_in": True, "srt": True}})
    assert r.status_code == 202, r.text
    assert r.json()["state"] == "queued" and r.json()["status_url"].startswith("/api/v1/jobs/")
    body = await _wait(async_client, r.json()["job_id"])
    assert body["state"] == "succeeded", body
    assert 0 <= body["progress"] <= 100 and body["progress"] == 100
    files = set(body["files"])
    assert {"video_16x9.mp4", "video_9x16.mp4", "video_1x1.mp4", "captions.srt", "timeline.json", "manifest.json"} <= files
    manifest = (await async_client.get(f"/api/v1/jobs/{body['job_id']}/files/manifest.json")).json()
    assert manifest["aspect_ratios"] == ["16:9", "9:16", "1:1"]
    assert set(manifest["outputs"]["videos"]) == {"16:9", "9:16", "1:1"}
    dims = {}
    import subprocess
    for slug, want in (("16x9", "1920x1080"), ("9x16", "1080x1920"), ("1x1", "1080x1080")):
        res = await async_client.get(f"/api/v1/jobs/{body['job_id']}/files/video_{slug}.mp4")
        assert res.status_code == 200 and res.content[:4] != b""
        p = services[0] / f"{slug}.mp4"
        p.write_bytes(res.content)
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate,codec_name",
                              "-of", "json", str(p)], capture_output=True, text=True).stdout
        st = json.loads(out)["streams"][0]
        assert f"{st['width']}x{st['height']}" == want and st["codec_name"] == "h264" and st["r_frame_rate"] == "30/1"


@pytest.mark.asyncio
async def test_scenes_input_and_default_ratio(async_client, services):
    r = await async_client.post(URL, json={"scenes": [{"text": SCRIPT, "visual_hint": "sea turtle swimming"}]})
    assert r.status_code == 202, r.text
    body = await _wait(async_client, r.json()["job_id"])
    assert body["state"] == "succeeded" and "video.mp4" in body["files"]


@pytest.mark.asyncio
async def test_multipart_audio_upload(async_client, services):
    audio = make_speechlike_audio(services[0] / "a.mp3", 8)
    r = await async_client.post(URL, files={"audio": ("a.mp3", audio.read_bytes(), "audio/mpeg")},
                                data={"aspect_ratios": '["16:9"]', "avoid": "alcohol, people's faces"})
    assert r.status_code == 202, r.text
    body = await _wait(async_client, r.json()["job_id"])
    assert body["state"] == "succeeded" and body["input_type"] == "uploaded_audio"


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [
    {}, {"script": "   "}, {"script": SCRIPT, "aspect_ratios": ["4:3"]}, {"script": SCRIPT, "aspect_ratios": []},
    {"script": SCRIPT, "scenes": [{"text": "x y z"}]}, {"script": SCRIPT, "bogus": 1},
    {"audio_url": "ftp://x/a.mp3"}, {"audio_url": "http://127.0.0.1/a.mp3"}, {"audio_url": "http://169.254.169.254/a.mp3"},
    {"script": SCRIPT, "callback_url": "http://localhost:9/cb"},
])
async def test_invalid_requests_use_error_envelope(async_client, services, payload):
    r = await async_client.post(URL, json=payload)
    assert r.status_code in (422, 413), (payload, r.text)
    err = r.json()["error"]
    assert set(err) >= {"code", "message", "retryable", "details"} and isinstance(err["retryable"], bool)
    assert "Traceback" not in r.text


@pytest.mark.asyncio
async def test_unsupported_content_type_and_bad_json(async_client, services):
    assert (await async_client.post(URL, content=b"hi", headers={"content-type": "text/plain"})).status_code == 415
    r = await async_client.post(URL, content=b"{not json", headers={"content-type": "application/json"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "INVALID_REQUEST"


@pytest.mark.asyncio
async def test_audio_and_audio_url_together_rejected(async_client, services):
    r = await async_client.post(URL, files={"audio": ("a.mp3", b"abc", "audio/mpeg")}, data={"audio_url": "https://example.com/a.mp3"})
    assert r.status_code == 422 and "not both" in r.json()["error"]["message"]


@pytest.mark.asyncio
async def test_corrupt_upload_gives_documented_error(async_client, services):
    r = await async_client.post(URL, files={"audio": ("a.mp3", b"definitely not audio" * 100, "audio/mpeg")})
    assert r.status_code == 422 and r.json()["error"]["code"] == "INVALID_AUDIO"


@pytest.mark.asyncio
async def test_idempotency_key_replays_and_detects_conflict(async_client, services):
    h = {"Idempotency-Key": "key-123"}
    a = await async_client.post(URL, json={"script": SCRIPT}, headers=h)
    b = await async_client.post(URL, json={"script": SCRIPT}, headers=h)
    assert a.status_code == b.status_code == 202
    assert a.json()["job_id"] == b.json()["job_id"] and b.json()["idempotent_replay"] is True
    c = await async_client.post(URL, json={"script": "A different script about birds."}, headers=h)
    assert c.status_code == 409 and c.json()["error"]["code"] == "IDEMPOTENCY_CONFLICT"
    await _wait(async_client, a.json()["job_id"])


@pytest.mark.asyncio
async def test_unknown_job_is_404_envelope(async_client, services):
    r = await async_client.get("/api/v1/jobs/abcdefabcdef")
    assert r.status_code == 404 and r.json()["error"]["code"] == "NOT_FOUND"


@pytest.mark.asyncio
async def test_api_keys_isolate_jobs(async_client, services, monkeypatch):
    monkeypatch.setenv("API_KEYS", "alpha,beta")
    assert (await async_client.post(URL, json={"script": SCRIPT})).status_code == 401
    r = await async_client.post(URL, json={"script": SCRIPT}, headers={"X-API-Key": "alpha"})
    assert r.status_code == 202
    jid = r.json()["job_id"]
    assert (await async_client.get(f"/api/v1/jobs/{jid}", headers={"X-API-Key": "beta"})).status_code == 404
    assert (await async_client.get(f"/api/v1/jobs/{jid}", headers={"X-API-Key": "alpha"})).status_code == 200
    await _wait(async_client, jid, headers={"X-API-Key": "alpha"})


def test_rate_limit_returns_429_with_retry_after():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.core.errors import install_error_handlers
    from app.core.security import RateLimitMiddleware
    app = FastAPI()
    app.add_middleware(RateLimitMiddleware, per_minute=2)

    @app.post("/x/jobs")
    def _post():
        return {"ok": True}

    c = TestClient(app)
    assert [c.post("/x/jobs").status_code for _ in range(3)] == [200, 200, 429]
    r = c.post("/x/jobs")
    assert r.json()["error"]["code"] == "RATE_LIMITED" and r.json()["error"]["retryable"] is True and "retry-after" in r.headers
    assert c.get("/x/jobs").status_code in (404, 405)       # only POST job creation is limited


def test_ssrf_guard():
    for bad in ("file:///etc/passwd", "http://127.0.0.1/x.mp3", "http://[::1]/x.mp3", "http://10.0.0.5/x.mp3",
                "http://169.254.169.254/latest", "http://user:pw@example.com/x.mp3", "gopher://x"):
        with pytest.raises(UnsafeURLError):
            assert_public_url(bad)


def test_signature_matches_hmac_sha256():
    body = b'{"a":1}'
    assert sign_body("s3cret", body) == "sha256=" + hmac.new(b"s3cret", body, hashlib.sha256).hexdigest()


def test_cors_is_restricted_and_root_health_exists():
    from fastapi.testclient import TestClient
    from app.main import app
    c = TestClient(app)
    r = c.options("/api/v1/broll/jobs", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in r.headers
    r = c.options("/api/v1/broll/jobs", headers={"Origin": "http://localhost:8501", "Access-Control-Request-Method": "POST"})
    assert r.headers.get("access-control-allow-origin") == "http://localhost:8501"
    assert c.get("/health").status_code == 200
    assert c.get("/ready").status_code in (200, 503)


@pytest.mark.asyncio
async def test_signed_callback_is_delivered(async_client, services, monkeypatch):
    import http.server
    import threading
    import time
    received = {}

    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers["content-length"])
            received["body"], received["sig"] = self.rfile.read(n), self.headers.get("X-IncuBrix-Signature")
            self.send_response(204)
            self.end_headers()

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("ALLOW_PRIVATE_URLS", "true")          # test receiver is on localhost
    monkeypatch.setenv("CALLBACK_SECRET", "topsecret")
    r = await async_client.post(URL, json={"script": SCRIPT, "callback_url": f"http://127.0.0.1:{srv.server_port}/cb"})
    assert r.status_code == 202, r.text
    await _wait(async_client, r.json()["job_id"])
    end = time.time() + 20
    while "body" not in received and time.time() < end:
        await asyncio.sleep(0.2)
    srv.shutdown()
    assert received["sig"] == sign_body("topsecret", received["body"])
    payload = json.loads(received["body"])
    assert payload["job_id"] == r.json()["job_id"] and payload["state"] == "succeeded"
