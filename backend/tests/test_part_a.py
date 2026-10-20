"""Assessment 3, part A: error codes, idempotency, health/ready, logs, min scene length, normalisation, providers, swap."""
import asyncio
import json
import logging
import re
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

from app.api.v1.endpoints import jobs as jobs_endpoint
from app.core.errors import ERROR_CODES, public_error, wire_code
from app.pipeline import normalize
from app.pipeline.alignment import merge_short_scenes
from app.pipeline.jobs import JobRunner, JobStore
from app.pipeline.multi_provider import MultiProviderSearch
from app.pipeline.narration import PlaceholderToneTTS
from app.pipeline.orchestrator import Deps
from tests.pipeline.helpers import CatalogSearch, FakeTranscriber, LocalFileServer, asset, make_clip, make_settings

URL = "/api/v1/broll/jobs"
SCRIPT = "Sea turtles swim through the ocean."


async def _wait(client, jid, timeout=240):
    end = time.time() + timeout
    while time.time() < end:
        body = (await client.get(f"/api/v1/jobs/{jid}")).json()
        if body["status"] in ("completed", "failed"):
            return body
        await asyncio.sleep(0.4)
    raise AssertionError("timeout")


@pytest.fixture
def two_clip_services(monkeypatch):
    root = Path(tempfile.mkdtemp())
    s = make_settings(root)
    srv = root / "srv"; srv.mkdir()
    make_clip(srv / "a.mp4", 4, pattern="testsrc2")
    make_clip(srv / "b.mp4", 4, pattern="smptebars")
    server = LocalFileServer(srv).__enter__()
    entries = [asset("turtle-a", "sea, turtle, turtles, ocean, swimming", server.url("a.mp4")),
               asset("turtle-b", "sea, turtle, turtles, ocean, swimming", server.url("b.mp4"))]
    store = JobStore(s)
    runner = JobRunner(store, lambda req: Deps(settings=s, search=CatalogSearch(entries), footage_provider_name="pixabay",
                                               get_tts=lambda: PlaceholderToneTTS(),
                                               get_transcriber=lambda: FakeTranscriber(SCRIPT)))
    monkeypatch.setattr(jobs_endpoint, "_services", lambda: (s, store, runner))
    yield root, s
    server.__exit__(None, None, None)


# 1. error codes ---------------------------------------------------------------------------------
def test_error_codes_are_upper_snake_case_and_documented():
    assert ERROR_CODES and all(re.fullmatch(r"[A-Z][A-Z0-9]*(_[A-Z0-9]+)*", c) for c in ERROR_CODES)
    readme = (Path(__file__).resolve().parents[2] / "README.md").read_text(encoding="utf-8")
    missing = [c for c in ERROR_CODES if f"`{c}`" not in readme]
    assert not missing, f"README error-code table is missing {missing}"


def test_wire_code_and_public_error():
    assert wire_code("invalid_audio") == "INVALID_AUDIO" and wire_code(None) == "INTERNAL_ERROR"
    assert public_error({"code": "download_failed", "message": "m"})["code"] == "DOWNLOAD_FAILED"


@pytest.mark.asyncio
async def test_error_envelope_code_is_uppercase(async_client, two_clip_services):
    r = await async_client.post(URL, json={})
    assert r.status_code == 422
    assert re.fullmatch(r"[A-Z_]+", r.json()["error"]["code"])
    r = await async_client.get("/api/v1/jobs/aaaaaaaaaaaa")
    assert r.status_code == 404 and r.json()["error"]["code"] == "NOT_FOUND"


# 2. idempotency on every POST ---------------------------------------------------------------------
@pytest.mark.asyncio
async def test_idempotency_replay_conflict_and_key_validation(async_client, two_clip_services):
    h = {"Idempotency-Key": "key-abc-1"}
    r1 = await async_client.post(URL, json={"script": SCRIPT}, headers=h)
    assert r1.status_code == 202
    r2 = await async_client.post(URL, json={"script": SCRIPT}, headers=h)
    assert r2.status_code == 202 and r2.json()["job_id"] == r1.json()["job_id"]
    assert r2.headers.get("idempotent-replay") == "true" and r2.json().get("idempotent_replay") is True
    r3 = await async_client.post(URL, json={"script": "A different script about fish."}, headers=h)
    assert r3.status_code == 409 and r3.json()["error"]["code"] == "IDEMPOTENCY_CONFLICT"
    r4 = await async_client.post(URL, json={"script": SCRIPT}, headers={"Idempotency-Key": "x" * 129})
    assert r4.status_code in (400, 422) and "Traceback" not in r4.text
    await _wait(async_client, r1.json()["job_id"])


@pytest.mark.asyncio
async def test_idempotency_applies_to_the_legacy_post_too(async_client, two_clip_services):
    h = {"Idempotency-Key": "legacy-1"}
    r1 = await async_client.post("/api/v1/jobs", data={"script": SCRIPT, "aspect_ratio": "16:9"}, headers=h)
    r2 = await async_client.post("/api/v1/jobs", data={"script": SCRIPT, "aspect_ratio": "16:9"}, headers=h)
    assert r1.status_code == 202 and r2.status_code == 202 and r1.json()["job_id"] == r2.json()["job_id"]
    await _wait(async_client, r1.json()["job_id"])


@pytest.mark.asyncio
async def test_failed_requests_are_not_cached(async_client, two_clip_services):
    h = {"Idempotency-Key": "retry-after-422"}
    assert (await async_client.post(URL, json={}, headers=h)).status_code == 422
    r = await async_client.post(URL, json={"script": SCRIPT}, headers=h)
    assert r.status_code == 202
    await _wait(async_client, r.json()["job_id"])


# 3. health, ready, logs ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_ready_reports_real_dependency_checks(async_client, two_clip_services):
    r = await async_client.get("/api/v1/ready")
    body = r.json()
    deps = body["dependencies"]
    assert {"redis", "ffmpeg", "ffprobe", "jobs_dir", "storage"} <= set(deps)
    assert deps["ffmpeg"]["status"] == "up" and deps["jobs_dir"]["status"] == "up"
    assert deps["redis"]["required"] is False
    assert r.status_code == (200 if body["ready"] else 503)


@pytest.mark.asyncio
async def test_correlation_id_and_job_id_in_json_logs(async_client, two_clip_services):
    from app.core.logging import StructuredJsonFormatter
    lines = []

    class H(logging.Handler):
        def emit(self, rec):
            if rec.name == "app.access":
                lines.append(json.loads(StructuredJsonFormatter().format(rec)))

    h = H()
    lg = logging.getLogger("app.access"); lg.addHandler(h); lg.setLevel(logging.INFO)
    try:
        r = await async_client.post(URL, json={"script": SCRIPT}, headers={"X-Correlation-Id": "corr-123"})
        assert r.headers["x-correlation-id"] == "corr-123"
        jid = r.json()["job_id"]
        await _wait(async_client, jid)
        await async_client.get(f"/api/v1/jobs/{jid}", headers={"X-Correlation-Id": "corr-456"})
        await async_client.get("/api/v1/health")
    finally:
        lg.removeHandler(h)
    assert any(l["correlation_id"] == "corr-123" for l in lines)
    assert any(l["correlation_id"] == "corr-456" and l["job_id"] == jid for l in lines)
    assert not any(l["message"].startswith("GET /api/v1/health") for l in lines)


# 5. minimum scene length --------------------------------------------------------------------------
def test_merge_short_scenes_enforces_minimum():
    from app.pipeline.models import Scene
    scenes = [Scene(scene_id=f"scene-{i+1}", index=i + 1, text=t, start=s, end=e)
              for i, (t, s, e) in enumerate([("One.", 0, 0.8), ("Two.", 0.8, 1.4), ("Three is long enough here.", 1.4, 5.0),
                                              ("Four.", 5.0, 5.5)])]
    out, groups = merge_short_scenes(scenes, 2.0, return_groups=True)
    assert all(s.end - s.start >= 2.0 - 1e-6 for s in out)
    assert out[0].start == 0 and out[-1].end == 5.5
    assert all(a.end == b.start for a, b in zip(out, out[1:]))
    assert sum(len(g) for g in groups) == 4


# 6. colour normalisation --------------------------------------------------------------------------
def test_normalisation_bounds_and_dead_zone():
    tgt = {"luma": 0.5, "saturation": 0.5}
    assert normalize.target_from([{"luma": .2, "saturation": .5}]) is None
    a = normalize.adjustment({"luma": 0.1, "saturation": 0.1}, tgt)
    assert a["applied"] and a["brightness"] == normalize.MAX_BRIGHTNESS_SHIFT and a["saturation"] == normalize.MAX_SATURATION_GAIN
    b = normalize.adjustment({"luma": 0.49, "saturation": 0.49}, tgt)
    assert not b["applied"] and normalize.eq_filter(b) == ""


def test_measure_clip_distinguishes_dark_and_bright(tmp_path):
    from tests.pipeline.helpers import ff
    ff("-f", "lavfi", "-i", "color=c=0x202020:s=320x180:d=2:r=25", "-pix_fmt", "yuv420p", str(tmp_path / "d.mp4"))
    ff("-f", "lavfi", "-i", "color=c=0xe0e0e0:s=320x180:d=2:r=25", "-pix_fmt", "yuv420p", str(tmp_path / "l.mp4"))
    d, l = normalize.measure_clip(tmp_path / "d.mp4"), normalize.measure_clip(tmp_path / "l.mp4")
    assert l["luma"] - d["luma"] > 0.5


# 7. second footage source -------------------------------------------------------------------------
class _Prov:
    def __init__(self, items=None, fail=False):
        self.items, self.fail, self.calls = items or [], fail, 0

    async def search_footage(self, query, aspect_ratio, max_results):
        self.calls += 1
        if self.fail:
            raise TimeoutError("x")
        return self.items[:max_results]


@pytest.mark.asyncio
async def test_fallback_provider_used_when_primary_is_thin_or_down():
    mk = lambda n: asset(n, "x", "http://x/" + n)  # noqa: E731
    thin, rich = _Prov([mk("p1")]), _Prov([mk("q1"), mk("q2"), mk("q3")])
    s = MultiProviderSearch([("pixabay", thin), ("pexels", rich)], min_results=3)
    got = await s("ocean", "16:9", 5)
    assert [a.id for a in got] == ["p1", "q1", "q2", "q3"] and rich.calls == 1
    summ = s.scene_summary(["ocean"], "pexels")
    assert summ["queries"][0]["returned_by_provider"] == {"pixabay": 1, "pexels": 3} and summ["queries"][0]["fallback_provider_queried"]

    full = _Prov([mk(f"p{i}") for i in range(4)]); spare = _Prov([mk("q")])
    s2 = MultiProviderSearch([("pixabay", full), ("pexels", spare)], min_results=3)
    await s2("ocean", "16:9", 5)
    assert spare.calls == 0                                    # primary was sufficient: no fallback call

    s3 = MultiProviderSearch([("pixabay", _Prov(fail=True)), ("pexels", _Prov([mk("q1")]))], min_results=3)
    assert [a.id for a in await s3("ocean", "16:9", 5)] == ["q1"]
    assert s3.log["ocean"]["errors"] == {"pixabay": "TimeoutError"}

    s4 = MultiProviderSearch([("pixabay", _Prov(fail=True)), ("pexels", _Prov(fail=True))])
    with pytest.raises(TimeoutError):
        await s4("ocean", "16:9", 5)


@pytest.mark.asyncio
async def test_timeline_stores_provider_selection_and_render_block(async_client, two_clip_services):
    r = await async_client.post(URL, json={"script": SCRIPT})
    body = await _wait(async_client, r.json()["job_id"])
    assert body["status"] == "completed", body
    tl = (await async_client.get(f"/api/v1/jobs/{r.json()['job_id']}/files/timeline.json")).json()
    sc = tl["scenes"][0]
    assert sc["clip"] and "provider_selection" in sc and tl["render"]["fps"] == 30 and "min_scene_seconds" in tl["render"]


# 8. swap-a-clip -----------------------------------------------------------------------------------
def _probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height,r_frame_rate:format=duration",
                          "-of", "json", str(path)], capture_output=True, text=True).stdout
    return json.loads(out)


@pytest.mark.asyncio
async def test_swap_rerenders_only_the_scene_and_records_timings(async_client, two_clip_services):
    root, _ = two_clip_services
    r = await async_client.post(URL, json={"script": SCRIPT, "aspect_ratios": ["16:9"]})
    jid = r.json()["job_id"]
    assert (await _wait(async_client, jid))["status"] == "completed"
    jd = jobs_endpoint._services()[1].job_dir(jid)
    rd = jd / "r_16x9" if (jd / "r_16x9").is_dir() else jd
    before = json.loads((rd / "timeline.json").read_text())["scenes"][0]
    before_clip = before["clip"]["clip_id"]
    video_before = (rd / "video.mp4").read_bytes()

    s = await async_client.post(f"{URL}/{jid}/scenes/scene-001/swap", json={})
    assert s.status_code == 200, s.text
    res = s.json()
    assert res["rendering"]["mode"] == "scene_only" and res["rendering"]["frames_reencoded"] < res["rendering"]["frames_total"] + 1
    print("SWAP_TIMINGS", json.dumps(res["rendering"]))
    assert res["rendering"]["original_full_render_seconds"] is not None
    after = json.loads((rd / "timeline.json").read_text())["scenes"][0]
    assert after["clip"]["clip_id"] != before_clip
    assert after.get("swap_history")
    assert (rd / "video.mp4").read_bytes() != video_before
    info = _probe(rd / "video.mp4")
    kinds = {st["codec_type"] for st in info["streams"]}
    assert kinds == {"video", "audio"} and abs(float(info["format"]["duration"]) - float(after["end"])) < 0.2


@pytest.mark.asyncio
async def test_swap_on_multi_ratio_job_updates_only_that_ratio(async_client, two_clip_services):
    r = await async_client.post(URL, json={"script": SCRIPT, "aspect_ratios": ["16:9", "1:1"]})
    jid = r.json()["job_id"]
    assert (await _wait(async_client, jid))["status"] == "completed"
    jd = jobs_endpoint._services()[1].job_dir(jid)
    wide_before = (jd / "video_16x9.mp4").read_bytes()
    s = await async_client.post(f"{URL}/{jid}/scenes/scene-001/swap", json={"aspect_ratio": "1:1"})
    assert s.status_code == 200, s.text
    assert (jd / "video_16x9.mp4").read_bytes() == wide_before
    assert (jd / "video_1x1.mp4").read_bytes() == (jd / "r_1x1" / "video.mp4").read_bytes()
    assert json.loads((jd / "timeline_1x1.json").read_text())["scenes"][0].get("swap_history")


@pytest.mark.asyncio
async def test_swap_rejects_bad_requests(async_client, two_clip_services):
    assert (await async_client.post(f"{URL}/aaaaaaaaaaaa/scenes/scene-001/swap", json={})).status_code == 404
    r = await async_client.post(URL, json={"script": SCRIPT})
    jid = r.json()["job_id"]
    await _wait(async_client, jid)
    assert (await async_client.post(f"{URL}/{jid}/scenes/scene-99/swap", json={})).status_code in (404, 422)
    assert (await async_client.post(f"{URL}/{jid}/scenes/..%2Fx/swap", json={})).status_code == 404


@pytest.mark.asyncio
async def test_swap_in_multi_scene_job_reencodes_only_one_scene(async_client, monkeypatch):
    root = Path(tempfile.mkdtemp())
    s = make_settings(root, min_scene_seconds=2.0)
    srv = root / "srv"; srv.mkdir()
    for i, pat in enumerate(["testsrc2", "smptebars", "testsrc", "rgbtestsrc", "yuvtestsrc"]):
        make_clip(srv / f"c{i}.mp4", 6, size="1280x720", pattern=pat)
    server = LocalFileServer(srv).__enter__()
    topics = ["turtle", "farmer", "teacher", "doctor", "pilot"]
    entries = [asset(f"{t}-{k}", f"{t}, {t}s", server.url(f"c{(i + k) % 5}.mp4"), duration=6, width=1280, height=720)
               for i, t in enumerate(topics) for k in (0, 1)]
    script = ("A turtle swims in the sea slowly. A farmer waters the field every morning. "
              "A teacher explains an experiment to students. A doctor checks a patient carefully.")
    store = JobStore(s)
    runner = JobRunner(store, lambda req: Deps(settings=s, search=CatalogSearch(entries), footage_provider_name="pixabay",
                                               get_tts=lambda: PlaceholderToneTTS(), get_transcriber=lambda: FakeTranscriber(script)))
    monkeypatch.setattr(jobs_endpoint, "_services", lambda: (s, store, runner))
    try:
        r = await async_client.post(URL, json={"script": script})
        jid = r.json()["job_id"]
        assert (await _wait(async_client, jid))["status"] == "completed"
        rd = store.job_dir(jid)
        tl0 = json.loads((rd / "timeline.json").read_text())["scenes"]
        assert len(tl0) >= 3
        seg_before = {p.name: p.read_bytes() for p in (rd / "segments").glob("*") if p.is_file()}
        res = (await async_client.post(f"{URL}/{jid}/scenes/scene-002/swap", json={})).json()
        rend = res["rendering"]
        print("SWAP_MULTI", json.dumps(rend))
        assert rend["frames_reencoded"] < rend["frames_total"]
        tl1 = json.loads((rd / "timeline.json").read_text())["scenes"]
        assert tl1[1]["clip"]["clip_id"] != tl0[1]["clip"]["clip_id"]
        assert [x["clip"]["clip_id"] for i, x in enumerate(tl1) if i != 1] == [x["clip"]["clip_id"] for i, x in enumerate(tl0) if i != 1]
        changed = [n for n, b in seg_before.items() if (rd / "segments" / n).read_bytes() != b]
        assert len(changed) == 1
    finally:
        server.__exit__(None, None, None)
