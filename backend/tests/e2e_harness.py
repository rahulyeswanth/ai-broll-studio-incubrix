"""TEST-ONLY harness: runs the real FastAPI app with a deterministic stand-in for the footage provider and TTS.

    python tests/e2e_harness.py [--port 8000]

It serves locally generated clips from a tiny HTTP server and exposes them through a fake catalogue whose metadata
mimics real stock-footage titles/tags (right clips, right-subject-wrong-action decoys, and a scene with NO footage).
Narration is a placeholder tone (``is_real: false`` in the manifest). It exists so the UI and pipeline can be exercised
without API keys or internet; it is never used by the normal app.
"""
import argparse
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent / "pipeline"))

import uvicorn  # noqa: E402

from app.api.v1.endpoints import jobs as jobs_ep  # noqa: E402
from app.main import app  # noqa: E402
from app.pipeline.narration import PlaceholderToneTTS  # noqa: E402
from app.pipeline.orchestrator import Deps  # noqa: E402
from helpers import CatalogSearch, FakeTranscriber, LocalFileServer, asset, make_clip  # noqa: E402

PATTERNS = {"watering": "testsrc2", "harvest": "testsrc", "consult": "smptebars", "photo": "yuvtestsrc", "board": "rgbtestsrc"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--root", default=None)
    a = ap.parse_args()
    root = Path(a.root or tempfile.mkdtemp(prefix="e2e_"))
    srv = root / "srv"
    srv.mkdir(parents=True, exist_ok=True)
    for name, pat in PATTERNS.items():
        make_clip(srv / f"{name}.mp4", 6, pattern=pat)
    # visually distinct generator for the coding clip (so near-duplicate detection has something to compare)
    import subprocess
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "cellauto=s=640x360:r=25:rule=110,format=yuv420p", "-t", "6",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(srv / "coding.mp4")], check=True)
    server = LocalFileServer(srv).__enter__()
    u = server.url
    catalog = [
        asset("w1", "farmer, watering, tomato, plants, greenhouse", u("watering.mp4"), duration=6),
        asset("h1", "farmer, harvesting, tomatoes, field", u("harvest.mp4"), duration=6),
        asset("c1", "doctor, patient, hospital, consultation, talking", u("consult.mp4"), duration=6),       # right people, wrong action
        asset("k1", "programmer, coding, python, code, laptop", u("coding.mp4"), duration=6),
        asset("p1", "tourist, taking photo, eiffel tower, paris", u("photo.mp4"), duration=6),
        asset("b1", "classroom, blackboard", u("board.mp4"), duration=6),                                    # weak: setting only
    ]

    def build(settings, req):
        settings.require_https, settings.allowed_download_hosts, settings.allow_mock_providers = False, (), True
        return Deps(settings=settings, search=CatalogSearch(catalog), footage_provider_name="pixabay",
                    get_tts=lambda: PlaceholderToneTTS(), get_transcriber=lambda: FakeTranscriber("x"))

    jobs_ep.build_deps = build
    print(f"[e2e-harness] storage={root} clips={srv}", flush=True)
    try:
        uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")
    finally:
        server.__exit__(None, None, None)
        shutil.rmtree(srv, ignore_errors=True)


if __name__ == "__main__":
    main()
