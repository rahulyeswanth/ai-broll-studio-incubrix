"""Shared test doubles. NOTHING here is a production provider; test doubles are always labelled as such."""
from __future__ import annotations

import http.server
import shutil
import socketserver
import subprocess
import sys
import tempfile
import threading
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

from app.pipeline.alignment import distribute_words, tokenize
from app.pipeline.config import PipelineSettings
from app.pipeline.models import WordTiming
from app.pipeline.transcription import ASRResult, PipelineTranscriber


def ff(*args: str) -> None:
    cp = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], capture_output=True, text=True)
    if cp.returncode != 0:
        raise RuntimeError(cp.stderr)


def make_clip(path: Path, seconds: float = 4, size: str = "640x360", rate: int = 25, pattern: str = "testsrc2") -> Path:
    ff("-f", "lavfi", "-i", f"{pattern}=s={size}:d={seconds}:r={rate}", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path))
    return path


def make_speechlike_audio(path: Path, seconds: float = 8.0) -> Path:
    """Amplitude-modulated tone: audible (not silent), decodable, any extension supported by ffmpeg."""
    codec = {".mp3": ["-c:a", "libmp3lame"], ".wav": ["-c:a", "pcm_s16le"], ".m4a": ["-c:a", "aac"]}.get(path.suffix.lower(), [])
    ff("-f", "lavfi", "-i", f"sine=frequency=220:duration={seconds}", "-af", "tremolo=f=4:d=0.8,volume=0.6", *codec, str(path))
    return path


def make_silent_audio(path: Path, seconds: float = 3.0) -> Path:
    ff("-f", "lavfi", "-i", f"anullsrc=r=44100:cl=mono", "-t", str(seconds), str(path))
    return path


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):  # silence
        pass

    def handle(self):
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError):
            pass  # client aborted (size-limit tests)


class LocalFileServer:
    def __init__(self, directory: Path):
        handler = lambda *a, **k: QuietHandler(*a, directory=str(directory), **k)  # noqa: E731
        socketserver.TCPServer.allow_reuse_address = True
        self.httpd = socketserver.ThreadingTCPServer(("127.0.0.1", 0), handler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *a):
        self.httpd.shutdown()
        self.httpd.server_close()

    def url(self, name: str) -> str:
        return f"http://127.0.0.1:{self.port}/{name}"


def asset(clip_id: str, tags: str, url: str, *, duration: float = 4.0, width: int = 640, height: int = 360,
          source: str = "pixabay", commercial: bool = True, author: str = "tester") -> SimpleNamespace:
    """Looks like the project's BrollAsset (attribute access only)."""
    return SimpleNamespace(id=clip_id, source=source, title=tags, download_url=url, preview_url=None, aspect_ratio="16:9",
                           duration=duration, width=width, height=height, commercial_use_allowed=commercial,
                           license_type="x", attribution_required=False, author_name=author, author_url=None)


class CatalogSearch:
    """Fake footage search: returns catalog entries sharing any (stemmed) token with the query."""
    def __init__(self, entries: List[SimpleNamespace], fail_first: int = 0, always_fail: bool = False):
        self.entries, self.calls, self.fail_first, self.always_fail = entries, [], fail_first, always_fail

    async def __call__(self, query: str, aspect: str, n: int):
        self.calls.append(query)
        if self.always_fail or len(self.calls) <= self.fail_first:
            raise RuntimeError("simulated provider outage")
        toks = {t.rstrip("s") for t in query.lower().replace("-", " ").split()}
        out = [e for e in self.entries if toks & {t.strip().rstrip("s") for t in e.title.lower().replace(",", " ").split()}]
        return out[:n]


class FakeTranscriber(PipelineTranscriber):
    """Deterministic test double for speech-to-text; ``is_real`` is False so the manifest says so."""
    name, is_real, model, version = "fake-test-asr", False, "fixture", None

    def __init__(self, text: str, granularity: str = "word", lead_in: float = 0.4, tail_out: float = 0.4):
        self.text, self.granularity, self.lead_in, self.tail_out = text, granularity, lead_in, tail_out

    async def transcribe(self, audio_path: Path, language: Optional[str]) -> ASRResult:
        from app.pipeline.media import audio_duration
        dur = audio_duration(audio_path)
        words = distribute_words(tokenize(self.text), self.lead_in, dur - self.tail_out)
        return ASRResult(self.text, words, self.granularity, "en", 0.99, [])


def install_fake_edge_tts(monkeypatch_dir: Path, *, fail: bool = False, no_audio: bool = False, words: bool = True) -> types.ModuleType:
    """Inject a stub ``edge_tts`` module (no network). The stub emits genuine MP3 audio generated with FFmpeg
    plus WordBoundary events, so EdgeTTSProvider's code path is exercised end-to-end except the network hop."""
    mod = types.ModuleType("edge_tts")

    class Communicate:
        def __init__(self, text, voice, boundary=None, **kw):
            self.text, self.voice, self.boundary = text, voice, boundary

        async def stream(self):
            if fail:
                raise ConnectionError("simulated network failure")
            if no_audio:
                return
            toks = self.text.split()
            dur = max(1.2, 0.35 * len(toks))
            tmp = monkeypatch_dir / f"_edge_{abs(hash(self.text))}.mp3"
            make_speechlike_audio(tmp, dur)
            yield {"type": "audio", "data": tmp.read_bytes()}
            if words:
                step = (dur - 0.2) / max(len(toks), 1)
                for i, t in enumerate(toks):
                    yield {"type": "WordBoundary", "offset": int((0.1 + i * step) * 1e7), "duration": int(step * 0.9 * 1e7), "text": t}
    mod.Communicate = Communicate
    sys.modules["edge_tts"] = mod
    return mod


def uninstall_fake_edge_tts() -> None:
    sys.modules.pop("edge_tts", None)


def make_settings(root: Path, **over: Any) -> PipelineSettings:
    s = PipelineSettings(storage_root=root, require_https=False, allowed_download_hosts=(), allow_mock_providers=True,
                         min_scene_seconds=1.0, max_scene_seconds=12.0, caption_min_seconds=0.5)
    for k, v in over.items():
        setattr(s, k, v)
    return s
