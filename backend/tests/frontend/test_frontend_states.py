"""Streamlit AppTest: renders frontend.py with mocked HTTP. Covers empty, offline, failed, completed, long-text states."""
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from .fixtures import LONG, bundle

ROOT = Path(__file__).resolve().parents[3]

JID = "abc123def456"


class Resp:
    def __init__(self, data=None, content=b"", status=200):
        self.ok, self.status_code, self._d, self.content, self.text = status < 400, status, data, content, json.dumps(data) if data is not None else ""
        self.headers = {"content-type": "application/json" if data is not None else "video/mp4"}

    def json(self):
        if self._d is None:
            raise ValueError
        return self._d


def install(monkeypatch, online=True, manifest=None, timeline=None):
    def fake_get(url, timeout=None, **kw):
        if not online:
            import requests
            raise requests.ConnectionError("offline")
        u = url.split("/api/v1")[-1]
        if u == "/health":
            return Resp({"status": "healthy"})
        if u == "/jobs/config":
            return Resp({"pixabay_key_configured": True, "pexels_key_configured": False, "tts_provider": "edge", "visual_verifier": "metadata",
                         "match_threshold": 0.45, "review_floor": 0.3, "fallback_policy": "review"})
        if u == "/jobs/voices":
            return Resp({"voices": [{"id": "en-US-AriaNeural"}]})
        if u.endswith("manifest.json"):
            return Resp(manifest)
        if u.endswith("timeline.json"):
            return Resp(timeline)
        return Resp(None, content=b"")
    monkeypatch.setattr("requests.get", fake_get)


def app():
    at = AppTest.from_file(str(ROOT / "frontend.py"), default_timeout=60)
    return at


def text_of(at):
    return " ".join(m.value for m in at.markdown)


def test_empty_state(monkeypatch):
    install(monkeypatch)
    at = app().run()
    assert not at.exception
    assert "Your video will appear here" in text_of(at)
    assert [b for b in at.button if b.label == "Generate Video"][0].disabled is False


def test_backend_offline_disables_generate(monkeypatch):
    install(monkeypatch, online=False)
    at = app().run()
    assert not at.exception
    assert "offline" in text_of(at)
    assert [b for b in at.button if b.label == "Generate Video"][0].disabled is True


def test_empty_script_shows_inline_validation(monkeypatch):
    install(monkeypatch)
    at = app().run()
    [b for b in at.button if b.label == "Generate Video"][0].click()
    at.run()
    assert "Enter a script first." in text_of(at)


def test_busy_state_disables_inputs(monkeypatch):
    install(monkeypatch)
    at = app()
    at.session_state["wf_job_id"] = JID
    at.run()
    assert [b for b in at.button if b.label == "Generate Video"][0].disabled is True


def test_failed_job_shows_friendly_message_and_collapsed_details(monkeypatch):
    install(monkeypatch)
    at = app()
    at.session_state["wf_job"] = {"job_id": JID, "status": "failed", "error": {"code": "provider_unavailable", "message": "Pixabay API key is not configured. " + LONG,
                                                                                 "user_message": "The footage provider is not configured.", "details": {}}}
    at.run()
    assert not at.exception
    assert "The footage provider is not configured." in text_of(at)
    ex = [e for e in at.expander if e.label == "Generation details"]
    assert ex and ex[0].proto.expanded is False


def test_completed_job_summary_review_and_diagnostics_collapsed(monkeypatch):
    manifest, timeline = bundle()
    install(monkeypatch, manifest=manifest, timeline=timeline)
    at = app()
    at.session_state["wf_job"] = {"job_id": JID, "status": "completed", "files": {}}
    at.run()
    assert not at.exception
    t = text_of(at)
    assert "1 scene has no matching footage." in t and "Some footage could not be verified visually." in t
    assert "Matched" in t and "Needs review" in t and "No footage" in t
    assert "Video generated successfully." not in t                      # not claimed when a scene is unmatched
    ex = [e for e in at.expander if e.label == "Generation details"]
    assert ex and ex[0].proto.expanded is False
    assert t.count("1 scene has no matching footage.") == 1             # shown once on the main screen


def test_long_scene_text_does_not_break_rendering(monkeypatch):
    manifest, timeline = bundle(("matched", "matched"), text=LONG)
    install(monkeypatch, manifest=manifest, timeline=timeline)
    at = app()
    at.session_state["wf_job"] = {"job_id": JID, "status": "completed", "files": {}}
    at.run()
    assert not at.exception
    assert "clamp-2" in text_of(at)


def test_legacy_pages_still_render(monkeypatch):
    install(monkeypatch)
    for page in ("overview", "script", "broll", "narration", "captions", "render"):
        at = app()
        at.session_state["page"] = page
        at.run()
        assert not at.exception, page
