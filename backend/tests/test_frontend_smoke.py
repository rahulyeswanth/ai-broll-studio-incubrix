"""Frontend smoke test using Streamlit's AppTest. Skipped when streamlit is not installed.
NOT executed in the authoring sandbox (streamlit not installable offline)."""
from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")
FRONTEND = Path(__file__).resolve().parents[2] / "frontend.py"


def test_frontend_starts_and_workflow_page_renders():
    at = st_testing.AppTest.from_file(str(FRONTEND), default_timeout=30)
    at.session_state["page"] = "workflow"
    at.run()
    assert not at.exception, at.exception
    labels = [b.label for b in at.button]
    assert "Generate Video" in labels


def test_frontend_other_pages_still_render():
    for page in ("overview", "script", "broll", "narration", "captions", "render"):
        at = st_testing.AppTest.from_file(str(FRONTEND), default_timeout=30)
        at.session_state["page"] = page
        at.run()
        assert not at.exception, (page, at.exception)


def test_frontend_reports_offline_backend_without_crashing(monkeypatch):
    monkeypatch.setenv("BROLL_API_URL", "http://127.0.0.1:9/api/v1")   # nothing listens here
    at = st_testing.AppTest.from_file(str(FRONTEND), default_timeout=30)
    at.session_state["page"] = "workflow"
    at.run()
    assert not at.exception
