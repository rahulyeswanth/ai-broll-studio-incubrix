import ui_helpers as ui

from .fixtures import LONG, bundle, scene


def test_scene_status_and_counts():
    _, tl = bundle(("matched", "review", "unmatched", "matched"))
    assert [ui.scene_status(s) for s in tl["scenes"]] == ["matched", "review", "unmatched", "matched"]
    assert ui.counts(tl["scenes"]) == {"all": 4, "matched": 2, "review": 1, "unmatched": 1}


def test_filters_accept_labels_with_counts():
    _, tl = bundle(("matched", "review", "unmatched"))
    s = tl["scenes"]
    assert len(ui.filter_scenes(s, "All (3)")) == 3
    assert [x["index"] for x in ui.filter_scenes(s, "Needs review (1)")] == [2]
    assert [x["index"] for x in ui.filter_scenes(s, "No footage (1)")] == [3]
    assert ui.filter_scenes(s, "Bogus") == []


def test_relevance_is_label_plus_score_never_a_percentage():
    label, small = ui.relevance_text(scene(1, "matched"))
    assert label == "High" and "0.97" in small and "%" not in small and "title/tags" in small
    assert ui.relevance_text(scene(1, "unmatched"))[0] == "None"


def test_html_is_escaped_and_long_text_clamped():
    assert "<script>" not in ui.esc("<script>alert(1)</script>")
    assert len(ui.clamp(LONG, 100)) <= 100 and ui.clamp(LONG, 100).endswith("…")
    _, tl = bundle(text="<img src=x onerror=alert(1)>")
    assert "<img" not in ui.timeline_html(tl["scenes"], 1)


def test_timeline_is_proportional_and_marks_selection():
    sc = [scene(1, "matched", start=0, end=2), scene(2, "unmatched", start=2, end=8)]
    html = ui.timeline_html(sc, 2)
    assert "flex:2.000" in html and "flex:6.000" in html
    assert html.count("is-selected") == 1 and "tone-bad" in html
    assert ui.timeline_html([], 1) == ""


def test_stage_tracking_and_scene_progress():
    assert ui.stage_for(10)[0] == 0 and ui.stage_for(65)[0] == 2 and ui.stage_for(100)[0] == 5
    assert ui.scene_progress("Prepared footage for scene 3/5") == (3, 5) and ui.scene_progress("Rendering") is None


def test_group_diagnostics_dedupes():
    it = {"code": "a", "scene_id": "s", "message": "m", "severity": "warning"}
    assert len(ui.group_diagnostics([it, dict(it)])["warning"]) == 1


def test_job_error_text_prefers_user_message():
    assert ui.job_error_text({"error": {"user_message": "Friendly", "message": "raw"}}) == ("Friendly", "raw")
    assert ui.job_error_text({})[0] == ui.ERROR_FALLBACK
