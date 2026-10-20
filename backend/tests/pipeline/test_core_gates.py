"""Anti-nonsense and sensitive-story rules (metadata only; no network)."""
from app.pipeline.config import PipelineSettings
from app.pipeline.models import Candidate
from app.pipeline.ranking import score_candidates
from app.pipeline.scene_analysis import RuleBasedSceneAnalyzer


def _cand(cid, title, tags):
    return Candidate("test", cid, f"https://example.com/{cid}.mp4", title=title, tags=tags.split(", "),
                     duration=8.0, width=1920, height=1080)


def _best(text, entries):
    a = RuleBasedSceneAnalyzer().analyze(text)
    scored = score_candidates([_cand(i, "", t) for i, t in entries], a, "16:9", 5.0, PipelineSettings())
    return scored[0], {s.candidate.clip_id: s for s in scored}


def test_gibberish_with_one_accidental_word_is_unmatched():
    s = PipelineSettings()
    top, _ = _best("Zorblax quantum flibbertigibbet snorfs wibbly.", [("neb", "nebula, quantum, wallpaper, space")])
    assert not top.passed and top.final_score < s.review_floor and "too weak" in (top.gated or "")


def test_named_brand_uses_generic_standin_not_unrelated_clip():
    text = "Elon Musk announced a new Tesla update."
    a = RuleBasedSceneAnalyzer().analyze(text)
    assert "electric car charging" in a.queries
    top, by = _best(text, [("rein", "reindeer, pattern, christmas, new, year"), ("car", "electric car, charging, tesla, technology")])
    assert by["rein"].final_score < PipelineSettings().review_floor
    assert top.candidate.clip_id == "car"


def test_sad_story_rejects_upbeat_clips():
    text = "Families lost their homes and many people are still missing."
    a = RuleBasedSceneAnalyzer().analyze(text)
    assert a.sensitive and a.queries[0].endswith("aftermath")
    _, by = _best(text, [("laptop", "woman, laptop, home, working, smile"), ("lawn", "students, relaxing, grass, happy, home"),
                         ("flood", "flood, damaged, houses, aftermath, disaster")])
    floor = PipelineSettings().review_floor
    assert by["laptop"].final_score < floor and by["lawn"].final_score < floor
    assert by["flood"].final_score >= floor


def test_normal_scene_is_not_over_blocked():
    top, _ = _best("Farmers pick coffee cherries by hand on a green hillside.",
                   [("c", "coffee, cherries, farmer, harvest, hillside")])
    assert top.passed
