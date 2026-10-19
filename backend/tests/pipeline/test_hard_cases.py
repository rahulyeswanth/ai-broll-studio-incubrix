"""Hard semantic cases. Provider = labelled test catalogue; scoring = the production metadata scorer (no visual check)."""
import asyncio
import tempfile
from pathlib import Path

import pytest

from app.pipeline.matching import SemanticMatcher, expand_avoid, violates_avoid
from app.pipeline.models import Scene
from app.pipeline.provider_adapter import candidate_from_asset
from tests.pipeline.helpers import CatalogSearch, asset, make_settings

URL = "http://127.0.0.1:1/x.mp4"

CASES = [
    ("cloud_computing", "We moved our data to the cloud and cut server costs.",
     [("sky", "sky, clouds, nature, blue, weather"), ("dc", "server, data center, cloud computing, technology, network")], "dc"),
    ("river_bank", "Anglers sat on the river bank at dawn.",
     [("fin", "bank, money, finance, building, business"), ("river", "river, bank, water, fishing, nature, dawn")], "river"),
    ("financial_bank", "The bank approved our small business loan.",
     [("river", "river, bank, water, fishing, nature"), ("fin", "bank, loan, finance, business, money, office")], "fin"),
    ("computer_mouse", "Click the mouse to open the file.",
     [("animal", "mouse, animal, rodent, cheese, cute"), ("pc", "computer mouse, click, keyboard, desk, office")], "pc"),
    ("python_language", "Learn Python programming to automate tasks.",
     [("snake", "python, snake, reptile, wildlife, jungle"), ("code", "python, programming, code, laptop, software, developer")], "code"),
    ("python_snake", "A python slithered through the jungle undergrowth.",
     [("code", "python, programming, code, laptop, software"), ("snake", "python, snake, reptile, jungle, wildlife")], "snake"),
    ("value_abstract", "Our product delivers real value to customers.",
     [("cash", "money, cash, dollars, value, pile"), ("team", "customers, product, team, handshake, business, service")], "team"),
    ("ocean_protection", "Volunteers clean plastic waste from the ocean beach.",
     [("party", "beach, party, summer, drinks"), ("clean", "volunteers, cleaning, plastic, waste, ocean, beach, environment")], "clean"),
]


def _run(text, entries):
    s = make_settings(Path(tempfile.gettempdir()))
    cat = [asset(i, tags, f"http://127.0.0.1:1/{i}.mp4") for i, tags in entries]
    dec = asyncio.run(SemanticMatcher(CatalogSearch(cat), s).match_scene(Scene("scene-001", 1, text, 0.0, 5.0), "16:9", set()))
    return dec


@pytest.mark.parametrize("name,text,entries,expected", CASES, ids=[c[0] for c in CASES])
def test_meaning_beats_keyword(name, text, entries, expected):
    dec = _run(text, entries)
    assert dec.selected is not None, (name, dec.reason)
    assert dec.selected.candidate.clip_id == expected, (name, [(r.candidate.clip_id, round(r.final_score, 2)) for r in dec.ranked])
    assert dec.selected.final_score > 0 and dec.scoring_method == "metadata_text_fallback"   # honest: no visual analysis
    assert dec.visual_verifier == "none"


def test_no_suitable_footage_is_not_reported_as_matched():
    dec = _run("Quarterly revenue hit 4.2 million dollars.", [("x", "kitten, cute, pets, sofa")])
    assert dec.status != "matched" and not dec.passed_threshold


def test_avoid_rules_reject_by_metadata():
    words = expand_avoid(["people's faces", "alcohol"])
    assert {"face", "faces", "people", "beer", "wine"} <= words
    ok = candidate_from_asset(asset("a", "ocean, waves, sunset", URL))
    bad = candidate_from_asset(asset("b", "friends, drinking beer, bar", URL))
    portrait = candidate_from_asset(asset("c", "woman, portrait, smiling", URL))
    assert violates_avoid(ok, words) is None
    assert violates_avoid(bad, words) and violates_avoid(portrait, words)


def test_avoid_is_applied_by_the_matcher():
    s = make_settings(Path(tempfile.gettempdir()))
    cat = [asset("beer", "bar, beer, friends, celebration, party", URL + "1"), asset("ok", "team, celebrating, launch, party, office, colleagues", URL + "2")]
    dec = asyncio.run(SemanticMatcher(CatalogSearch(cat), s, avoid=["alcohol"]).match_scene(
        Scene("scene-001", 1, "The team celebrated the launch with a party.", 0.0, 5.0), "16:9", set()))
    assert dec.selected and dec.selected.candidate.clip_id == "ok"
    assert any("avoid rule" in r for a in dec.attempts for r in a.rejected)
