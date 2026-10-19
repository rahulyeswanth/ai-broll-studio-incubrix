import tempfile
import unittest
from pathlib import Path

from app.pipeline.matching import SemanticMatcher
from app.pipeline.models import Scene, WordTiming
from app.pipeline.ranking import filter_candidates, score_candidates
from app.pipeline.provider_adapter import candidate_from_asset
from app.pipeline.scene_analysis import RuleBasedSceneAnalyzer
from app.pipeline.visual import VisualVerifier

from .helpers import CatalogSearch, asset, make_settings

URL = "http://localhost/x.mp4"
FARMER = "A farmer inspects damaged crops after a pest infestation."


def scene(text=FARMER, i=1, dur=4.0):
    return Scene(f"scene-{i:03d}", i, text, 0.0, dur)


def run(coro):
    import asyncio
    return asyncio.run(coro)


class RankingTests(unittest.TestCase):
    def setUp(self):
        self.s = make_settings(Path(tempfile.gettempdir()))
        self.an = RuleBasedSceneAnalyzer().analyze(FARMER)

    def test_relevant_clip_outranks_generic_and_weak_is_below_threshold(self):
        good = candidate_from_asset(asset("1", "farmer, crops, field, pest, leaves", URL))
        generic = candidate_from_asset(asset("2", "nature, landscape, sky, clouds", URL))
        scored = score_candidates([generic, good], self.an, "16:9", 4, self.s)
        self.assertEqual(scored[0].candidate.clip_id, "1")
        self.assertTrue(scored[0].passed)
        self.assertFalse(scored[1].passed)
        self.assertEqual(scored[0].method, "metadata_text_fallback")
        self.assertFalse(scored[0].visual_verified)

    def test_avoid_terms_penalise(self):
        plain = candidate_from_asset(asset("1", "farmer, field, crops", URL))
        cartoon = candidate_from_asset(asset("2", "farmer, field, crops, cartoon", URL))
        sc = {x.candidate.clip_id: x for x in score_candidates([plain, cartoon], self.an, "16:9", 4, self.s)}
        self.assertGreater(sc["1"].final_score, sc["2"].final_score)
        self.assertTrue(sc["2"].penalties)

    def test_filter_removes_unusable_duplicates_and_policy_violations(self):
        s = make_settings(Path(tempfile.gettempdir()), require_https=True, allowed_download_hosts=("cdn.pixabay.com",))
        ok = "https://cdn.pixabay.com/a.mp4"
        cands = [
            candidate_from_asset(asset("ok", "farmer", ok)),
            candidate_from_asset(asset("ok", "farmer", ok)),                                   # duplicate
            candidate_from_asset(asset("http", "farmer", "http://cdn.pixabay.com/b.mp4")),    # not https
            candidate_from_asset(asset("host", "farmer", "https://evil.example/c.mp4")),      # host not allowed
            candidate_from_asset(asset("short", "farmer", "https://cdn.pixabay.com/d.mp4", duration=0.4)),
            candidate_from_asset(asset("lowres", "farmer", "https://cdn.pixabay.com/e.mp4", width=200, height=100)),
            candidate_from_asset(asset("nc", "farmer", "https://cdn.pixabay.com/f.mp4", commercial=False)),
            candidate_from_asset(asset("", "farmer", "https://cdn.pixabay.com/g.mp4")),         # no id
        ]
        res = filter_candidates(cands, s)
        self.assertEqual([c.clip_id for c in res.kept], ["ok"])
        reasons = " | ".join(r for _, r in res.rejected)
        for expected in ("duplicate", "HTTPS", "allow-list", "too short", "resolution", "NOT allowed", "missing id"):
            self.assertIn(expected, reasons)

    def test_ai_generated_flagged_clips_excluded(self):
        a = asset("ai", "farmer", URL)
        a.ai_generated = True
        res = filter_candidates([candidate_from_asset(a)], self.s)
        self.assertEqual(res.kept, [])


class MatcherTests(unittest.TestCase):
    def setUp(self):
        self.s = make_settings(Path(tempfile.gettempdir()))

    def test_selects_best_and_records_decision(self):
        search = CatalogSearch([asset("g", "nature, sky, clouds", URL), asset("f", "farmer, crops, field, pest", URL)])
        dec = run(SemanticMatcher(search, self.s).match_scene(scene(), "16:9", set()))
        self.assertEqual(dec.status, "matched")
        self.assertEqual(dec.selected.candidate.clip_id, "f")
        self.assertTrue(dec.passed_threshold)
        d = dec.to_dict()
        for key in ("scene_id", "narration", "analysis", "queries_tried", "candidate_count", "status", "reason",
                    "passed_threshold", "scoring_method", "retry_attempts", "warnings", "selected"):
            self.assertIn(key, d)
        self.assertEqual(dec.narration, FARMER)
        self.assertIn("title/tags only", " ".join(dec.warnings))        # honest about no visual check

    def test_weak_matches_rejected_then_disclosed_fallback(self):
        search = CatalogSearch([asset("a", "farmer, sunset", URL), asset("b", "field, tractor", URL)])
        dec = run(SemanticMatcher(search, self.s).match_scene(scene(), "16:9", set()))
        self.assertEqual(dec.status, "fallback_low_confidence")
        self.assertFalse(dec.passed_threshold)
        self.assertIn("may NOT accurately depict", " ".join(dec.warnings))
        self.assertGreater(len(dec.attempts), 1)   # retried with alternative queries first

    def test_fail_policy_marks_unmatched(self):
        s = make_settings(Path(tempfile.gettempdir()), fallback_policy="fail")
        search = CatalogSearch([asset("a", "sunset, beach", URL)])
        # query must still return something to be weak: reuse token 'farmer' only in title irrelevant? use 'field'
        search.entries[0].title = "field, sunset"
        dec = run(SemanticMatcher(search, s).match_scene(scene(), "16:9", set()))
        self.assertEqual(dec.status, "unmatched")
        self.assertIsNone(dec.selected)

    def test_retries_alternative_queries_when_first_returns_nothing(self):
        # only matches a broad token that appears in later (broader) queries, not in the specific first one
        class Picky(CatalogSearch):
            async def __call__(self, query, aspect, n):
                self.calls.append(query)
                if len(query.split()) > 3:
                    return []
                return await super().__call__(query, aspect, n)
        search = Picky([asset("f", "farmer, crops, field, pest, leaves", URL)])
        dec = run(SemanticMatcher(search, self.s).match_scene(scene(), "16:9", set()))
        self.assertEqual(dec.status, "matched")
        self.assertGreaterEqual(dec.retry_attempts, 1)
        self.assertEqual(dec.attempts[0].returned, 0)

    def test_provider_outage_reported_not_hidden(self):
        dec = run(SemanticMatcher(CatalogSearch([], always_fail=True), self.s).match_scene(scene(), "16:9", set()))
        self.assertEqual(dec.status, "provider_error")
        self.assertIsNone(dec.selected)
        self.assertTrue(all(a.error for a in dec.attempts))

    def test_no_candidates_status(self):
        dec = run(SemanticMatcher(CatalogSearch([]), self.s).match_scene(scene(), "16:9", set()))
        self.assertEqual(dec.status, "no_candidates")

    def test_duplicate_clip_avoided_when_alternative_exists(self):
        entries = [asset("a", "farmer, crops, field, pest", URL), asset("b", "farmer, crops, field", URL)]
        scenes = [scene(FARMER, 1), scene("A farmer examines crops in a field.", 2)]
        decs = run(SemanticMatcher(CatalogSearch(entries), self.s).match_scenes(scenes, "16:9"))
        keys = [d.selected.candidate.key for d in decs]
        self.assertEqual(len(set(keys)), 2)
        self.assertFalse(any(d.reused_clip for d in decs))

    def test_duplicate_reuse_is_disclosed_when_no_alternative(self):
        entries = [asset("only", "farmer, crops, field, pest", URL)]
        scenes = [scene(FARMER, 1), scene(FARMER, 2)]
        decs = run(SemanticMatcher(CatalogSearch(entries), self.s).match_scenes(scenes, "16:9"))
        self.assertTrue(decs[1].reused_clip)
        self.assertIn("reused", " ".join(decs[1].warnings))

    def test_reuse_can_be_disabled(self):
        s = make_settings(Path(tempfile.gettempdir()), allow_clip_reuse=False, fallback_policy="fail")
        entries = [asset("only", "farmer, crops, field, pest", URL)]
        decs = run(SemanticMatcher(CatalogSearch(entries), s).match_scenes([scene(FARMER, 1), scene(FARMER, 2)], "16:9"))
        self.assertEqual(decs[0].status, "matched")
        self.assertEqual(decs[1].status, "unmatched")

    def test_visual_verifier_scores_used_and_labelled_only_when_real(self):
        class FakeVisual(VisualVerifier):       # test double standing in for a VLM; reports "visual" scoring
            name, performs_visual_analysis, version = "fake-vlm", True, "test"
            def score(self, analysis, candidates, fetch_preview):
                return {c.key: (0.95 if c.clip_id == "wrongtags" else 0.05) for c in candidates}
        entries = [asset("goodtags", "farmer, crops, field, pest", URL), asset("wrongtags", "farmer, crops, field", URL)]
        m = SemanticMatcher(CatalogSearch(entries), self.s, verifier=FakeVisual(), fetch_preview=lambda c, d: Path("x"))
        dec = run(m.match_scene(scene(), "16:9", set()))
        self.assertEqual(dec.selected.candidate.clip_id, "wrongtags")      # visual evidence overrides tag text
        self.assertEqual(dec.scoring_method, "visual_clip+metadata_text")
        self.assertTrue(dec.selected.visual_verified)

    def test_verifier_crash_degrades_to_fallback_with_warning(self):
        class Broken(VisualVerifier):
            name, performs_visual_analysis = "broken", True
            def score(self, *a):
                raise RuntimeError("gpu on fire")
        entries = [asset("f", "farmer, crops, field, pest", URL)]
        m = SemanticMatcher(CatalogSearch(entries), self.s, verifier=Broken(), fetch_preview=lambda c, d: None)
        dec = run(m.match_scene(scene(), "16:9", set()))
        self.assertEqual(dec.scoring_method, "metadata_text_fallback")
        self.assertIn("Visual verifier failed", " ".join(dec.warnings))


if __name__ == "__main__":
    unittest.main()
