"""Action-aware matching: right clip vs right-subject-wrong-action decoys, thresholds, policies, bad inputs.
Metadata is deterministic and synthetic (no network). Only title/tags are available, so these tests prove what the
metadata scorer can and cannot establish - they do NOT claim visual verification."""
import asyncio
import tempfile
import unittest
from pathlib import Path

from app.pipeline.matching import SemanticMatcher
from app.pipeline.models import Scene
from app.pipeline.provider_adapter import candidate_from_asset
from app.pipeline.ranking import filter_candidates, score_candidates
from app.pipeline.scene_analysis import RuleBasedSceneAnalyzer

from .helpers import CatalogSearch, asset, make_settings

URL = "http://localhost/{}.mp4"

# (narration, clip that shows the action, [(why it is wrong, tags) ...], unrelated clip)
DOMAINS = {
    "farmer": ("A farmer waters tomato plants in a sunny field.", "farmer, watering, tomato, plants, greenhouse",
               [("harvesting", "farmer, harvesting, tomatoes, field"), ("subject only", "farmer, sunset, field")], "city, traffic, night"),
    "doctor": ("A doctor examines a patient and checks the patient's blood pressure.", "doctor, patient, blood pressure, cuff, clinic",
               [("consultation", "doctor, patient, hospital, consultation, talking"), ("corridor", "hospital, corridor, doctor")], "beach, sunset"),
    "engineer": ("A software engineer writes Python code on a laptop.", "programmer, coding, python, code, laptop",
                 [("laptop only", "woman, laptop, coffee, cafe"), ("meeting", "business, meeting, office, laptop, team")], "mountain, river"),
    "teacher": ("A teacher explains a science experiment to students.", "teacher, science experiment, students, classroom, beaker",
                [("blackboard", "teacher, blackboard, classroom, writing"), ("lecture", "professor, lecture, university")], "dog, park"),
    "tourist": ("A tourist photographs the Eiffel Tower in Paris.", "tourist, taking photo, eiffel tower, paris",
                [("landmark only", "eiffel tower, paris, france"), ("aerial", "eiffel tower, aerial, drone, paris")], "forest, lake"),
}


class FixedSearch:
    """Like a real provider: returns the same (possibly poor) hits whatever the query is."""
    def __init__(self, entries):
        self.entries, self.calls = entries, []

    async def __call__(self, query, aspect, n):
        self.calls.append(query)
        return self.entries[:n]


def run(c):
    return asyncio.run(c)


def cands(entries):
    return [candidate_from_asset(a) for a in entries]


def settings(**over):
    return make_settings(Path(tempfile.gettempdir()), **over)


class DomainMatching(unittest.TestCase):
    def setUp(self):
        self.s = settings()
        self.an = RuleBasedSceneAnalyzer()

    def entries(self, key, include_right=True, decoys=True, unrelated=True):
        _, right, wrong, other = DOMAINS[key]
        out = []
        if include_right:
            out.append(asset(f"{key}-right", right, URL.format(f"{key}-right")))
        if decoys:
            out += [asset(f"{key}-wrong{i}", t, URL.format(f"{key}-wrong{i}")) for i, (_, t) in enumerate(wrong)]
        if unrelated:
            out.append(asset(f"{key}-other", other, URL.format(f"{key}-other")))
        return out

    def test_analysis_extracts_action_object_and_alternative_queries(self):
        for key, (text, *_rest) in DOMAINS.items():
            a = self.an.analyze(text)
            with self.subTest(domain=key):
                self.assertTrue(a.subject, "subject")
                self.assertTrue(a.action_terms, "action evidence terms")
                self.assertTrue(a.frame_ids, "activity frame")
                self.assertGreaterEqual(len(a.queries), 4)
                self.assertEqual(len(a.queries), len({q.lower() for q in a.queries}), "no identical queries")
                self.assertTrue(all(len(q.split()) <= 8 for q in a.queries))

    def test_doctor_scene_keeps_both_actions_and_prefers_blood_pressure_queries(self):
        a = self.an.analyze(DOMAINS["doctor"][0])
        self.assertIn("blood pressure", a.visual_description.lower())
        self.assertTrue(any("blood pressure" in q for q in a.queries[:3]), a.queries)
        self.assertIn("patient", [o.split()[0] for o in a.objects])

    def test_correct_action_clip_wins_and_is_matched(self):
        for key, (text, *_r) in DOMAINS.items():
            with self.subTest(domain=key):
                m = SemanticMatcher(CatalogSearch(self.entries(key)), self.s)
                dec = run(m.match_scene(Scene("scene-001", 1, text, 0, 5), "16:9", set()))
                self.assertEqual(dec.status, "matched", dec.reason)
                self.assertEqual(dec.selected.candidate.clip_id, f"{key}-right")
                self.assertIn(dec.selected.label, ("High", "Medium"))
                self.assertTrue(dec.selected.passed)

    def test_right_subject_wrong_action_never_passes(self):
        for key, (text, *_r) in DOMAINS.items():
            with self.subTest(domain=key):
                an = self.an.analyze(text)
                _, _, wrong, _ = DOMAINS[key]
                scored = score_candidates(cands([asset(f"w{i}", t, URL.format(i)) for i, (_, t) in enumerate(wrong)]), an, "16:9", 5, self.s)
                for sc in scored:
                    self.assertFalse(sc.passed, f"{sc.candidate.title} should not pass: {sc.final_score:.2f}")
                    self.assertEqual(sc.label, "Low")

    def test_only_wrong_action_clips_means_review_or_unmatched_not_matched(self):
        for key, (text, *_r) in DOMAINS.items():
            with self.subTest(domain=key):
                m = SemanticMatcher(CatalogSearch(self.entries(key, include_right=False, unrelated=False)), self.s)
                dec = run(m.match_scene(Scene("scene-001", 1, text, 0, 5), "16:9", set()))
                self.assertIn(dec.status, ("fallback_low_confidence", "unmatched"))
                self.assertFalse(dec.passed_threshold)
                if dec.selected:
                    self.assertTrue(dec.needs_review)
                    self.assertTrue(any("NOT" in w or "may" in w for w in dec.warnings))

    def test_unrelated_clip_is_unmatched_under_review_policy(self):
        for key, (text, *_r) in DOMAINS.items():
            with self.subTest(domain=key):
                other = [asset("x", DOMAINS[key][3], URL.format("x"))]
                dec = run(SemanticMatcher(FixedSearch(other), self.s).match_scene(Scene("scene-001", 1, text, 0, 5), "16:9", set()))
                self.assertEqual(dec.status, "unmatched")
                self.assertIsNone(dec.selected)

    def test_wrong_environment_same_action_ranks_below_right_environment(self):
        text = DOMAINS["doctor"][0]
        an = self.an.analyze(text)
        right_env = asset("a", "doctor, blood pressure, cuff, clinic, patient", URL.format("a"))
        wrong_env = asset("b", "blood pressure, cuff, arm, living room, home", URL.format("b"))
        sc = score_candidates(cands([wrong_env, right_env]), an, "16:9", 5, self.s)
        self.assertEqual(sc[0].candidate.clip_id, "a")
        self.assertTrue(sc[1].passed, "same action in another place is still acceptable, just ranked lower")

    def test_missing_metadata_is_not_a_match(self):
        an = self.an.analyze(DOMAINS["farmer"][0])
        blank = asset("blank", "", URL.format("blank"))
        sc = score_candidates(cands([blank]), an, "16:9", 5, self.s)[0]
        self.assertFalse(sc.passed)
        self.assertLess(sc.final_score, self.s.review_floor)

    def test_empty_provider_results(self):
        dec = run(SemanticMatcher(CatalogSearch([]), self.s).match_scene(Scene("scene-001", 1, DOMAINS["farmer"][0], 0, 5), "16:9", set()))
        self.assertIn(dec.status, ("no_candidates", "unmatched"))
        self.assertIsNone(dec.selected)

    def test_duplicate_and_invalid_candidates_are_rejected_with_reasons(self):
        good = asset("1", "farmer, watering, plants", URL.format("1"))
        dup = asset("1", "farmer, watering, plants", URL.format("1"))
        same_url = asset("2", "farmer, watering, plants", URL.format("1"))
        bad_url = asset("3", "farmer, watering, plants", "not a url")
        no_url = asset("4", "farmer, watering, plants", "")
        fr = filter_candidates(cands([good, dup, same_url, bad_url, no_url]), self.s)
        self.assertEqual([c.clip_id for c in fr.kept], ["1"])
        reasons = " ".join(r for _, r in fr.rejected)
        self.assertIn("duplicate", reasons)
        self.assertIn("invalid download URL", reasons)

    def test_no_identical_query_is_sent_twice(self):
        search = CatalogSearch([])
        run(SemanticMatcher(search, self.s).match_scene(Scene("scene-001", 1, DOMAINS["doctor"][0], 0, 5), "16:9", set()))
        norm = [" ".join(q.lower().split()) for q in search.calls]
        self.assertEqual(len(norm), len(set(norm)))
        self.assertGreater(len(norm), 1, "alternative phrasings must be tried when results are empty")

    def test_policy_review_best_available_and_fail(self):
        text = DOMAINS["doctor"][0]
        entries = [asset("c", DOMAINS["doctor"][2][0][1], URL.format("c"))]   # right people, wrong action
        scene = Scene("scene-001", 1, text, 0, 5)
        review = run(SemanticMatcher(CatalogSearch(entries), settings(fallback_policy="review")).match_scene(scene, "16:9", set()))
        self.assertEqual(review.status, "fallback_low_confidence")
        fail = run(SemanticMatcher(CatalogSearch(entries), settings(fallback_policy="fail")).match_scene(scene, "16:9", set()))
        self.assertEqual(fail.status, "unmatched")
        junk = [asset("j", "beach, sunset", URL.format("j"))]
        legacy = run(SemanticMatcher(FixedSearch(junk), settings(fallback_policy="best_available")).match_scene(scene, "16:9", set()))
        self.assertEqual(legacy.status, "fallback_low_confidence")   # legacy policy keeps its old (disclosed) behaviour

    def test_stricter_threshold_rejects_marginal_clip(self):
        an = self.an.analyze(DOMAINS["farmer"][0])
        c = cands([asset("m", "farmer, watering", URL.format("m"))])
        lenient = score_candidates(c, an, "16:9", 5, settings(match_threshold=0.30))[0]
        strict = score_candidates(c, an, "16:9", 5, settings(match_threshold=0.95))[0]
        self.assertTrue(lenient.passed)
        self.assertFalse(strict.passed)

    def test_scene_without_known_activity_uses_weighted_coverage_without_gates(self):
        an = self.an.analyze("Gentle waves roll onto a quiet beach at dawn.")
        sc = score_candidates(cands([asset("b", "beach, waves, dawn, ocean", URL.format("b"))]), an, "16:9", 5, self.s)[0]
        self.assertIsNone(sc.gated)
        self.assertTrue(sc.passed)

    def test_static_scenes_named_by_nouns_are_not_hard_gated(self):
        """Regression: 'Coral reefs are home to colourful marine life' has no action, so a static reef clip must be able to pass."""
        for text, tags in (("Coral reefs are home to colorful marine life.", "coral, reef, marine, life, fish, colorful"),
                           ("A car drives along the coastal highway.", "car, highway, coast"),
                           ("Students sit in a classroom.", "students, classroom, desks")):
            with self.subTest(text=text):
                an = self.an.analyze(text)
                sc = score_candidates(cands([asset("s", tags, URL.format("s"))]), an, "16:9", 5, self.s)[0]
                self.assertIsNone(sc.gated)
                self.assertTrue(sc.passed, f"{sc.final_score:.2f}")

    def test_label_is_not_a_percentage_and_documents_evidence(self):
        an = self.an.analyze(DOMAINS["farmer"][0])
        sc = score_candidates(cands([asset("r", DOMAINS["farmer"][1], URL.format("r"))]), an, "16:9", 5, self.s)[0]
        self.assertEqual(sc.label, "High")
        self.assertIn("action", sc.evidence["matched_terms"])
        self.assertFalse(sc.visual_verified)
        self.assertEqual(sc.method, "metadata_text_fallback")


if __name__ == "__main__":
    unittest.main()
