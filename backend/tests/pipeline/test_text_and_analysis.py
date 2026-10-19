import json
import unittest

from app.pipeline.scene_analysis import LLMSceneAnalyzer, RuleBasedSceneAnalyzer
from app.pipeline.text import build_scene_texts, normalize_whitespace, split_sentences

SCRIPT = ("Coral reefs are home to colorful marine life.\nSea turtles swim through the ocean.\n\n"
          "Dr. Smith measured 3.5 metres of growth. Wow! Protecting our oceans is important, because they feed us, "
          "cool the planet, and house wonders we have barely begun to understand.")


class SceneSplittingTests(unittest.TestCase):
    def test_no_content_lost_or_rewritten(self):
        scenes = build_scene_texts(SCRIPT)
        self.assertGreaterEqual(len(scenes), 3)
        self.assertEqual(" ".join(scenes), normalize_whitespace(SCRIPT))

    def test_abbreviation_and_decimal_not_split(self):
        sents = split_sentences("Dr. Smith measured 3.5 metres. Then he left.")
        self.assertEqual(sents, ["Dr. Smith measured 3.5 metres.", "Then he left."])

    def test_very_long_sentence_is_split_at_clauses_without_loss(self):
        long = ", ".join(f"item number {i} is here" for i in range(30)) + "."
        scenes = build_scene_texts(long, max_words=20)
        self.assertGreater(len(scenes), 1)
        self.assertTrue(all(len(s.split()) <= 20 for s in scenes))
        self.assertEqual(" ".join(scenes), long)

    def test_empty_script_gives_no_scenes(self):
        self.assertEqual(build_scene_texts("   \n  "), [])


class SceneAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.a = RuleBasedSceneAnalyzer()

    def test_farmer_example_preserves_subject_action_setting(self):
        r = self.a.analyze("A farmer inspects damaged crops after a pest infestation.")
        self.assertEqual(r.subject, "farmer")
        self.assertEqual(r.action, "examining")
        self.assertEqual(r.setting, "agricultural field")
        self.assertEqual(r.queries[0], "farmer examining pest-damaged crop leaves in agricultural field")
        self.assertIn("pest", r.term_weights)
        self.assertTrue(any("inferred" in n for n in r.notes))   # inferred setting is disclosed

    def test_alternative_queries_progressively_broader_and_unique(self):
        r = self.a.analyze("A farmer inspects damaged crops after a pest infestation.")
        self.assertGreaterEqual(len(r.queries), 4)
        self.assertEqual(len(r.queries), len({q.lower() for q in r.queries}))
        self.assertGreater(len(r.queries[0].split()), len(r.queries[-1].split()))
        self.assertTrue(all(len(q.split()) <= 8 for q in r.queries))

    def test_queries_never_end_with_dangling_preposition(self):
        r = self.a.analyze("Workers on a crane lift steel beams at a construction site.")
        for q in r.queries:
            self.assertNotIn(q.split()[-1], {"in", "at", "on", "over"})

    def test_ambiguous_word_disambiguated_by_context(self):
        r = self.a.analyze("A crane lifts steel beams at a construction site.")
        self.assertIn("construction crane", " ".join(r.queries + [r.subject or ""]))

    def test_visual_hint_has_priority_but_narration_untouched(self):
        text = "Coral reefs are home to colorful marine life."
        r = self.a.analyze(text, visual_hint="underwater coral reef with fish")
        self.assertEqual(r.queries[0], "underwater coral reef with fish")
        self.assertEqual(text, "Coral reefs are home to colorful marine life.")

    def test_abstract_narration_gets_disclosed_concept_query(self):
        r = self.a.analyze("The economy is growing faster than expected.")
        self.assertIn("city financial district", r.queries)
        self.assertTrue(any("Abstract concept" in n for n in r.notes))

    def test_avoid_terms_present(self):
        self.assertIn("cartoon", self.a.analyze("Sea turtles swim through the ocean.").avoid_terms)


class LLMAnalyzerTests(unittest.TestCase):
    def test_valid_llm_json_used(self):
        good = json.dumps({"subject": "chef", "action": "chopping", "setting": "kitchen", "objects": ["onions"],
                           "visual_description": "A chef chopping onions", "queries": ["chef chopping onions kitchen"],
                           "avoid_terms": ["cartoon"]})
        r = LLMSceneAnalyzer(lambda p: "Sure! " + good).analyze("Chef cooks.")
        self.assertTrue(r.method.startswith("llm"))
        self.assertEqual(r.queries[0], "chef chopping onions kitchen")

    def test_garbage_llm_output_falls_back_and_is_disclosed(self):
        r = LLMSceneAnalyzer(lambda p: "not json").analyze("Sea turtles swim through the ocean.")
        self.assertEqual(r.method, "rule_based_v1")
        self.assertTrue(any("LLM analysis failed" in n for n in r.notes))

    def test_llm_exception_falls_back(self):
        def boom(_):
            raise TimeoutError("down")
        r = LLMSceneAnalyzer(boom).analyze("Sea turtles swim through the ocean.")
        self.assertTrue(any("failed" in n for n in r.notes))


if __name__ == "__main__":
    unittest.main()
