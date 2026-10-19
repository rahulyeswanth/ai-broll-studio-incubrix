"""Frame-level verification logic with a deterministic fake embedding backend (NOT real CLIP).

Colour stands in for content: a red clip "is" blood-pressure measurement, a blue clip "is" a doctor consultation.
This proves frame extraction, contrastive foil scoring, gating and graceful degradation. It does not prove that
OpenCLIP discriminates real footage - that needs the optional model and was not run here."""
import asyncio
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from app.pipeline.matching import SemanticMatcher
from app.pipeline.models import Scene
from app.pipeline.provider_adapter import candidate_from_asset
from app.pipeline.ranking import score_candidates
from app.pipeline.scene_analysis import RuleBasedSceneAnalyzer
from app.pipeline.visual import (ContrastiveVisualVerifier, EmbeddingBackend, NullVisualVerifier,
                                 UnavailableVisualVerifier, VisualVerifier, build_prompts, get_visual_verifier)

from .helpers import CatalogSearch, asset, ff, make_settings

TEXT = "A doctor examines a patient and checks the patient's blood pressure."


class ColourBackend(EmbeddingBackend):
    name, version = "fake-colour", "test"

    def embed_texts(self, texts):
        out = []
        for t in texts:
            t = t.lower()
            out.append([1, 0, 0] if "blood pressure" in t else [0, 0, 1] if ("consultation" in t or "talking" in t) else [0, 1, 0] if "empty place" in t else [0, 0, 0])
        return out

    def embed_images(self, paths):
        from PIL import Image
        out = []
        for p in paths:
            r, g, b = Image.open(p).convert("RGB").resize((1, 1)).getpixel((0, 0))
            n = max((r * r + g * g + b * b) ** 0.5, 1e-6)
            out.append([r / n, g / n, b / n])
        return out


class BrokenBackend(ColourBackend):
    def embed_images(self, paths):
        raise RuntimeError("simulated GPU out of memory")


class DeadTextBackend(ColourBackend):
    def embed_texts(self, texts):
        raise RuntimeError("model weights missing")


class VisualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(tempfile.mkdtemp())
        for name, colour in (("red", "red"), ("blue", "blue")):
            ff("-f", "lavfi", "-i", f"color=c={colour}:s=320x240:d=3:r=10", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(cls.root / f"{name}.mp4"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def setUp(self):
        self.s = make_settings(self.root)
        self.an = RuleBasedSceneAnalyzer().analyze(TEXT)
        # metadata that looks equally GOOD for both clips: only the frames can tell them apart
        self.red = candidate_from_asset(asset("red", "doctor, patient, blood pressure, cuff, clinic", "http://x/red.mp4"))
        self.blue = candidate_from_asset(asset("blue", "doctor, patient, blood pressure, cuff, clinic", "http://x/blue.mp4"))

    def fetch(self, cand, tmp):
        return self.root / f"{cand.clip_id}.mp4"

    def test_prompts_contain_foils_for_confusable_actions(self):
        pos, foils = build_prompts(self.an)
        self.assertTrue(any("blood pressure" in p.lower() for p in pos))
        self.assertTrue(any("consultation" in f or "talking" in f for f in foils))

    def test_frames_are_inspected_and_foil_leading_clip_is_detected(self):
        v = ContrastiveVisualVerifier(ColourBackend(), frames=3)
        scores = v.score(self.an, [self.red, self.blue], self.fetch)
        self.assertGreater(scores[self.red.key], 0.8)
        self.assertLess(scores[self.blue.key], 0.3)
        self.assertEqual(v.details[self.red.key]["frames_inspected"], 3)
        self.assertFalse(v.details[self.red.key]["foil_leading"])
        self.assertTrue(v.details[self.blue.key]["foil_leading"])

    def test_visual_evidence_overrides_matching_metadata(self):
        v = ContrastiveVisualVerifier(ColourBackend())
        scores = v.score(self.an, [self.red, self.blue], self.fetch)
        sc = {x.candidate.clip_id: x for x in score_candidates([self.red, self.blue], self.an, "16:9", 5, self.s, scores, v.details)}
        self.assertTrue(sc["red"].passed and sc["red"].visual_verified)
        self.assertFalse(sc["blue"].passed, "metadata matched but the frames show a consultation")
        self.assertIn("frame analysis", sc["blue"].gated)
        self.assertEqual(sc["red"].method, "visual_clip+metadata_text")

    def test_matcher_end_to_end_with_visual_verifier(self):
        entries = [asset("red", "doctor, patient, blood pressure, cuff", "http://x/red.mp4"), asset("blue", "doctor, patient, blood pressure, cuff", "http://x/blue.mp4")]
        m = SemanticMatcher(CatalogSearch(entries), self.s, verifier=ContrastiveVisualVerifier(ColourBackend()), fetch_preview=self.fetch)
        dec = asyncio.run(m.match_scene(Scene("scene-001", 1, TEXT, 0, 5), "16:9", set()))
        self.assertEqual(dec.selected.candidate.clip_id, "red")
        self.assertTrue(dec.selected.visual_verified)

    def test_model_failure_on_frames_falls_back_to_metadata(self):
        v = ContrastiveVisualVerifier(BrokenBackend())
        self.assertEqual(v.score(self.an, [self.red], self.fetch), {})

    def test_model_failure_on_load_does_not_abort_matching(self):
        entries = [asset("red", "doctor, patient, blood pressure, cuff", "http://x/red.mp4")]
        m = SemanticMatcher(CatalogSearch(entries), self.s, verifier=ContrastiveVisualVerifier(DeadTextBackend()), fetch_preview=self.fetch)
        dec = asyncio.run(m.match_scene(Scene("scene-001", 1, TEXT, 0, 5), "16:9", set()))
        self.assertEqual(dec.status, "matched")
        self.assertFalse(dec.selected.visual_verified)
        self.assertTrue(any("Visual verifier failed" in w for w in dec.warnings))

    def test_missing_configuration_degrades_with_a_reason_instead_of_raising(self):
        cfg = SimpleNamespace(clip_model_name="ViT-B-32", clip_pretrained="x")
        try:
            import open_clip  # noqa: F401
            self.skipTest("open_clip installed in this environment")
        except ImportError:
            pass
        v = get_visual_verifier("clip", cfg)
        self.assertIsInstance(v, UnavailableVisualVerifier)
        self.assertFalse(v.performs_visual_analysis)
        self.assertIn("open_clip_torch", v.unavailable_reason)
        self.assertEqual(v.score(self.an, [self.red], self.fetch), {})

    def test_unknown_and_metadata_modes(self):
        cfg = SimpleNamespace(clip_model_name="a", clip_pretrained="b")
        self.assertIsInstance(get_visual_verifier("metadata", cfg), NullVisualVerifier)
        self.assertIsInstance(get_visual_verifier("nonsense", cfg), UnavailableVisualVerifier)

    def test_no_visual_scores_are_invented_when_unavailable(self):
        sc = score_candidates([self.red], self.an, "16:9", 5, self.s, {})[0]
        self.assertIsNone(sc.visual_score)
        self.assertFalse(sc.visual_verified)


if __name__ == "__main__":
    unittest.main()
