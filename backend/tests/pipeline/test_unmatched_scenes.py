"""Pipeline behaviour when footage is missing, broken, duplicated or unverifiable. Real FFmpeg, mocked provider."""
import asyncio
import json
import shutil
import tempfile
import time
import unittest
from pathlib import Path

from app.pipeline.errors import DownloadError, FootageError
from app.pipeline.jobs import JobRunner, JobStore
from app.pipeline.media import video_info
from app.pipeline.narration import PlaceholderToneTTS
from app.pipeline.orchestrator import Deps, JobRequest, run_pipeline
from app.pipeline.visual import UnavailableVisualVerifier

from .helpers import CatalogSearch, FakeTranscriber, LocalFileServer, asset, ff, make_clip, make_settings

TWO = "A farmer waters tomato plants in a field. A teacher explains a science experiment to students."


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(tempfile.mkdtemp())
        cls.srv_dir = cls.root / "srv"
        cls.srv_dir.mkdir()
        make_clip(cls.srv_dir / "water.mp4", 5, pattern="testsrc2")
        make_clip(cls.srv_dir / "other.mp4", 5, pattern="smptebars")
        shutil.copy(cls.srv_dir / "water.mp4", cls.srv_dir / "water_copy.mp4")           # same footage, different upload
        (cls.srv_dir / "broken.mp4").write_bytes(b"this is not a video" * 2000)
        cls.server = LocalFileServer(cls.srv_dir).__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.server.__exit__(None, None, None)
        shutil.rmtree(cls.root, ignore_errors=True)

    def setUp(self):
        self.jobs = Path(tempfile.mkdtemp(dir=self.root))
        self.s = make_settings(self.jobs)

    def go(self, script, entries, name="j", **kw):
        deps = Deps(settings=self.s, search=CatalogSearch(entries), footage_provider_name="pixabay", get_tts=lambda: PlaceholderToneTTS(),
                    get_transcriber=lambda: FakeTranscriber(script), **kw)
        d = self.jobs / name
        m = asyncio.run(run_pipeline(JobRequest(script_text=script), deps, d, name, lambda p, s: None))
        return d, m

    def url(self, n):
        return self.server.url(n)


class UnmatchedSceneTests(Base):
    def test_unmatched_scene_gets_placeholder_and_job_still_completes(self):
        d, m = self.go(TWO, [asset("w", "farmer, watering, tomato, plants, field", self.url("water.mp4"), duration=5)])
        self.assertEqual(m["status"], "completed")
        tl = json.loads((d / "timeline.json").read_text())
        a, b = tl["scenes"]
        self.assertEqual(a["match_status"], "matched")
        self.assertIn(b["match_status"], ("unmatched", "no_candidates"))
        self.assertEqual((b["placeholder"], b["clip"]), (True, None))
        self.assertIn("no usable candidates", b["selection_reason"])
        self.assertEqual(m["matching"]["scenes_unmatched"], 1)
        self.assertEqual(m["diagnostics"]["counts"]["unmatched"], 1)
        self.assertTrue(any("no matching footage" in x["text"] for x in m["diagnostics"]["summary"]))
        self.assertFalse(any(x["text"] == "Video generated successfully." for x in m["diagnostics"]["summary"]))
        info = video_info(d / "video.mp4")
        self.assertAlmostEqual(info["duration"], tl["audio_duration_seconds"] if "audio_duration_seconds" in tl else info["duration"], delta=0.6)
        self.assertTrue(m["outputs"]["video"]["duration_seconds"] > 3)

    def test_placeholder_is_not_reported_as_a_frozen_video_warning(self):
        _, m = self.go(TWO, [asset("w", "farmer, watering, tomato, plants, field", self.url("water.mp4"), duration=5)])
        self.assertFalse(any("frozen" in w and "scene-002" in w for w in m["warnings"]))

    def test_nothing_matched_at_all_is_an_error(self):
        with self.assertRaises(FootageError):
            self.go(TWO, [])

    def test_strict_policy_fails_on_any_unmatched_scene(self):
        self.s.fallback_policy = "fail"
        with self.assertRaises(FootageError):
            self.go(TWO, [asset("w", "farmer, watering, tomato, plants, field", self.url("water.mp4"), duration=5)])

    def test_wrong_action_clip_is_used_only_as_flagged_review(self):
        _, m = self.go("A doctor checks the patient's blood pressure with a cuff.",
                       [asset("c", "doctor, patient, hospital, consultation, talking", self.url("other.mp4"), duration=5)])
        self.assertEqual(m["matching"]["scenes_needs_review"], 1)
        self.assertTrue(any("need" in x["text"] and "review" in x["text"] for x in m["diagnostics"]["summary"]))


class DownloadFallbackTests(Base):
    SENT = "A farmer waters tomato plants in a field."

    def test_dead_top_clip_falls_back_to_next_acceptable_candidate(self):
        entries = [asset("dead", "farmer, watering, tomato, plants, field, greenhouse", self.url("missing-404.mp4"), duration=5),
                   asset("ok", "farmer, watering, tomato, plants", self.url("water.mp4"), duration=5)]
        d, m = self.go(self.SENT, entries)
        s = json.loads((d / "timeline.json").read_text())["scenes"][0]
        self.assertEqual(s["clip"]["clip_id"], "ok")
        self.assertTrue(s["used_alternate"])
        self.assertEqual(s["download_attempts"][0]["outcome"], "download_failed")
        self.assertEqual(s["match_status"], "matched")
        self.assertEqual(m["diagnostics"]["counts"]["alternate_clips"], 1)
        self.assertTrue(any(i["code"] == "alternate_clip" for i in m["diagnostics"]["items"]))

    def test_invalid_media_is_rejected_and_next_candidate_used(self):
        entries = [asset("bad", "farmer, watering, tomato, plants, field, greenhouse", self.url("broken.mp4"), duration=5),
                   asset("ok", "farmer, watering, tomato, plants", self.url("water.mp4"), duration=5)]
        d, _ = self.go(self.SENT, entries)
        s = json.loads((d / "timeline.json").read_text())["scenes"][0]
        self.assertEqual(s["clip"]["clip_id"], "ok")
        self.assertEqual(s["download_attempts"][0]["outcome"], "download_failed")

    def test_alternate_must_still_pass_the_relevance_threshold(self):
        entries = [asset("dead", "farmer, watering, tomato, plants, field", self.url("missing-404.mp4"), duration=5),
                   asset("junk", "beach, sunset, sea", self.url("other.mp4"), duration=5)]
        with self.assertRaises(DownloadError):          # single scene, the only relevant clip is dead -> nothing to render
            self.go(self.SENT, entries)

    def test_one_scene_all_dead_does_not_fail_the_other_scene(self):
        sent = "A farmer waters tomato plants in a field. A tourist photographs the Eiffel Tower in Paris."
        entries = [asset("w", "farmer, watering, tomato, plants", self.url("water.mp4"), duration=5),
                   asset("p", "tourist, taking photo, eiffel tower, paris", self.url("missing-404.mp4"), duration=5)]
        d, m = self.go(sent, entries)
        a, b = json.loads((d / "timeline.json").read_text())["scenes"]
        self.assertEqual(a["match_status"], "matched")
        self.assertEqual((b["match_status"], b["placeholder"]), ("unmatched", True))   # a failed download is never reported as a match
        self.assertIn("failed to download", b["selection_reason"])


class DuplicateTests(Base):
    def test_near_duplicate_footage_is_skipped_for_a_distinct_alternative(self):
        sent = "A farmer waters tomato plants in a sunny field. A farmer waters crops with a sprinkler."
        entries = [asset("A", "farmer, watering, tomato, plants, field, sunny", self.url("water.mp4"), duration=5),
                   asset("A2", "farmer, watering, sprinkler, crops", self.url("water_copy.mp4"), duration=5),   # same footage re-uploaded
                   asset("B", "farmer, watering, crops", self.url("other.mp4"), duration=5)]
        d, _ = self.go(sent, entries)
        s1, s2 = json.loads((d / "timeline.json").read_text())["scenes"]
        self.assertEqual(s1["clip"]["clip_id"], "A")
        self.assertEqual(s2["clip"]["clip_id"], "B")
        self.assertTrue(any(x["outcome"] == "near_duplicate" for x in s2["download_attempts"]))

    def test_near_duplicate_is_reused_but_disclosed_when_nothing_else_exists(self):
        sent = "A farmer waters tomato plants in a sunny field. A farmer waters crops with a sprinkler."
        entries = [asset("A", "farmer, watering, tomato, plants, field, sunny", self.url("water.mp4"), duration=5),
                   asset("A2", "farmer, watering, sprinkler, crops", self.url("water_copy.mp4"), duration=5)]
        d, m = self.go(sent, entries)
        tl = json.loads((d / "timeline.json").read_text())
        self.assertEqual(tl["scenes"][1]["clip"]["clip_id"], "A2")
        self.assertTrue(any("resembles" in w for s in tl["scenes"] for w in s.get("warnings", [])) or
                        any(e.get("type") == "near_duplicate_footage" for e in m["fallback_events"]))


class VisualUnavailableTests(Base):
    def test_requested_but_unavailable_visual_check_is_disclosed_and_job_completes(self):
        v = UnavailableVisualVerifier("clip", "open_clip_torch is not installed")
        _, m = self.go("A farmer waters tomato plants in a field.", [asset("w", "farmer, watering, tomato, plants", self.url("water.mp4"), duration=5)], verifier=v)
        self.assertEqual(m["status"], "completed")
        self.assertFalse(m["matching"]["visual_ai_verification_performed"])
        self.assertEqual(m["providers"]["visual_verifier"]["unavailable_reason"], "open_clip_torch is not installed")
        codes = {i["code"] for i in m["diagnostics"]["items"]}
        self.assertTrue({"visual_unavailable", "metadata_only"} <= codes)


class ConcurrencyAndPersistence(Base):
    def test_two_concurrent_jobs_do_not_share_state_and_state_persists(self):
        store = JobStore(self.s)
        entries = [asset("w", "farmer, watering, tomato, plants", self.url("water.mp4"), duration=5),
                   asset("p", "tourist, taking photo, eiffel tower, paris", self.url("other.mp4"), duration=5)]
        runner = JobRunner(store, lambda req: Deps(settings=self.s, search=CatalogSearch(entries), footage_provider_name="pixabay",
                                                   get_tts=lambda: PlaceholderToneTTS(), get_transcriber=lambda: FakeTranscriber("x")), max_workers=2)
        ids = []
        for text in ("A farmer waters tomato plants in a field.", "A tourist photographs the Eiffel Tower in Paris."):
            jid = store.new_id()
            store.job_dir(jid)
            runner.submit(jid, JobRequest(script_text=text), "script_only", fingerprint=text)
            ids.append(jid)
        self.assertEqual(runner.find_active("A farmer waters tomato plants in a field."), ids[0])
        end = time.time() + 240
        while time.time() < end and any(store.read(i)["status"] not in ("completed", "failed") for i in ids):
            time.sleep(0.5)
        for jid, word in zip(ids, ("farmer", "tourist")):
            st = store.read(jid)
            self.assertEqual(st["status"], "completed", st.get("error"))
            tl = json.loads((store.job_dir(jid) / "timeline.json").read_text())
            self.assertEqual(len(tl["scenes"]), 1)
            self.assertIn(word, tl["scenes"][0]["narration_text"].lower())
            self.assertEqual(st["summary"][0]["level"], "success")
        self.assertIsNone(runner.find_active("A farmer waters tomato plants in a field."))
        self.assertTrue(JobStore(self.s).read(ids[0])["status"] == "completed")     # persisted on disk, not in memory


if __name__ == "__main__":
    unittest.main()
