"""End-to-end orchestrator tests. External services are replaced by test doubles (local HTTP file server for
footage, stubbed edge_tts module, deterministic ASR). FFmpeg/FFprobe are REAL."""
import asyncio
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from app import pipeline  # noqa: F401
from app.pipeline import artifacts
from app.pipeline.captions import validate_srt
from app.pipeline.errors import (AudioValidationError, FootageError, InputValidationError, ProviderUnavailableError,
                                 TTSError)
from app.pipeline.media import audio_duration, probe
from app.pipeline.narration import EdgeTTSProvider, PlaceholderToneTTS
from app.pipeline.orchestrator import Deps, JobRequest, run_pipeline

from .helpers import (CatalogSearch, FakeTranscriber, LocalFileServer, asset, install_fake_edge_tts, make_clip,
                      make_settings, make_silent_audio, make_speechlike_audio, uninstall_fake_edge_tts)

SCRIPT = ("A farmer inspects damaged crops after a pest infestation. "
          "Sea turtles swim through the ocean.\n"
          "Coral reefs are home to colorful marine life and countless fish.")


class E2EBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(tempfile.mkdtemp())
        cls.srv_dir = cls.root / "srv"; cls.srv_dir.mkdir()
        for name, rate in (("farm", 25), ("turtle", 24), ("reef", 30), ("generic", 25), ("farm2", 25)):
            make_clip(cls.srv_dir / f"{name}.mp4", 4, rate=rate)
        cls.server = LocalFileServer(cls.srv_dir).__enter__()
        u = cls.server.url
        cls.entries = [
            asset("farm", "farmer, crops, field, pest, leaves", u("farm.mp4")),
            asset("turtle", "sea, turtle, turtles, ocean, underwater, swimming", u("turtle.mp4")),
            asset("reef", "coral, reef, marine, life, fish, colorful", u("reef.mp4")),
            asset("generic", "sky, clouds, nature", u("generic.mp4")),
        ]

    @classmethod
    def tearDownClass(cls):
        cls.server.__exit__(None, None, None)
        shutil.rmtree(cls.root, ignore_errors=True)

    def setUp(self):
        self.jobs = Path(tempfile.mkdtemp(dir=self.root))
        self.s = make_settings(self.jobs)
        self.progress = []

    def tearDown(self):
        uninstall_fake_edge_tts()

    def deps(self, entries=None, tts=None, transcriber=None, **kw):
        return Deps(settings=self.s, search=CatalogSearch(self.entries if entries is None else entries),
                    footage_provider_name="pixabay", get_tts=lambda: tts or PlaceholderToneTTS(),
                    get_transcriber=lambda: transcriber or FakeTranscriber(SCRIPT), **kw)

    def go(self, req, deps, name="job1"):
        d = self.jobs / name
        manifest = asyncio.run(run_pipeline(req, deps, d, name, lambda p, m: self.progress.append((p, m))))
        return d, manifest


class ScriptOnlyTests(E2EBase):
    def test_script_only_with_real_tts_code_path(self):
        install_fake_edge_tts(self.jobs)
        d, m = self.go(JobRequest(script_text=SCRIPT, aspect_ratio="16:9"), self.deps(tts=EdgeTTSProvider()))
        self.assertEqual(m["status"], "completed")
        self.assertEqual(m["input_type"], "script_only")
        self.assertTrue(m["providers"]["tts"]["is_real"])
        self.assertEqual(m["providers"]["tts"]["name"], "edge-tts")
        for f in ("video.mp4", "narration.mp3", "captions.srt", "timeline.json", "manifest.json"):
            self.assertTrue((d / f).is_file() and (d / f).stat().st_size > 0, f)
        self.assertFalse((d / "tmp").exists(), "temporary files must be cleaned up")
        dur = audio_duration(d / "narration.mp3")
        self.assertAlmostEqual(m["outputs"]["audio"]["duration_seconds"], dur, delta=0.05)
        # captions valid and inside the audio duration
        self.assertEqual(validate_srt((d / "captions.srt").read_text("utf-8"), dur, self.s), [])
        self.assertEqual(m["captions"]["timing_method"], "tts_word_boundaries")
        self.assertTrue(m["captions"]["burned_in"])
        # video streams
        info = probe(d / "video.mp4")
        kinds = {s["codec_type"]: s for s in info["streams"]}
        self.assertEqual((kinds["video"]["width"], kinds["video"]["height"]), (1920, 1080))
        self.assertEqual(kinds["audio"]["codec_name"], "aac")
        self.assertAlmostEqual(float(info["format"]["duration"]), dur, delta=0.25)
        self.assertTrue(m["validation"]["video"]["passed"], m["validation"]["video"])
        self.assertEqual(m["validation"]["duplicate_clips_in_video"], [])
        self.assertEqual(self.progress[-1][0], 100)

    def test_timeline_and_manifest_contents(self):
        d, m = self.go(JobRequest(script_text=SCRIPT), self.deps())
        tl = json.loads((d / "timeline.json").read_text("utf-8"))
        self.assertEqual(artifacts.validate_timeline(tl, m["outputs"]["audio"]["duration_seconds"]), [])
        self.assertEqual(artifacts.validate_manifest(json.loads((d / "manifest.json").read_text("utf-8"))), [])
        scenes = tl["scenes"]
        self.assertEqual(" ".join(s["narration_text"] for s in scenes), " ".join(SCRIPT.split()))
        by_clip = {s["clip"]["clip_id"] for s in scenes}
        self.assertTrue({"farm", "turtle", "reef"} <= by_clip, by_clip)
        first = scenes[0]
        self.assertIn("farmer", first["visual_description"].lower())
        self.assertEqual(first["clip"]["clip_id"], "farm")
        self.assertTrue(first["relevance"]["passed_threshold"])
        self.assertEqual(first["relevance"]["scoring_method"], "metadata_text_fallback")
        self.assertFalse(first["relevance"]["visual_verified"])
        self.assertEqual(first["license"]["commercial_use_basis"], "provider_policy_documented")
        self.assertFalse(first["license"]["verified_per_clip"])
        self.assertEqual(first["clip"]["creator"], "tester")
        self.assertIsNone(first["clip"]["ai_generated"])
        self.assertTrue(first["search_queries"])
        # manifest honesty
        self.assertFalse(m["matching"]["visual_ai_verification_performed"])
        self.assertTrue(any("No vision-language model" in w for w in m["warnings"]))
        self.assertTrue(any(e["type"] == "mock_provider" for e in m["fallback_events"]))   # placeholder TTS disclosed
        self.assertFalse(m["providers"]["tts"]["is_real"])
        self.assertTrue(any("PLACEHOLDER" in w for w in m["warnings"]))
        self.assertIn("total", m["processing_durations_seconds"])
        self.assertEqual(m["cost_estimate"]["amount_usd"], 0.0)
        self.assertNotIn("pixabay_api_key", json.dumps(m).lower())
        self.assertNotIn(str(self.jobs), json.dumps(m), "manifest must not leak absolute paths")

    def test_aspect_ratios_end_to_end(self):
        for asp, (w, h) in {"9:16": (1080, 1920), "1:1": (1080, 1080)}.items():
            d, m = self.go(JobRequest(script_text=SCRIPT, aspect_ratio=asp, burn_captions=False), self.deps(), name=f"asp{w}x{h}")
            v = next(s for s in probe(d / "video.mp4")["streams"] if s["codec_type"] == "video")
            self.assertEqual((v["width"], v["height"]), (w, h))
            self.assertEqual(m["requested_aspect_ratio"], asp)
            self.assertFalse(m["captions"]["burned_in"])

    def test_captions_can_be_disabled(self):
        d, m = self.go(JobRequest(script_text=SCRIPT, captions_enabled=False), self.deps())
        self.assertFalse((d / "captions.srt").exists())
        self.assertIsNone(m["outputs"]["captions"])

    def test_mock_tts_refused_without_opt_in(self):
        self.s.allow_mock_providers = False
        from app.pipeline.narration import get_pipeline_tts
        deps = self.deps()
        deps.get_tts = lambda: get_pipeline_tts("mock", self.s)
        with self.assertRaises(ProviderUnavailableError):
            self.go(JobRequest(script_text=SCRIPT), deps)

    def test_tts_failure_returns_clear_error(self):
        install_fake_edge_tts(self.jobs, fail=True)
        with self.assertRaises(TTSError):
            self.go(JobRequest(script_text=SCRIPT), self.deps(tts=EdgeTTSProvider()))
        self.assertFalse((self.jobs / "job1" / "video.mp4").exists())
        self.assertFalse((self.jobs / "job1" / "tmp").exists())

    def test_empty_and_invalid_inputs_rejected(self):
        for bad in ("", "   \n\t ", "!!! ???"):
            with self.assertRaises(InputValidationError, msg=repr(bad)):
                self.go(JobRequest(script_text=bad), self.deps())
        with self.assertRaises(InputValidationError):
            self.go(JobRequest(), self.deps())
        with self.assertRaises(InputValidationError):
            self.go(JobRequest(script_text="x" * (self.s.max_script_chars + 1)), self.deps())
        with self.assertRaises(InputValidationError):
            self.go(JobRequest(script_text="Hello there world.", aspect_ratio="4:3"), self.deps())

    def test_no_footage_fails_with_per_scene_detail(self):
        with self.assertRaises(FootageError) as cm:
            self.go(JobRequest(script_text=SCRIPT), self.deps(entries=[]))
        scenes = cm.exception.details["scenes"]
        self.assertEqual(len(scenes), 3)
        self.assertTrue(all(s["queries"] for s in scenes))

    def test_provider_outage_reported(self):
        deps = self.deps()
        deps.search = CatalogSearch([], always_fail=True)
        with self.assertRaises(FootageError) as cm:
            self.go(JobRequest(script_text="Sea turtles swim through the ocean."), deps)
        self.assertEqual(cm.exception.details["scenes"][0]["status"], "provider_error")
        self.assertTrue(cm.exception.details["scenes"][0]["errors"])

    def test_low_confidence_fallback_is_disclosed_in_outputs(self):
        entries = [asset("generic", "sky, clouds, nature, ocean", self.server.url("generic.mp4"))]
        # this test covers the LEGACY "best_available" policy (always use the best clip, disclosed). The default policy
        # ("review") would call this clip unmatched; see test_action_matching.py / test_unmatched_scenes.py.
        self.s.fallback_policy = "best_available"
        d, m = self.go(JobRequest(script_text="Sea turtles swim through the ocean."), self.deps(entries=entries))
        tl = json.loads((d / "timeline.json").read_text("utf-8"))
        sc = tl["scenes"][0]
        self.assertEqual(sc["match_status"], "fallback_low_confidence")
        self.assertFalse(sc["relevance"]["passed_threshold"])
        self.assertTrue(any(e["type"] == "low_confidence_fallback" for e in m["fallback_events"]))
        self.assertEqual(m["matching"]["scenes_low_confidence_fallback"], 1)
        self.assertTrue(any("may NOT accurately depict" in w for w in m["warnings"]))

    def test_unavailable_top_clip_falls_back_to_next_best_with_warning(self):
        entries = [asset("dead", "farmer, crops, field, pest, leaves, damaged", self.server.url("missing.mp4")),
                   asset("farm2", "farmer, crops, field, pest", self.server.url("farm2.mp4"))]
        d, m = self.go(JobRequest(script_text="A farmer inspects damaged crops after a pest infestation."), self.deps(entries=entries))
        tl = json.loads((d / "timeline.json").read_text("utf-8"))
        self.assertEqual(tl["scenes"][0]["clip"]["clip_id"], "farm2")
        self.assertTrue(any(e["type"] == "download_failure" for e in m["fallback_events"]))
        self.assertTrue(any("next-best" in w for w in m["warnings"]))

    def test_all_clips_unavailable_is_error(self):
        from app.pipeline.errors import DownloadError
        entries = [asset("dead", "farmer, crops, field, pest", self.server.url("missing.mp4"))]
        with self.assertRaises(DownloadError):
            self.go(JobRequest(script_text="A farmer inspects damaged crops after a pest infestation."), self.deps(entries=entries))

    def test_clip_reuse_only_when_unavoidable_and_disclosed(self):
        entries = [asset("farm", "farmer, crops, field, pest", self.server.url("farm.mp4"))]
        d, m = self.go(JobRequest(script_text="A farmer inspects damaged crops after a pest infestation. "
                                              "A farmer inspects damaged crops after a pest infestation again today."),
                       self.deps(entries=entries))
        self.assertEqual(m["validation"]["duplicate_clips_in_video"], ["pixabay:farm"])
        self.assertTrue(any(e["type"] == "clip_reuse" for e in m["fallback_events"]))


class UploadedAudioTests(E2EBase):
    SPEECH = ("A farmer inspects damaged crops after a pest infestation. Sea turtles swim through the ocean. "
              "Coral reefs are home to colorful marine life and countless fish.")

    def audio(self, seconds=14.0, ext=".mp3", name="up"):
        return make_speechlike_audio(self.jobs / f"{name}{ext}", seconds)

    def test_uploaded_audio_is_source_of_truth(self):
        up = self.audio(14.0)
        tr = FakeTranscriber(self.SPEECH, lead_in=0.5, tail_out=0.5)
        d, m = self.go(JobRequest(audio_path=up, audio_original_name="my talk.mp3"), self.deps(transcriber=tr))
        self.assertEqual(m["input_type"], "uploaded_audio")
        self.assertEqual(m["providers"]["transcription"]["name"], "fake-test-asr")
        self.assertFalse(m["providers"]["transcription"]["is_real"])
        self.assertTrue(any("MOCK TRANSCRIPTION" in w for w in m["warnings"]))
        self.assertIsNone(m["providers"]["tts"]["name"])        # no narration regenerated
        self.assertEqual(m["outputs"]["audio"]["source"], "uploaded")
        self.assertAlmostEqual(m["outputs"]["audio"]["duration_seconds"], 14.0, delta=0.15)
        self.assertEqual(m["captions"]["timing_method"], "asr_word_timestamps")
        tl = json.loads((d / "timeline.json").read_text("utf-8"))
        self.assertEqual(artifacts.validate_timeline(tl, m["outputs"]["audio"]["duration_seconds"]), [])
        self.assertEqual([s["clip"]["clip_id"] for s in tl["scenes"]][:3], ["farm", "turtle", "reef"])
        # narration in the video is the upload (same duration), captions follow speech start
        self.assertAlmostEqual(audio_duration(d / "narration.mp3"), 14.0, delta=0.15)
        cues = (d / "captions.srt").read_text("utf-8")
        self.assertIn("00:00:00,500", cues)   # first speech at 0.5s, not at 0
        self.assertEqual(validate_srt(cues, audio_duration(d / "narration.mp3"), self.s), [])
        self.assertTrue(m["validation"]["video"]["passed"], m["validation"]["video"])

    def test_wav_and_m4a_accepted(self):
        for ext in (".wav", ".m4a"):
            up = self.audio(8.0, ext, name=f"u{ext[1:]}")
            d, m = self.go(JobRequest(audio_path=up, audio_original_name=f"x{ext}"),
                           self.deps(transcriber=FakeTranscriber("Sea turtles swim through the ocean today.")), name=f"j{ext[1:]}")
            self.assertEqual(m["status"], "completed")
            self.assertTrue((d / f"narration{ext}").is_file())

    def test_script_plus_audio_small_mismatch_uses_script_text_audio_timing(self):
        up = self.audio(10.0)
        script = "Sea turtles swim through the open ocean."
        tr = FakeTranscriber("Sea turtles swim through the ocean.")
        d, m = self.go(JobRequest(audio_path=up, script_text=script, audio_original_name="a.mp3"), self.deps(transcriber=tr))
        rec = m["input"]["script_audio_reconciliation"]
        self.assertEqual(rec["rule"], "script_wording_with_audio_timing")
        self.assertTrue(any("differ slightly" in w for w in m["warnings"]))
        tl = json.loads((d / "timeline.json").read_text("utf-8"))
        self.assertEqual(" ".join(s["narration_text"] for s in tl["scenes"]), script)

    def test_script_plus_audio_large_mismatch_is_reported_and_audio_wins(self):
        up = self.audio(10.0)
        tr = FakeTranscriber("Sea turtles swim through the ocean today.")
        d, m = self.go(JobRequest(audio_path=up, script_text="Stock markets crashed during a volatile trading session.",
                                  audio_original_name="a.mp3"), self.deps(transcriber=tr))
        self.assertEqual(m["input"]["script_audio_reconciliation"]["rule"], "transcript_of_audio_used_script_ignored")
        self.assertTrue(any(e["type"] == "script_audio_mismatch" for e in m["fallback_events"]))
        tl = json.loads((d / "timeline.json").read_text("utf-8"))
        self.assertIn("turtles", tl["scenes"][0]["narration_text"].lower())

    def test_segment_level_asr_timing_is_disclosed(self):
        d, m = self.go(JobRequest(audio_path=self.audio(8.0), audio_original_name="a.mp3"),
                       self.deps(transcriber=FakeTranscriber("Sea turtles swim through the ocean today.", granularity="segment_interpolated")))
        self.assertEqual(m["captions"]["timing_method"], "asr_segment_interpolated")
        self.assertTrue(m["captions"]["alignment_precision"].startswith("approximate"))
        self.assertTrue(any(e["type"] == "timing_fallback" for e in m["fallback_events"]))

    def test_invalid_audio_rejected(self):
        bad = self.jobs / "bad.mp3"; bad.write_bytes(b"junk" * 500)
        for p, label in ((bad, "corrupt"), (make_silent_audio(self.jobs / "s.wav"), "silent")):
            with self.assertRaises(AudioValidationError, msg=label):
                self.go(JobRequest(audio_path=p, audio_original_name=p.name), self.deps(), name=f"bad_{label}")

    def test_real_transcriber_required_by_default(self):
        from app.pipeline.transcription import get_pipeline_transcriber
        self.s.allow_mock_providers = False
        deps = self.deps()
        deps.get_transcriber = lambda: get_pipeline_transcriber("mock", self.s)
        with self.assertRaises(ProviderUnavailableError):
            self.go(JobRequest(audio_path=self.audio(5.0), audio_original_name="a.mp3"), deps)


if __name__ == "__main__":
    unittest.main()
