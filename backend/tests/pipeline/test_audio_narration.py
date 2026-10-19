import asyncio
import tempfile
import unittest
from pathlib import Path

from app.pipeline.errors import AudioValidationError, ProviderUnavailableError, TTSError
from app.pipeline.media import audio_duration, validate_audio_file
from app.pipeline.narration import EdgeTTSProvider, PlaceholderToneTTS, get_pipeline_tts, narrate_scenes
from app.pipeline.transcription import get_pipeline_transcriber

from .helpers import (install_fake_edge_tts, make_settings, make_silent_audio, make_speechlike_audio,
                      uninstall_fake_edge_tts)


def run(c):
    return asyncio.run(c)


class AudioValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.s = make_settings(self.tmp)

    def test_supported_formats_accepted(self):
        for ext in (".mp3", ".wav", ".m4a"):
            p = make_speechlike_audio(self.tmp / f"a{ext}", 3)
            info = validate_audio_file(p, self.s)
            self.assertAlmostEqual(info.duration_seconds, 3.0, delta=0.2, msg=ext)

    def test_rejections(self):
        good = make_speechlike_audio(self.tmp / "ok.mp3", 3)
        cases = {}
        empty = self.tmp / "empty.mp3"; empty.write_bytes(b"")
        cases["empty"] = empty
        bad = self.tmp / "bad.mp3"; bad.write_bytes(b"this is not audio" * 100)
        cases["corrupt"] = bad
        trunc = self.tmp / "trunc.mp3"; trunc.write_bytes(good.read_bytes()[:200])
        cases["truncated"] = trunc
        cases["silent"] = make_silent_audio(self.tmp / "silent.wav")
        txt = self.tmp / "x.txt"; txt.write_text("hi"); cases["wrong_ext"] = txt
        for name, p in cases.items():
            with self.assertRaises(AudioValidationError, msg=name):
                validate_audio_file(p, self.s)

    def test_too_long_and_too_large_and_too_short(self):
        p = make_speechlike_audio(self.tmp / "l.mp3", 5)
        with self.assertRaises(AudioValidationError):
            validate_audio_file(p, make_settings(self.tmp, max_audio_seconds=2))
        with self.assertRaises(AudioValidationError):
            validate_audio_file(p, make_settings(self.tmp, max_audio_mb=0))
        with self.assertRaises(AudioValidationError):
            validate_audio_file(p, make_settings(self.tmp, min_audio_seconds=30))


class ProviderSelectionTests(unittest.TestCase):
    def test_mock_tts_refused_by_default_and_flagged_when_allowed(self):
        with self.assertRaises(ProviderUnavailableError):
            get_pipeline_tts("mock", make_settings(Path("."), allow_mock_providers=False))
        t = get_pipeline_tts("mock", make_settings(Path(".")))
        self.assertFalse(t.is_real)

    def test_mock_transcriber_refused_by_default(self):
        with self.assertRaises(ProviderUnavailableError):
            get_pipeline_transcriber("mock", make_settings(Path("."), allow_mock_providers=False))

    def test_unknown_providers(self):
        with self.assertRaises(ProviderUnavailableError):
            get_pipeline_tts("nope", make_settings(Path(".")))
        with self.assertRaises(ProviderUnavailableError):
            get_pipeline_transcriber("nope", make_settings(Path(".")))

    def test_edge_selected_is_real_flag(self):
        self.assertTrue(get_pipeline_tts("edge", make_settings(Path("."))).is_real)


class NarrationTests(unittest.TestCase):
    TEXTS = ["Sea turtles swim through the ocean.", "Coral reefs host colorful marine life and many fish."]

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.s = make_settings(self.tmp)

    def tearDown(self):
        uninstall_fake_edge_tts()

    def test_edge_provider_generates_playable_audio_with_word_timings(self):
        install_fake_edge_tts(self.tmp)   # stubbed network; real FFmpeg audio
        res = run(narrate_scenes(self.TEXTS, EdgeTTSProvider(), "en-US-AriaNeural", "en", self.tmp / "w", self.s))
        self.assertTrue(res.audio_path.is_file() and res.audio_path.stat().st_size > 0)
        self.assertAlmostEqual(audio_duration(res.audio_path), res.duration_seconds, delta=0.05)
        self.assertEqual(res.timing_method, "tts_word_boundaries")
        self.assertTrue(res.is_real)
        # scenes tile the audio exactly and keep the original text
        self.assertEqual([s.text for s in res.scenes], self.TEXTS)
        self.assertEqual(res.scenes[0].start, 0.0)
        self.assertAlmostEqual(res.scenes[-1].end, res.duration_seconds, delta=0.01)
        self.assertAlmostEqual(res.scenes[0].end, res.scenes[1].start, delta=0.001)
        words = [w for sc in res.scenes for w in sc.words]
        self.assertEqual(" ".join(w.word for w in words), " ".join(self.TEXTS))
        self.assertTrue(all(b.start >= a.start for a, b in zip(words, words[1:])))
        self.assertTrue(all(w.end > w.start for w in words))

    def test_phrase_free_fallback_when_no_boundaries_is_disclosed(self):
        install_fake_edge_tts(self.tmp, words=False)
        res = run(narrate_scenes(self.TEXTS, EdgeTTSProvider(), "v", "en", self.tmp / "w", self.s))
        self.assertEqual(res.timing_method, "scene_exact_words_estimated")

    def test_tts_network_failure_is_a_clear_error(self):
        install_fake_edge_tts(self.tmp, fail=True)
        with self.assertRaises(TTSError) as cm:
            run(narrate_scenes(self.TEXTS, EdgeTTSProvider(), "v", "en", self.tmp / "w", self.s))
        self.assertIn("ConnectionError", cm.exception.message)

    def test_tts_returning_no_audio_is_error(self):
        install_fake_edge_tts(self.tmp, no_audio=True)
        with self.assertRaises(TTSError):
            run(narrate_scenes(self.TEXTS, EdgeTTSProvider(), "v", "en", self.tmp / "w", self.s))

    def test_edge_package_missing_gives_install_hint(self):
        uninstall_fake_edge_tts()
        import sys
        sys.modules["edge_tts"] = None   # forces ImportError even if installed
        try:
            with self.assertRaises(ProviderUnavailableError) as cm:
                run(EdgeTTSProvider().synthesize("hi", "v", "en", self.tmp / "x.mp3"))
            self.assertIn("pip install edge-tts", cm.exception.message)
        finally:
            sys.modules.pop("edge_tts", None)

    def test_timeout(self):
        import sys, types
        mod = types.ModuleType("edge_tts")
        class C:
            def __init__(self, *a, **k): pass
            async def stream(self):
                await asyncio.sleep(5)
                yield {}
        mod.Communicate = C
        sys.modules["edge_tts"] = mod
        with self.assertRaises(TTSError) as cm:
            run(EdgeTTSProvider(timeout=0.2).synthesize("hi", "v", "en", self.tmp / "x.mp3"))
        self.assertIn("timed out", cm.exception.message)

    def test_placeholder_never_reported_as_real(self):
        res = run(narrate_scenes(self.TEXTS, PlaceholderToneTTS(), "v", "en", self.tmp / "w", self.s))
        self.assertFalse(res.is_real)
        self.assertIn("PLACEHOLDER", " ".join(res.warnings))

    def test_empty_scene_list_rejected(self):
        from app.pipeline.errors import InputValidationError
        with self.assertRaises(InputValidationError):
            run(narrate_scenes([], PlaceholderToneTTS(), "v", "en", self.tmp / "w", self.s))


if __name__ == "__main__":
    unittest.main()
