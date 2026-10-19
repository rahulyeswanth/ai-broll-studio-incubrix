import unittest
from pathlib import Path

from app.pipeline.alignment import (distribute_words, map_script_to_timed_words, scenes_from_words, similarity, tokenize)
from app.pipeline.captions import build_cues, parse_srt, to_srt, validate_srt
from app.pipeline.models import WordTiming

from .helpers import make_settings

TEXT = ("Sea turtles swim through the ocean. Coral reefs host colorful marine life and many fish, "
        "while sharks patrol the deeper water. Protecting our oceans is important for everyone.")


def words(dur=14.0):
    return distribute_words(tokenize(TEXT), 0.3, dur - 0.3)


class SceneFromWordsTests(unittest.TestCase):
    def test_scenes_tile_audio_exactly_and_keep_text(self):
        s = make_settings(Path("."), min_scene_seconds=1.0)
        scenes = scenes_from_words(words(), 14.0, s, "asr_word_timestamps")
        self.assertGreaterEqual(len(scenes), 2)
        self.assertEqual(scenes[0].start, 0.0)
        self.assertEqual(scenes[-1].end, 14.0)
        for a, b in zip(scenes, scenes[1:]):
            self.assertEqual(a.end, b.start)
        self.assertEqual(" ".join(sc.text for sc in scenes), TEXT)
        for sc in scenes:   # scene boundary agrees with its words
            self.assertLessEqual(sc.start, sc.words[0].start + 1e-6)
            self.assertGreaterEqual(sc.end, sc.words[-1].end - 1e-6)

    def test_short_scenes_merged(self):
        s = make_settings(Path("."), min_scene_seconds=6.0)
        scenes = scenes_from_words(words(), 14.0, s, "x")
        self.assertTrue(all(sc.end - sc.start >= 6.0 - 1e-6 for sc in scenes[:-1]) or len(scenes) == 1)


class ScriptAudioReconcileTests(unittest.TestCase):
    def test_mapping_keeps_script_text_with_audio_timing(self):
        script = "Sea turtles swim through the ocean."
        asr = [WordTiming(w, 0.5 + i * 0.4, 0.8 + i * 0.4) for i, w in enumerate("sea turtle swim through the ocean".split())]
        mapped, matched = map_script_to_timed_words(tokenize(script), asr, 0.0, 4.0)
        self.assertEqual([w.word for w in mapped], tokenize(script))
        self.assertGreater(matched, 0.6)
        self.assertTrue(all(b.start >= a.start and b.end > b.start for a, b in zip(mapped, mapped[1:])))
        self.assertAlmostEqual(mapped[0].start, 0.5, places=2)

    def test_similarity_scores(self):
        self.assertGreater(similarity("Sea turtles swim.", "sea turtles swim"), 0.99)
        self.assertLess(similarity("Sea turtles swim.", "stock markets crashed today"), 0.3)
        self.assertEqual(similarity("", "x"), 0.0)


class CaptionTests(unittest.TestCase):
    def setUp(self):
        self.s = make_settings(Path("."))
        self.dur = 14.0
        self.cues = build_cues(words(self.dur), self.dur, self.s)
        self.srt = to_srt(self.cues)

    def test_valid_srt(self):
        self.assertEqual(validate_srt(self.srt, self.dur, self.s), [])
        self.assertGreater(len(self.cues), 2)

    def test_properties(self):
        prev = 0.0
        for c in self.cues:
            self.assertGreaterEqual(c.start, 0)
            self.assertGreater(c.end, c.start)
            self.assertGreaterEqual(c.start, prev - 1e-9)
            self.assertLessEqual(c.end, self.dur)
            self.assertLessEqual(len(c.text.split("\n")), self.s.caption_max_lines)
            self.assertTrue(all(len(l) <= self.s.caption_max_chars_per_line + 12 for l in c.text.split("\n")))
            prev = c.end

    def test_all_words_present_in_order(self):
        joined = " ".join(c.text.replace("\n", " ") for c in self.cues)
        self.assertEqual(joined, TEXT)

    def test_timing_follows_speech_not_equal_chunks(self):
        # a long pause in the middle of speech must show up as a cue gap/boundary
        ws = distribute_words(tokenize("one two three four five six"), 0.0, 3.0)
        ws += distribute_words(tokenize("seven eight nine ten eleven twelve"), 8.0, 11.0)
        cues = build_cues(ws, 12.0, self.s)
        self.assertTrue(any(c.start >= 8.0 - 1e-6 for c in cues))
        self.assertTrue(all(not (c.start < 3.0 and c.end > 5.0) for c in cues))

    def test_captions_never_exceed_audio_even_if_words_do(self):
        ws = words(14.0)
        cues = build_cues(ws, 10.0, self.s)   # audio shorter than the last words
        self.assertTrue(all(c.end <= 10.0 for c in cues))
        self.assertEqual(validate_srt(to_srt(cues), 10.0, self.s), [])

    def test_validator_catches_bad_srt(self):
        bad = ("1\n00:00:02,000 --> 00:00:01,000\nx\n\n"
               "2\n00:00:00,500 --> 00:00:03,000\noverlap\n\n"
               "3\n00:00:03,000 --> 00:00:03,000\nzero\n\n"
               "4\n00:00:05,000 --> 00:00:99,000\ntoo long\n")
        problems = " ".join(validate_srt(bad, 10.0))
        for expected in ("duration", "overlaps", "after audio end"):
            self.assertIn(expected, problems)
        self.assertTrue(validate_srt("garbage", 10.0))
        self.assertTrue(validate_srt("", 10.0))

    def test_roundtrip(self):
        parsed = parse_srt(self.srt)
        self.assertEqual(len(parsed), len(self.cues))
        self.assertAlmostEqual(parsed[0].start, self.cues[0].start, places=3)


if __name__ == "__main__":
    unittest.main()
