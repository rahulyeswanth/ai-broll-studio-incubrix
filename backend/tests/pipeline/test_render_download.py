import tempfile
import unittest
from pathlib import Path

from app.pipeline.downloader import download_clip
from app.pipeline.errors import DownloadError, RenderError
from app.pipeline.framing import crop_offsets
from app.pipeline.media import make_tone_audio, probe
from app.pipeline.renderer import DIMENSIONS, RenderSegment, render_video
from app.pipeline.validation import validate_output

from .helpers import LocalFileServer, ff, make_clip, make_settings


class RenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.s = make_settings(cls.tmp)
        cls.a = make_clip(cls.tmp / "a.mp4", 4)
        cls.b = make_clip(cls.tmp / "b.mp4", 1, "1280x720", 24)   # shorter than scene -> loop
        # subject (yellow box) on the right edge of a 16:9 clip
        ff("-f", "lavfi", "-i", "color=c=0x203040:s=640x360:d=3:r=25,drawbox=x=500:y=120:w=110:h=110:color=yellow:t=fill,"
           "noise=alls=20:allf=t", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(cls.tmp / "right.mp4"))
        cls.audio = make_tone_audio(cls.tmp / "n.mp3", 6.0)
        (cls.tmp / "c.srt").write_text("1\n00:00:00,000 --> 00:00:02,500\nHello\n\n2\n00:00:02,500 --> 00:00:05,900\nWorld\n", encoding="utf-8")

    def segs(self):
        return [RenderSegment("scene-001", self.a, 3.0), RenderSegment("scene-002", self.b, 3.0)]

    def test_all_aspect_ratios_have_correct_streams(self):
        for asp, (w, h) in DIMENSIONS.items():
            out = self.tmp / f"o_{w}x{h}.mp4"
            r = render_video(self.segs(), self.audio, out, asp, self.s, srt_path=self.tmp / "c.srt")
            info = probe(out)
            v = next(x for x in info["streams"] if x["codec_type"] == "video")
            a = next(x for x in info["streams"] if x["codec_type"] == "audio")
            self.assertEqual((v["width"], v["height"]), (w, h))
            self.assertEqual((v["codec_name"], v["pix_fmt"]), ("h264", "yuv420p"))
            self.assertEqual(v["r_frame_rate"], "30/1")
            self.assertEqual(a["codec_name"], "aac")
            res = validate_output(out, expected_w=w, expected_h=h, expected_duration=6.0, expect_audio=True, run_detectors=False)
            self.assertTrue(res["passed"], res["errors"])
            self.assertEqual((r.width, r.height), (w, h))

    def test_short_clip_looped_with_warning_and_duration_exact(self):
        out = self.tmp / "loop.mp4"
        r = render_video(self.segs(), self.audio, out, "16:9", self.s)
        self.assertTrue(any("looped" in w for w in r.warnings))
        self.assertTrue(r.segments[1]["looped"])
        res = validate_output(out, expected_w=1920, expected_h=1080, expected_duration=6.0, expect_audio=True, run_detectors=False)
        self.assertTrue(res["passed"], res["errors"])

    def test_vertical_crop_keeps_off_centre_subject(self):
        import cv2
        out = self.tmp / "vert.mp4"
        render_video([RenderSegment("s", self.tmp / "right.mp4", 2.0)], self.audio, out, "9:16", self.s)
        cap = cv2.VideoCapture(str(out)); cap.set(cv2.CAP_PROP_POS_FRAMES, 20); ok, frame = cap.read(); cap.release()
        self.assertTrue(ok)
        yellow = ((frame[:, :, 2] > 200) & (frame[:, :, 1] > 200) & (frame[:, :, 0] < 90)).sum()
        self.assertGreater(yellow, 20000, "subject was cropped out of the 9:16 frame")

    def test_crop_offsets_bounds(self):
        sw, sh, x, y = crop_offsets(640, 360, 1080, 1920, 0.95, 0.5)
        self.assertEqual((sw, sh), (3413, 1920))
        self.assertTrue(0 <= x <= sw - 1080 and y == 0)

    def test_validator_detects_wrong_dimensions_and_missing_audio(self):
        out = self.tmp / "silent.mp4"
        render_video(self.segs(), None, out, "16:9", self.s)
        res = validate_output(out, expected_w=1080, expected_h=1920, expected_duration=6.0, expect_audio=True, run_detectors=False)
        self.assertFalse(res["passed"])
        joined = " ".join(res["errors"])
        self.assertIn("resolution", joined)
        self.assertIn("no audio stream", joined)

    def test_validator_detects_truncated_audio_duration(self):
        short_audio = make_tone_audio(self.tmp / "short.mp3", 2.0)
        out = self.tmp / "trunc.mp4"
        render_video(self.segs(), short_audio, out, "16:9", self.s)
        res = validate_output(out, expected_w=1920, expected_h=1080, expected_duration=6.0, expect_audio=True, run_detectors=False)
        self.assertFalse(res["passed"])

    def test_freeze_detector_flags_static_video(self):
        ff("-f", "lavfi", "-i", "color=c=blue:s=640x360:d=4:r=25", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(self.tmp / "static.mp4"))
        out = self.tmp / "frozen.mp4"
        render_video([RenderSegment("s", self.tmp / "static.mp4", 4.0)], make_tone_audio(self.tmp / "t4.mp3", 4.0), out, "16:9", self.s)
        res = validate_output(out, expected_w=1920, expected_h=1080, expected_duration=4.0, expect_audio=True)
        self.assertTrue(any("frozen" in w for w in res["warnings"]))

    def test_missing_or_invalid_clip_errors_clearly(self):
        with self.assertRaises(RenderError):
            render_video([RenderSegment("s", self.tmp / "nope.mp4", 2.0)], self.audio, self.tmp / "x.mp4", "16:9", self.s)
        bad = self.tmp / "bad.mp4"; bad.write_bytes(b"not a video")
        with self.assertRaises(RenderError):
            render_video([RenderSegment("s", bad, 2.0)], self.audio, self.tmp / "x.mp4", "16:9", self.s)
        with self.assertRaises(RenderError):
            render_video([], self.audio, self.tmp / "x.mp4", "16:9", self.s)
        with self.assertRaises(RenderError):
            render_video(self.segs(), self.audio, self.tmp / "x.mp4", "4:3", self.s)


class DownloaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.srv_dir = self.tmp / "srv"; self.srv_dir.mkdir()
        make_clip(self.srv_dir / "ok.mp4", 2)
        (self.srv_dir / "fake.mp4").write_text("<html>not a video</html>")
        (self.srv_dir / "empty.mp4").write_bytes(b"")
        (self.srv_dir / "big.mp4").write_bytes((self.srv_dir / "ok.mp4").read_bytes() * 3)

    def test_valid_download(self):
        s = make_settings(self.tmp)
        with LocalFileServer(self.srv_dir) as srv:
            p = download_clip(srv.url("ok.mp4"), self.tmp / "d" / "ok.mp4", s)
        self.assertTrue(p.is_file())

    def test_rejects_invalid_content_empty_missing_and_oversize(self):
        s = make_settings(self.tmp)
        with LocalFileServer(self.srv_dir) as srv:
            for name in ("fake.mp4", "empty.mp4", "missing.mp4"):
                with self.assertRaises(DownloadError, msg=name):
                    download_clip(srv.url(name), self.tmp / "d" / name, s)
                self.assertFalse((self.tmp / "d" / name).exists())
            tiny = make_settings(self.tmp, max_download_mb=0)
            with self.assertRaises(DownloadError):
                download_clip(srv.url("ok.mp4"), self.tmp / "d" / "big.mp4", tiny)

    def test_https_and_allowlist_enforced(self):
        strict = make_settings(self.tmp, require_https=True, allowed_download_hosts=("cdn.pixabay.com",))
        for url in ("http://cdn.pixabay.com/a.mp4", "https://evil.example/a.mp4", "https://user:pw@cdn.pixabay.com/a.mp4"):
            with self.assertRaises(DownloadError, msg=url):
                download_clip(url, self.tmp / "x.mp4", strict)

    def test_unreachable_host(self):
        with self.assertRaises(DownloadError):
            download_clip("http://127.0.0.1:9/x.mp4", self.tmp / "x.mp4", make_settings(self.tmp, download_timeout_seconds=2))


if __name__ == "__main__":
    unittest.main()
