import os
import tempfile
import time
import unittest
from pathlib import Path

from app.pipeline.config import PipelineSettings, load_settings
from app.pipeline.jobs import JobRunner, JobStore, downloadable_name
from app.pipeline.narration import PlaceholderToneTTS
from app.pipeline.orchestrator import Deps, JobRequest

from .helpers import CatalogSearch, FakeTranscriber, LocalFileServer, asset, make_clip, make_settings


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.store = JobStore(make_settings(self.root))

    def test_job_ids_are_validated(self):
        for bad in ("../etc", "", "ABC", "a" * 11, "zzzzzzzzzzzz", "a/b"):
            with self.assertRaises(ValueError):
                self.store.job_dir(bad)
            self.assertIsNone(self.store.read(bad))

    def test_download_whitelist_blocks_traversal_and_unknown_files(self):
        jid = self.store.new_id()
        d = self.store.job_dir(jid); d.mkdir(parents=True)
        (d / "video.mp4").write_bytes(b"x"); (d / "secret.txt").write_text("s"); (d / "narration.mp3").write_bytes(b"x")
        self.assertIsNotNone(downloadable_name(d, "video.mp4"))
        self.assertIsNotNone(downloadable_name(d, "narration.mp3"))
        for bad in ("secret.txt", "../video.mp4", "..%2fvideo.mp4", "narration.exe", "manifest.json"):
            self.assertIsNone(downloadable_name(d, bad), bad)

    def test_recover_interrupted_and_cleanup(self):
        jid = self.store.new_id()
        self.store.write(jid, status="processing", progress=40)
        self.assertEqual(self.store.recover_interrupted(), 1)
        st = self.store.read(jid)
        self.assertEqual(st["status"], "failed")
        self.assertEqual(st["error"]["code"], "interrupted")
        tmp = self.store.job_dir(jid) / "tmp"; tmp.mkdir()
        old = time.time() - 48 * 3600
        os.utime(tmp, (old, old))
        self.assertEqual(self.store.cleanup(max_tmp_age_hours=24)["tmp_dirs"], 1)
        self.assertFalse(tmp.exists())


class RunnerTests(unittest.TestCase):
    def test_runner_completes_and_records_failure(self):
        root = Path(tempfile.mkdtemp())
        s = make_settings(root)
        srv_dir = root / "srv"; srv_dir.mkdir()
        make_clip(srv_dir / "turtle.mp4", 4)
        with LocalFileServer(srv_dir) as srv:
            entries = [asset("turtle", "sea, turtle, turtles, ocean, swimming", srv.url("turtle.mp4"))]
            def deps(req):
                return Deps(settings=s, search=CatalogSearch(entries), footage_provider_name="pixabay",
                            get_tts=lambda: PlaceholderToneTTS(), get_transcriber=lambda: FakeTranscriber("x"))
            store = JobStore(s)
            runner = JobRunner(store, deps, max_workers=1)
            ok, bad = store.new_id(), store.new_id()
            runner.submit(ok, JobRequest(script_text="Sea turtles swim through the ocean.", burn_captions=False), "script_only")
            runner.submit(bad, JobRequest(script_text="   "), "script_only")
            runner.pool.shutdown(wait=True)
            a, b = store.read(ok), store.read(bad)
            self.assertEqual(a["status"], "completed", a)
            self.assertEqual(a["progress"], 100)
            self.assertTrue((store.job_dir(ok) / "video.mp4").is_file())
            self.assertEqual(b["status"], "failed")
            self.assertEqual(b["error"]["code"], "invalid_input")


class ConfigTests(unittest.TestCase):
    def test_env_precedence_and_no_secret_leak(self):
        d = Path(tempfile.mkdtemp())
        (d / ".env").write_text("TTS_PROVIDER=mock\nPIXABAY_API_KEY=file-key\nMATCH_THRESHOLD=0.7\n")
        s = load_settings(env={"TTS_PROVIDER": "edge"}, dotenv_path=d / ".env")
        self.assertEqual(s.tts_provider, "edge")          # explicit env wins
        self.assertEqual(s.match_threshold, 0.7)          # .env value used
        self.assertEqual(s.pixabay_api_key, "file-key")
        public = str(s.public_dict()) + repr(s)
        self.assertNotIn("file-key", public)
        self.assertNotIn(str(s.storage_root), str(s.public_dict()))

    def test_defaults_are_not_machine_specific(self):
        s = load_settings(env={}, dotenv_path=Path("/nonexistent/.env"))
        self.assertTrue(s.storage_root.is_absolute())
        self.assertEqual(s.jobs_dir.name, "jobs")


if __name__ == "__main__":
    unittest.main()
