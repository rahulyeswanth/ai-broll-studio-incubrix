import unittest
from types import SimpleNamespace as NS

from app.pipeline.diagnostics import build_diagnostics, friendly_error
from app.pipeline.visual import NullVisualVerifier, UnavailableVisualVerifier


def dec(sid, status, selected=True, reason="r", log=None, alt=False, reused=False):
    sel = NS(final_score=0.4, label="Low") if selected else None
    return NS(scene_id=sid, status=status, selected=sel, reason=reason, threshold=0.45, label=sel.label if sel else "None", attempts=[],
              download_log=log or [], used_alternate=alt, reused_clip=reused)


class DiagnosticsTests(unittest.TestCase):
    def build(self, decisions, raw=(), verifier=None, visual=False, **kw):
        return build_diagnostics(decisions, raw_warnings=raw, events=[], verifier=verifier or NullVisualVerifier(), visual_performed=visual, **kw)

    def test_clean_run_is_one_success_line_plus_unverified_note(self):
        d = self.build([dec("scene-001", "matched")])
        self.assertEqual(d["summary"][0], {"level": "success", "text": "Video generated successfully."})
        self.assertEqual(len(d["summary"]), 2)

    def test_visually_verified_clean_run_has_no_unverified_note(self):
        d = self.build([dec("scene-001", "matched")], visual=True)
        self.assertEqual([s["text"] for s in d["summary"]], ["Video generated successfully."])

    def test_counts_and_singular_plural_wording(self):
        d = self.build([dec("a", "fallback_low_confidence"), dec("b", "fallback_low_confidence"), dec("c", "unmatched", selected=False), dec("d", "matched", alt=True)])
        texts = [s["text"] for s in d["summary"]]
        self.assertIn("1 scene has no matching footage.", texts)
        self.assertIn("2 scenes need review.", texts)
        self.assertIn("One scene uses an alternate clip.", texts)
        self.assertNotIn("Video generated successfully.", texts)

    def test_repeated_messages_are_deduplicated(self):
        log = [{"clip": "x", "outcome": "download_failed", "reason": "HTTP 404"}]
        d = self.build([dec("s1", "matched", log=log), dec("s1", "matched", log=log)], raw=["same", "same"])
        msgs = [(i["code"], i["scene_id"], i["message"]) for i in d["items"]]
        self.assertEqual(len(msgs), len(set(msgs)))

    def test_failures_are_not_suppressed(self):
        log = [{"clip": "pixabay:1", "outcome": "download_failed", "reason": "HTTP 404 for clip"}]
        d = self.build([dec("s1", "matched", log=log, alt=True)])
        self.assertTrue(any("HTTP 404" in i["message"] for i in d["items"]))

    def test_unavailable_visual_check_is_reported(self):
        d = self.build([dec("s1", "matched")], verifier=UnavailableVisualVerifier("clip", "torch missing"))
        self.assertTrue(any(i["code"] == "visual_unavailable" and "torch missing" in i["message"] for i in d["items"]))

    def test_friendly_error_mapping(self):
        self.assertIn("API key", friendly_error("provider_unavailable", "x"))
        self.assertEqual(friendly_error("unknown_code", "raw message"), "raw message")
        self.assertTrue(friendly_error(None, None))


if __name__ == "__main__":
    unittest.main()
