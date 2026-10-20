# TEST REPORT (Phase 8)

Environment: Linux sandbox, Python 3.12, FFmpeg/FFprobe (libx264, aac, libass), Playwright Chromium. **No access to Pixabay/Pexels,
Edge TTS or HuggingFace**, no torch/open_clip. Everything below was actually run; limits are stated in section 3.

## 1. Executed
| Run | Result |
|---|---|
| Baseline before changes (`pytest`, live-provider file excluded) | 155 passed, 2 failed (a config test that read the developer's real `.env`; a Pexels error-message regression) |
| **Full suite after changes** (`pytest --ignore=tests/test_real_providers_integration.py`) | **231 passed, 0 failed** in 7m27s |
| After final small edits (sidebar setting, Pillow call): `tests/frontend` + `test_unmatched_scenes.py` (concurrency test deselected) | 36 passed |

New test files: `pipeline/test_action_matching.py` (15), `pipeline/test_visual_verifier.py` (9), `pipeline/test_unmatched_scenes.py` (13),
`pipeline/test_diagnostics.py` (7), `test_jobs_api_phase8.py` (5), `frontend/test_ui_helpers.py`, `frontend/test_frontend_states.py` (8, Streamlit
AppTest: empty, offline, validation, busy, failed, completed, long text, legacy pages), `frontend/test_layout_browser.py` (real Chromium at
1366/1440/1920/390 px: no horizontal overflow, icon font intact, type scale, long text clamped, diagnostics collapsed).

Existing tests changed: `test_config.py` now ignores a local `.env` (tests code defaults); one end-to-end test now sets
`fallback_policy="best_available"` because the default policy intentionally no longer fills a scene with a generic clip. No assertions were weakened otherwise.

## 2. Manual end-to-end run (real FastAPI + Streamlit, test-only fake catalogue, `tests/e2e_harness.py`)
Five-domain script -> 3 matched (High), 1 "needs review" (doctor/patient clip tagged consultation: right people, wrong action), 1 "no footage" (teacher) with
placeholder. Job completed; MP4 verified by ffprobe: H.264 High / yuv420p, AAC, 1920x1080, 30 fps, 17.17 s. A duplicate submission returned the same job id.
UI screenshots reviewed: no overlapping text; type scale 30/16/14/12 px confirmed via computed styles.

## 3. NOT done / limits
* **No real frame-level verification**: the contrastive logic is tested with a fake embedding backend over real FFmpeg frames. OpenCLIP was never run.
* **No live Pixabay/Pexels request**; footage in all runs came from locally generated clips and synthetic title/tags.
* Narration in the manual run used the placeholder tone TTS (disclosed `is_real: false`); the Edge TTS code path is tested with a stub only.
* In-browser video playback could not be shown (sandbox browsers lack H.264); the file itself was validated with ffprobe/decoding.
* Swap-a-clip is tested end to end on synthetic clips only (4-scene job: 80 of 308 frames re-encoded, 1.25 s vs 2.46 s full render); real-footage timings are not measured.
* Docker files were not built.


## Assessment 3 run (authoring environment)
Command: `cd backend && python -m pytest -q -p no:cacheprovider`
Result: **347 passed, 4 skipped** in 179 s (includes `tests/test_part_a.py`: 17 tests for error codes, idempotency, readiness, logs, min scene length, normalisation, provider fallback, swap) (Python 3.12, static FFmpeg, mock footage providers).
Not run: Docker, MinIO, real footage providers, live URL, test pack end-to-end.
