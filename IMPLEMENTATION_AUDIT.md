# Implementation audit

| Requirement | Finding (before) | Evidence | Fix | Final result |
|---|---|---|---|---|
| Audio transcription | Failed: `av.open() got unexpected keyword 'metadata_errors'`. faster-whisper's `decode_audio` passes it; PyAV 19.0.1 removed it; faster-whisper declares only `av>=11`. | Installed-version check and signature inspection of `av.open` | FFmpeg decodes to 16 kHz mono float32 PCM which is passed to `transcribe()`; `av>=12,<19` pinned (`backend/app/pipeline/media.py`, `transcription.py`, `requirements.txt`) | `tests/test_audio_inputs.py` 23 tests; Whisper-dependent ones skip with a reason if the model is unavailable |
| Caption overflow | libass converts SRT to ASS on a 384x288 canvas, so `force_style` sizes were rescaled | Rendered frames | ASS with PlayRes = real frame size, PIL-measured wrapping, cue splitting (`pipeline/caption_layout.py`, `renderer.py`) | 68 tests incl. FFmpeg frame bbox checks on dark and bright backgrounds. 16:9 and 1:1 frames were bbox-tested, not eyeballed |
| `app.storage` package | Missing from the supplied zip (imports broke) | import error | Reconstructed `app/storage/*` | storage tests pass |
| Existing tests | Mock-only; did not exercise real audio or FFmpeg caption rendering | review | Added real-audio and real-FFmpeg tests | 347 passed, 4 skipped |
| `POST /api/v1/broll/jobs` | Missing | route listing | Added with validation, idempotency, multipart | `tests/test_broll_jobs_api.py` 24 tests |
| Multi-ratio, avoid, error envelope, SSRF, rate limit, CORS, ownership, callbacks | Missing | | Added (see README) | tested offline |
| Word-sense matching | Wrong sense picked (python snake as code, mouse as animal, river bank unmatched) | `tests/pipeline/test_hard_cases.py` failed | `SENSE_CLUSTERS` penalty/bonus in `ranking.py` | 11 tests pass; metadata only, mock providers |
| Real footage relevance | Not measurable here | no footage API key | n/a | **Not verified** |
| Docker | compose/Dockerfile reviewed and edited (non-root, fonts, MinIO creds from env, test profile) | | | **Not run** |
| Frontend | Not updated for the new endpoint | | | Still uses legacy `/api/v1/jobs` (still supported) |
| Part A: UPPER_SNAKE_CASE codes | Lowercase codes | `core/errors.py` | `wire_code`/`public_error` at the API boundary; README table | `test_part_a.py` (codes documented, envelope) |
| Part A: Idempotency-Key on every POST | Only `/broll/jobs` | `core/idempotency.py` | One ASGI middleware, 409 on conflict/in-flight | `test_part_a.py` (replay, conflict, legacy POST, failures not cached) |
| Part A: /health, /ready, JSON logs | Readiness checked Redis/storage only | `endpoints/health.py`, `core/middleware.py` | ffmpeg, ffprobe, jobs dir, storage, optional Redis; access log with correlation_id and job_id | `test_part_a.py` |
| Part A: minimum scene length | Only on the audio path | `alignment.merge_short_scenes` | Applied on the script-only path too | `test_part_a.py` |
| Part A: colour normalisation | Missing | `pipeline/normalize.py`, `renderer.py` | Median-target `eq` per clip, bounded; stored in timeline | `test_part_a.py` (synthetic clips; no real footage) |
| Part A: second source + rule | Adapters existed, no rule | `pipeline/multi_provider.py` | Documented rule; `provider_selection` per scene | `test_part_a.py` (test doubles; live Pexels not called) |
| Part A: swap a clip | Missing | `pipeline/swap.py`, `broll.py` | Scene-only re-encode + stream-copy stitch | `test_part_a.py` (synthetic: 1.25 s vs 2.46 s) |
