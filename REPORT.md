# Report

## Done and verified (offline, mock providers, real FFmpeg and real audio fixtures)
Transcription fix; ASS caption layout for 3 ratios; `POST /api/v1/broll/jobs`; multi-ratio output; error envelope (UPPER_SNAKE_CASE codes); Idempotency-Key on every POST via middleware; /health and /ready with real dependency checks; JSON logs with X-Correlation-Id and job_id; minimum scene length (merge) on both input paths; per-clip brightness/saturation normalisation; two-stage frame-exact renderer; multi-provider footage search with a stored selection rule (`provider_selection`); swap-a-clip endpoint that re-renders only that scene; SSRF guard; rate limit; CORS; ownership; callbacks; avoid rules; sense-aware matching. 347 tests pass, 4 skipped (backend suite, `--ignore=tests/frontend`).

## Not verified / not done
Live URL; real-footage run and relevance ratings (no API key; no human ratings, so no relevance average is claimed); Docker and MinIO runs; cost and performance metrics (not measured); 10-minute audio end-to-end timing; live Pexels calls (second source tested with test doubles only); swap timings on real footage (only synthetic-clip timings: 1.25 s vs 2.46 s full render); normalisation on real stock footage; VLM verification; Hindi/Tamil; frontend update for the new endpoint.

## Caption design
Serif font, about 32 px at 1080p (35 px for 9:16), bottom-centred, split into cues rather than shrunk. Override with CAPTION_FONT_PX.
