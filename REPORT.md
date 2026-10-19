# Report

## Done and verified (offline, mock providers, real FFmpeg and real audio fixtures)
Transcription fix; ASS caption layout for 3 ratios; `POST /api/v1/broll/jobs`; multi-ratio output; error envelope; idempotency; SSRF guard; rate limit; CORS; ownership; callbacks; avoid rules; sense-aware matching. 349 tests pass, 4 skipped.

## Not verified / not done
Live URL; real-footage run and relevance ratings (no API key; no human ratings, so no relevance average is claimed); Docker and MinIO runs; cost and performance metrics (not measured); 10-minute audio end-to-end timing; swap-a-clip; colour normalisation; VLM verification; Hindi/Tamil; frontend update for the new endpoint.

## Caption design
56 px (16:9, 1:1) and 60 px (9:16) at 1080p class, SAFETY 0.97, split into cues rather than shrunk. See README for the deviation from the brief's ranges.
