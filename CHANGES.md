# Phase 8 - premium UI, action-aware matching, honest failure handling

## UI
* **Root cause of the overlapping text**: a global `font-family: Inter !important` on every `span`/`div` replaced Streamlit's Material
  icon font, so ligature names (`keyboard_arrow_right`, `arrow_down`) were printed over expander labels. Fonts are now set on text
  containers only and icon spans are pinned to the Material font (`theme.css`).
* Other defects fixed: panels opened/closed across separate `st.markdown` calls (empty "Quick Actions"), no dark base theme
  (`.streamlit/config.toml`), hero repeated on every page, Streamlit's 44px headings out-ranking the type scale, sidebar covering the
  page on phones (`initial_sidebar_state="auto"`).
* New Studio workflow page: header with status chips, input workspace, fragment-polled progress (stages, elapsed, scene n of m,
  inputs disabled while running), result workspace, proportional timeline, filterable scene review with thumbnails, collapsed
  "Generation details". Legacy pages kept under *Tools*. Pure presentation logic is in `ui_helpers.py` (unit-tested).

## Matching
* `lexicon.py`: activity frames (action/instrument evidence, foil actions, alternative queries) + role synonyms.
* Analyser: recognises more verbs, keeps a second coordinated action, separates recipients ("...to students"), fixes duplicated
  words ("programming code code"), adds action-specific alternative queries.
* Scoring: per-element evidence (subject/action/object/setting) with documented weights; gates stop right-people/wrong-action clips
  from passing; High/Medium/Low labels instead of pseudo-confidence; invalid URLs rejected.
* Policies: `review` (default) / `best_available` (legacy) / `fail`; `MATCH_REVIEW_FLOOR`. Unmatched scenes get a placeholder and the job
  still completes unless *no* scene matched.
* Downloads: next-best *passing* candidate on failure; near-duplicate footage detected by frame hash; every attempt logged.
* Visual verifier: contrastive positive-vs-foil scoring behind a pluggable embedding backend; unavailable model degrades with a reason
  instead of aborting the job.

## Backend / API
* Duplicate identical submissions while a job runs return the existing job (`deduplicated: true`).
* `diagnostics` (short summary + deduplicated items + counts) in the manifest and `summary` in job status; `user_message` on failures;
  per-scene JPEG thumbnails (`thumb_scene-NNN.jpg`, whitelisted, served as `image/jpeg`); Pexels/Pixabay missing-key messages name the env var.

## Behaviour changes you should know about
* Default `MATCH_FALLBACK_POLICY` is now `review`. **Update your own `backend/.env`** (the old example pinned `best_available`).
* A scene with only a generic clip (e.g. an ocean clip for "sea turtles swim") is now *unmatched* rather than silently filled.
  One legacy test was changed to opt into `best_available`; `test_config.py` now ignores the developer's local `.env`.
* **`backend/.env` was present in the uploaded archive despite an earlier note saying it was removed. Rotate its keys.**

---

# CHANGES

## New: one-call pipeline (`backend/app/pipeline/`)
* `POST /api/v1/jobs` (script and/or audio upload) → background job → `GET /jobs/{id}` progress → file downloads. Output per job:
  `video.mp4`, narration audio, `captions.srt`, `timeline.json`, `manifest.json`.
* **Scene understanding** replaces 2–4-keyword extraction for the new flow: subject/action/setting/objects, a separate visual
  description, 4–6 specific→broad queries, avoid terms. Rule-based (not an LLM); `LLMSceneAnalyzer` hook for a real model.
* **Matching**: multi-candidate retrieval, filtering (duplicates, bad hosts, short/low-res, non-commercial, AI-flagged), documented
  scoring, threshold, query retries, no-duplicate-clips preference, disclosed low-confidence fallback or failure.
  Visual verification adapter (`NullVisualVerifier` default; optional OpenCLIP). Without it, results are labelled
  `metadata_text_fallback` everywhere.
* **Narration**: real Edge TTS (per scene, joined gaplessly, validated playable, real durations). Mock TTS is placeholder tones,
  refused unless `ALLOW_MOCK_PROVIDERS=true`, flagged in warnings/manifest.
* **Uploaded audio**: validated (type, size, duration, decodability, silence), transcribed with faster-whisper word timestamps,
  scenes aligned to real speech, uploaded audio stays the timing source of truth; documented script-vs-audio reconciliation rule.
* **Captions**: word-timed cues, ≤2 lines, never overlap/zero-length/past audio end; SRT re-parsed and validated; timing method
  and precision disclosed in the manifest.
* **Rendering**: new renderer (H.264 high/yuv420p/CFR 30 fps + AAC 48 kHz), scenes sized to the narration, short clips looped
  (disclosed) instead of freezing, subject-aware crop (faces / edge-energy) for 9:16 and 1:1, optional burned captions.
* **Output validation**: ffprobe checks (codec, pix_fmt, resolution, fps, streams, durations) + blackdetect/freezedetect.
* **Licensing/attribution** metadata per clip; commercial status only from documented provider policy, otherwise `unknown`.
* Job store/runner (threads, `job.json`), restart recovery, temp cleanup, safe download whitelist, structured errors.

## Changed (existing code)
* Streamlit **One-Click Workflow** rewritten: script/audio input, voice + caption options, one "Generate Video" button, live progress,
  results (video, audio, SRT, timeline, manifest, warnings, per-scene decisions), no `st.stop()`, no hard-coded filenames.
  Narration page can now request real audio. `BROLL_API_URL` env var overrides the backend URL. Other pages untouched.
* `BrollAsset` gained optional `page_url`, `tags`, `ai_generated`. Pixabay: reports the clip's actual aspect ratio (was echoing the
  request), exposes tags/page URL, optional non-strict orientation (default strict = old behaviour). Pexels: page URL.
* `/audio/narrate` accepts `provider: "edge"` (real audio file). `mock` behaves as before.
* Legacy `/video/render-from-broll` and `VideoRenderingService` use a storage path anchored to `backend/` instead of the working
  directory; legacy render adds `yuv420p`, 30 fps, 48 kHz audio flags (otherwise unchanged).
* Dependencies: added `edge-tts`, `faster-whisper`, `opencv-python-headless`, `numpy`; `requirements-frontend.txt`, `requirements-optional.txt`.
* Docker: FFmpeg installed, real provider defaults, storage volume. **Not built/tested.**
* `.env.example` added (placeholders). **The original `.env` files were removed from this package; they contained credentials.**
  Create `backend/.env` from the template and re-enter your key.

## Unchanged on purpose
Legacy endpoints and pages (`/scripts/ingest`, `/broll/match`, `/captions/generate`, `/video/render`, health), their schemas and tests,
and the sample clips/audio in `backend/storage/source_*`. Previously rendered demo videos were left out of the ZIP to keep it small.

## Assessment 3
See IMPLEMENTATION_AUDIT.md for the file-by-file list. Added: `app/core/{errors,security}.py`, `pipeline/{caption_layout,multi}.py`, `app/storage/*`, `/api/v1/broll/jobs`, `tools/batch_run.py`, `examples/test_pack/`, `openapi.yaml`, `.env.example`.
