# Representative outputs

`example_timeline.json`, `example_manifest.json` and `example_captions.srt` were produced by a **real run of the
pipeline in the test harness** (real FFmpeg rendering, real scene analysis/ranking/captioning/validation), but with
**test doubles for the external services**: a stubbed `edge_tts` module that emits FFmpeg-generated audio with word
boundaries, and a local HTTP server standing in for the footage CDN.

Therefore: clip IDs, creators (`tester`), tags and `127.0.0.1` download URLs are **test fixtures, not real Pixabay data**,
and `providers.tts.is_real` is `true` only because the production `EdgeTTSProvider` code path was exercised
(no speech was actually synthesised by Microsoft's service). A run against live providers produces the same structure
with real values.
