# Test pack

Ten script inputs (JSON bodies for `POST /api/v1/broll/jobs`) covering ocean/environment, ambiguous terms
(cloud, bank x2, Python), numbers, abstract narration, no-footage, long words and Unicode.
Audio inputs (MP3/WAV/M4A, silent, music-only, corrupt, wrong-extension, empty) live in
`backend/tests/fixtures/audio/` (speech fixtures synthesised offline with Piper; see its README).

Run: `python tools/batch_run.py examples/test_pack --out results --api http://localhost:8000`
Requires a real footage key (PIXABAY_API_KEY) and a TTS provider with network access.
**This pack has NOT been run end-to-end against real footage in the authoring environment**
(no footage API key; TTS network blocked). Offline behaviour is covered by the pytest suite using mock providers.
