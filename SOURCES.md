# Sources and licences

Only items actually used are listed. Licences below are not legal advice; verify before commercial use.

- Footage: Pixabay and Pexels via their APIs. Each clip's licence/attribution is recorded per scene in `timeline.json` from provider metadata. Their licence terms must be reviewed by the operator. No footage was downloaded in the authoring environment (no API key).
- Narration: edge-tts (Microsoft Edge online voices) at runtime; terms are Microsoft's and should be reviewed for commercial use. Test speech fixtures were synthesised offline with Piper (see `backend/tests/fixtures/audio/README.md`).
- ASR: faster-whisper (CTranslate2 Whisper models). Libraries: FastAPI, Starlette, Pillow, PyAV, MinIO client, Streamlit (each under its own open-source licence; see package metadata).
- Fonts: DejaVu (installed in the Docker image via `fonts-dejavu-core`).
- FFmpeg: system package; a static build was used for local tests.
- No background music is bundled.
