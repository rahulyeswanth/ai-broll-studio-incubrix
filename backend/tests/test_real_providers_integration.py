"""OPTIONAL integration tests against REAL providers. Skipped unless explicitly enabled:

    $env:RUN_REAL_PROVIDER_TESTS = "1"      # PowerShell
    pytest tests/test_real_providers_integration.py -v

Needs: internet access, `pip install edge-tts` (TTS test), PIXABAY_API_KEY (footage test),
`pip install faster-whisper` (ASR test, downloads a model on first run)."""
import asyncio
import importlib.util
import os
import tempfile
from pathlib import Path

import pytest

from app.pipeline.config import load_settings
from app.pipeline.media import audio_duration, validate_audio_file

pytestmark = pytest.mark.skipif(os.environ.get("RUN_REAL_PROVIDER_TESTS") != "1", reason="set RUN_REAL_PROVIDER_TESTS=1 to run")


@pytest.mark.skipif(importlib.util.find_spec("edge_tts") is None, reason="edge-tts not installed")
def test_real_edge_tts_produces_playable_speech():
    from app.pipeline.narration import EdgeTTSProvider, narrate_scenes
    s = load_settings()
    out = Path(tempfile.mkdtemp())
    res = asyncio.run(narrate_scenes(["Sea turtles swim through the open ocean.", "Coral reefs shelter colorful fish."],
                                     EdgeTTSProvider(), s.default_voice, "en", out, s))
    assert res.is_real and res.duration_seconds > 2
    validate_audio_file(res.audio_path, s)       # exists, non-empty, decodable, not silent


@pytest.mark.skipif(not load_settings().pixabay_api_key, reason="PIXABAY_API_KEY not configured")
def test_real_pixabay_search_returns_candidates():
    from app.providers.pixabay import PixabayFootageProvider
    from app.pipeline.provider_adapter import candidate_from_asset
    s = load_settings()
    assets = asyncio.run(PixabayFootageProvider(s.pixabay_api_key, strict_orientation=False).search_footage("sea turtle swimming", "16:9", 5))
    assert assets
    c = candidate_from_asset(assets[0])
    assert c.download_url.startswith("https://") and c.license_basis == "provider_policy_documented"


@pytest.mark.skipif(importlib.util.find_spec("faster_whisper") is None, reason="faster-whisper not installed")
def test_real_whisper_transcribes_tts_audio():
    from app.pipeline.narration import EdgeTTSProvider, narrate_scenes
    from app.pipeline.transcription import FasterWhisperTranscriber
    s = load_settings()
    out = Path(tempfile.mkdtemp())
    nar = asyncio.run(narrate_scenes(["Sea turtles swim through the ocean."], EdgeTTSProvider(), s.default_voice, "en", out, s))
    asr = asyncio.run(FasterWhisperTranscriber(s.whisper_model_size).transcribe(nar.audio_path, "en"))
    assert "turtle" in asr.text.lower() and asr.granularity == "word"
