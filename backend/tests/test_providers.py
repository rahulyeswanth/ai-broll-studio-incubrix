import pytest
from app.providers.mocks import MockTTSProvider, MockTranscriptionProvider, MockFootageProvider


@pytest.mark.asyncio
async def test_mock_tts_provider():
    provider = MockTTSProvider()
    result = await provider.synthesize("A sample script for AI B-roll Studio testing.")
    assert result.audio_path_or_url.startswith("mock://")
    assert result.duration_seconds > 0


@pytest.mark.asyncio
async def test_mock_transcription_provider():
    provider = MockTranscriptionProvider()
    result = await provider.transcribe("fake_audio.mp3")
    assert len(result.segments) == 2
    assert result.segments[0].start_time == 0.0


@pytest.mark.asyncio
async def test_mock_footage_provider():
    provider = MockFootageProvider()
    assets = await provider.search_footage(query="office technology", aspect_ratio="16:9", max_results=3)
    assert len(assets) == 3
    assert assets[0].commercial_use_allowed is True
    assert assets[0].license_type == "CC0-Commercial-Safe"
