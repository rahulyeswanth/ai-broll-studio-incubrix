from app.providers.base import TTSProvider
from app.providers.mocks import MockTTSProvider


def get_tts_provider(provider_name: str = "mock") -> TTSProvider:
    if provider_name.lower() == "mock":
        return MockTTSProvider()
    raise NotImplementedError(f"TTS provider '{provider_name}' is not configured.")
