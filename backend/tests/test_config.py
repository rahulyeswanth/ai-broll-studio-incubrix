from app.core.config import Settings


def test_settings_defaults():
    settings = Settings(_env_file=None)   # hermetic: test the code defaults, not a developer's local backend/.env
    assert settings.APP_NAME == "AI B-roll Studio"
    assert settings.API_V1_PREFIX == "/api/v1"
    assert settings.PORT == 8000
    assert settings.TTS_PROVIDER == "mock"
