from functools import lru_cache
from typing import Literal, List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    APP_NAME: str = "AI B-roll Studio"
    APP_ENV: Literal["development", "staging", "production", "test"] = "development"
    DEBUG: bool = False
    API_V1_PREFIX: str = "/api/v1"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    LOG_LEVEL: str = "INFO"

    # Storage config
    STORAGE_TYPE: Literal["local", "minio"] = "local"
    STORAGE_LOCAL_DIR: str = "./storage_data"
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET_NAME: str = "broll-assets"
    MINIO_SECURE: bool = False

    # Redis config
    REDIS_URL: str = "redis://localhost:6379/0"

    # Audio validation constraints
    MAX_AUDIO_FILE_SIZE_MB: int = 50
    ALLOWED_AUDIO_EXTENSIONS: List[str] = [".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"]

    # Script Ingestion & Segmentation Settings
    MAX_SCRIPT_CHAR_LENGTH: int = 10000
    DEFAULT_WORDS_PER_MINUTE: int = 140
    MIN_SCENE_DURATION_SECONDS: float = 2.0
    MAX_SCENE_DURATION_SECONDS: float = 12.0

    # Provider selections
    TTS_PROVIDER: str = "mock"
    TRANSCRIPTION_PROVIDER: str = "mock"
    FOOTAGE_PROVIDER: str = "mock"

    # External Provider API Keys
    PEXELS_API_KEY: str | None = None
    
    PIXABAY_API_KEY: str | None = None

    COVERR_API_KEY: str | None = None

    # End-to-end pipeline (see app/pipeline/config.py, which reads the same .env). Declared here so they are
    # documented and validated; the pipeline re-reads them from the environment.
    STORAGE_ROOT: str = "./storage"
    ALLOW_MOCK_PROVIDERS: bool = False
    TTS_DEFAULT_VOICE: str = "en-US-AriaNeural"
    DEFAULT_LANGUAGE: str = "en"
    WHISPER_MODEL_SIZE: str = "base"
    MATCH_THRESHOLD: float = 0.45
    VISUAL_VERIFIER: str = "metadata"
    MATCH_FALLBACK_POLICY: str = "best_available"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
