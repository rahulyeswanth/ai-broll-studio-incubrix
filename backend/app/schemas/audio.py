from typing import List, Optional
from pydantic import BaseModel, Field, field_validator

from app.schemas.domain import SUPPORTED_ASPECT_RATIOS, SceneHintInput


class NarrationRequest(BaseModel):
    script: str = Field(..., min_length=1, max_length=10000, description="Script to convert into narration")
    provider: str = Field(default="mock", description="TTS provider: 'mock' (no audio file) or 'edge' (real narration audio via edge-tts)")
    voice_id: Optional[str] = Field(default=None, description="Optional provider-specific voice identifier")
    target_aspect_ratio: str = Field(default="16:9")
    scene_hints: List[SceneHintInput] = Field(default_factory=list)

    @field_validator("script")
    @classmethod
    def clean_script(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Script cannot be empty.")
        return value

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, value: str) -> str:
        value = value.lower().strip()
        if value not in {"mock", "edge"}:
            raise ValueError("Supported TTS providers are 'mock' (placeholder, no audio) and 'edge' (real narration).")
        return value

    @field_validator("target_aspect_ratio")
    @classmethod
    def validate_aspect_ratio(cls, value: str) -> str:
        if value not in SUPPORTED_ASPECT_RATIOS:
            raise ValueError(f"Unsupported aspect ratio '{value}'.")
        return value


class NarrationSceneResult(BaseModel):
    scene_id: str
    scene_index: int
    narration_text: str
    audio_path_or_url: str
    start_time: float = Field(ge=0)
    end_time: float = Field(gt=0)
    duration_seconds: float = Field(gt=0)


class NarrationResponse(BaseModel):
    provider_used: str
    target_aspect_ratio: str
    total_scenes: int
    total_duration_seconds: float
    audio_is_placeholder: bool = Field(description="True when mock audio references are used instead of real audio files")
    audio_file: Optional[str] = Field(default=None, description="Path of the combined narration file when a real provider was used")
    scenes: List[NarrationSceneResult]
    warnings: List[str] = Field(default_factory=list)
