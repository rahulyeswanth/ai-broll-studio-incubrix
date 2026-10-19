
from typing import List, Literal

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
)

from app.schemas.domain import BrollMatchRequest

AspectRatio = Literal["16:9", "9:16", "1:1"]

VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".webm", ".avi")
AUDIO_EXTENSIONS = (".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac")


def os_path_unsafe(value: str) -> bool:
    """Reject paths when a simple filename is expected."""
    return (
        value in {".", ".."}
        or "/" in value
        or "\\" in value
        or ":" in value
    )


class RenderRequest(BaseModel):
    clips: List[str] = Field(
        min_length=1,
        max_length=30,
        description="Video filenames located in backend/storage/source_videos",
    )
    target_aspect_ratio: AspectRatio = "16:9"
    clip_durations: List[float] | None = Field(
        default=None,
        description="Optional duration in seconds for each clip",
    )
    output_filename: str = Field(
        default="rendered_video.mp4",
        min_length=5,
        max_length=100,
    )
    audio_filename: str | None = Field(
        default=None,
        description="Optional audio filename in backend/storage/source_audio",
    )
    include_audio_from_clips: bool = False
    subtitle_text: str | None = Field(
        default=None,
        max_length=200000,
        description="Optional SRT text to burn into the MP4",
    )

    @model_validator(mode="after")
    def validate_duration_count(self):
        if (
            self.clip_durations is not None
            and len(self.clip_durations) != len(self.clips)
        ):
            raise ValueError(
                "clip_durations must contain one duration for each clip."
            )
        return self

    @field_validator("clip_durations")
    @classmethod
    def validate_clip_durations(cls, values):
        if values is not None and any(
            value <= 0 or value > 3600 for value in values
        ):
            raise ValueError(
                "Each clip duration must be greater than 0 "
                "and at most 3600 seconds."
            )
        return values

    @field_validator("clips")
    @classmethod
    def validate_clips(cls, values):
        for value in values:
            if (
                not value
                or value != value.strip()
                or os_path_unsafe(value)
                or not value.lower().endswith(VIDEO_EXTENSIONS)
            ):
                raise ValueError(
                    "Each clip must be a simple video filename "
                    "with a supported extension."
                )
        return values

    @field_validator("audio_filename")
    @classmethod
    def validate_audio(cls, value):
        if value is not None and (
            value != value.strip()
            or os_path_unsafe(value)
            or not value.lower().endswith(AUDIO_EXTENSIONS)
        ):
            raise ValueError(
                "audio_filename must be a simple audio filename "
                "with a supported extension."
            )
        return value

    @field_validator("output_filename")
    @classmethod
    def validate_output(cls, value):
        if os_path_unsafe(value) or not value.lower().endswith(".mp4"):
            raise ValueError(
                "output_filename must be a simple .mp4 filename, not a path."
            )
        return value


class BrollRenderRequest(BrollMatchRequest):
    """Match footage, download selected clips, and render the final video."""

    output_filename: str = Field(
        default="pixabay_ocean_video.mp4",
        min_length=5,
        max_length=100,
    )
    audio_filename: str | None = None
    subtitle_text: str | None = Field(
        default=None,
        max_length=200000,
    )

    @field_validator("output_filename")
    @classmethod
    def validate_output_filename(cls, value):
        if os_path_unsafe(value) or not value.lower().endswith(".mp4"):
            raise ValueError(
                "output_filename must be a simple .mp4 filename, not a path."
            )
        return value

    @field_validator("audio_filename")
    @classmethod
    def validate_audio_filename(cls, value):
        if value is not None and (
            value != value.strip()
            or os_path_unsafe(value)
            or not value.lower().endswith(AUDIO_EXTENSIONS)
        ):
            raise ValueError(
                "audio_filename must be a simple audio filename "
                "with a supported extension."
            )
        return value


class RenderResponse(BaseModel):
    status: Literal["completed"] = "completed"
    output_filename: str
    output_path: str
    target_aspect_ratio: AspectRatio
    clip_count: int
    duration_seconds: float
    width: int
    height: int
    file_size_bytes: int
    message: str
