import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator

from app.providers.base import BrollAsset

# Supported aspect ratios
SUPPORTED_ASPECT_RATIOS = {"16:9", "9:16", "1:1"}
ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"}
DEFAULT_MAX_AUDIO_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB
MAX_SCRIPT_CHAR_LIMIT = 10000


class AspectRatio(str, Enum):
    LANDSCAPE = "16:9"
    PORTRAIT = "9:16"
    SQUARE = "1:1"


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class SceneHintInput(BaseModel):
    text: Optional[str] = Field(default=None, description="Optional scene narrative text")
    visual_hint: Optional[str] = Field(default=None, description="Keywords or visual description for B-roll matching")


class ScriptInput(BaseModel):
    text: str = Field(..., description="Written script text for voiceover narration")
    target_aspect_ratio: str = Field(default="16:9", description="Target video aspect ratio (16:9, 9:16, 1:1)")
    voice_id: Optional[str] = Field(default=None, description="Optional voice identifier for TTS")
    scene_hints: List[SceneHintInput] = Field(default_factory=list, description="Optional list of scene hints")

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Script text cannot be empty or contain only whitespace.")
        if len(cleaned) > MAX_SCRIPT_CHAR_LIMIT:
            raise ValueError(
                f"Script text length ({len(cleaned)} chars) exceeds maximum limit of {MAX_SCRIPT_CHAR_LIMIT} characters."
            )
        return cleaned

    @field_validator("target_aspect_ratio")
    @classmethod
    def validate_aspect_ratio(cls, v: str) -> str:
        if v not in SUPPORTED_ASPECT_RATIOS:
            raise ValueError(
                f"Invalid aspect ratio '{v}'. Supported aspect ratios are: {sorted(SUPPORTED_ASPECT_RATIOS)}"
            )
        return v


class AudioInputMetadata(BaseModel):
    filename: str = Field(..., description="Uploaded audio file name with extension")
    file_size_bytes: int = Field(gt=0, description="Audio file size in bytes")
    content_type: Optional[str] = Field(default=None, description="MIME content type")
    duration_seconds: Optional[float] = Field(default=None, gt=0.0, description="Audio duration if pre-probed")

    @field_validator("filename")
    @classmethod
    def validate_extension(cls, v: str) -> str:
        suffix = Path(v).suffix.lower()
        if suffix not in ALLOWED_AUDIO_EXTENSIONS:
            raise ValueError(
                f"Unsupported audio format '{suffix}'. Allowed formats: {sorted(ALLOWED_AUDIO_EXTENSIONS)}"
            )
        return v

    @field_validator("file_size_bytes")
    @classmethod
    def validate_file_size(cls, v: int) -> int:
        if v > DEFAULT_MAX_AUDIO_SIZE_BYTES:
            raise ValueError(
                f"Audio file size ({v} bytes) exceeds maximum limit of {DEFAULT_MAX_AUDIO_SIZE_BYTES} bytes (50MB)."
            )
        return v


class SceneSegment(BaseModel):
    scene_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8], description="Unique scene identifier")
    scene_index: int = Field(default=0, ge=0, description="1-based sequential order index of the scene")
    start_time: float = Field(ge=0.0, description="Scene start time in seconds")
    end_time: float = Field(gt=0.0, description="Scene end time in seconds")
    duration_seconds: float = Field(default=0.0, ge=0.0, description="Estimated duration in seconds")
    narration_text: str = Field(..., description="Narration transcript for this scene segment")
    broll_query: str = Field(..., description="Derived search query for B-roll retrieval")
    visual_hint: Optional[str] = Field(default=None, description="Optional visual hint provided by user")
    selected_asset: Optional[BrollAsset] = Field(default=None, description="Selected footage asset for this scene")

    @model_validator(mode="after")
    def validate_timeline(self) -> "SceneSegment":
        if self.end_time <= self.start_time:
            raise ValueError(
                f"Scene end_time ({self.end_time}s) must be strictly greater than start_time ({self.start_time}s)."
            )
        if self.duration_seconds == 0.0:
            self.duration_seconds = round(self.end_time - self.start_time, 2)
        return self


class ScriptIngestionResponse(BaseModel):
    script_text: str = Field(..., description="Sanitized original narration script")
    target_aspect_ratio: str = Field(..., description="Target aspect ratio for B-roll sourcing")
    estimated_total_duration: float = Field(..., ge=0.0, description="Estimated total narration duration in seconds")
    total_scenes: int = Field(..., ge=1, description="Number of segmented scenes")
    scenes: List[SceneSegment] = Field(..., description="List of segmented scenes with timing and queries")


class CaptionSegment(BaseModel):
    start_time: float = Field(ge=0.0, description="Caption start time in seconds")
    end_time: float = Field(gt=0.0, description="Caption end time in seconds")
    text: str = Field(..., description="Caption text string")

    @model_validator(mode="after")
    def validate_caption_timing(self) -> "CaptionSegment":
        if self.end_time <= self.start_time:
            raise ValueError("Caption end_time must be strictly greater than start_time.")
        return self


class GeneratedVideoMetadata(BaseModel):
    video_id: str = Field(..., description="Unique generated video identifier")
    storage_path: str = Field(..., description="URI or path in object storage")
    download_url: str = Field(..., description="Direct or signed download URL")
    aspect_ratio: str = Field(..., description="Video aspect ratio (16:9, 9:16, 1:1)")
    duration_seconds: float = Field(gt=0.0, description="Rendered video duration in seconds")
    file_size_bytes: int = Field(gt=0, description="Final video file size in bytes")
    width: int = Field(gt=0, description="Video width in pixels")
    height: int = Field(gt=0, description="Video height in pixels")
    fps: float = Field(default=30.0, gt=0.0, description="Frames per second")
    scenes_count: int = Field(ge=1, description="Number of scenes in video")
    captions_count: int = Field(default=0, ge=0, description="Number of caption segments burned or attached")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @field_validator("aspect_ratio")
    @classmethod
    def validate_video_aspect_ratio(cls, v: str) -> str:
        if v not in SUPPORTED_ASPECT_RATIOS:
            raise ValueError(f"Invalid video aspect ratio '{v}'. Supported: {sorted(SUPPORTED_ASPECT_RATIOS)}")
        return v


class Job(BaseModel):
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Job tracking identifier")
    status: JobStatus = Field(default=JobStatus.PENDING, description="Current job lifecycle status")
    progress: float = Field(default=0.0, ge=0.0, le=100.0, description="Completion progress percentage (0.0 - 100.0)")
    current_step: Optional[str] = Field(default=None, description="Human-readable description of current task")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    error_message: Optional[str] = Field(default=None, description="Failure reason if job failed")
    aspect_ratio: str = Field(default="16:9", description="Target output aspect ratio")
    scenes: List[SceneSegment] = Field(default_factory=list, description="Computed scene breakdown")
    video_metadata: Optional[GeneratedVideoMetadata] = Field(default=None, description="Finished video metadata")


# ==========================================
# Phase 4: B-roll Matching Models
# ==========================================

class BrollMatchRequest(BaseModel):
    scenes: Optional[List[SceneSegment]] = Field(default=None, description="Pre-segmented scenes to match")
    script: Optional[ScriptInput] = Field(default=None, description="Raw script to segment and match")
    target_aspect_ratio: str = Field(default="16:9", description="Target aspect ratio: 16:9, 9:16, or 1:1")
    provider: Optional[str] = Field(default=None, description="Optional provider override: mock or pexels")

    @field_validator("target_aspect_ratio")
    @classmethod
    def validate_target_aspect_ratio(cls, value: str) -> str:
        if value not in SUPPORTED_ASPECT_RATIOS:
            raise ValueError(f"Invalid aspect ratio '{value}'. Supported aspect ratios are: {sorted(SUPPORTED_ASPECT_RATIOS)}")
        return value

    @model_validator(mode="after")
    def validate_input_source(self) -> "BrollMatchRequest":
        if not self.scenes and not self.script:
            raise ValueError("Either 'scenes' or 'script' must be provided for B-roll matching.")
        return self


class SceneMatchResult(BaseModel):
    scene: SceneSegment
    match_status: Literal["matched", "no_candidates", "provider_error"]
    candidates_found: int = Field(default=0, ge=0)
    warning: Optional[str] = None


class BrollMatchResponse(BaseModel):
    target_aspect_ratio: str
    provider_used: str
    total_scenes: int
    matched_scenes: int
    unmatched_scenes: int
    results: List[SceneMatchResult]
    scenes: List[SceneSegment]
    warnings: List[str] = Field(default_factory=list)
