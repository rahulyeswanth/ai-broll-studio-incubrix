from typing import List, Literal
from pydantic import BaseModel, Field, field_validator, model_validator

class CaptionSceneInput(BaseModel):
    scene_id: str = Field(min_length=1)
    start_time: float = Field(ge=0)
    end_time: float = Field(gt=0)
    narration_text: str = Field(min_length=1, max_length=10000)

    @model_validator(mode="after")
    def validate_timing(self):
        if self.end_time <= self.start_time:
            raise ValueError("Scene end_time must be greater than start_time.")
        if not self.narration_text.strip():
            raise ValueError("Narration text cannot be blank.")
        return self

class CaptionRequest(BaseModel):
    scenes: List[CaptionSceneInput] = Field(min_length=1, max_length=500)
    format: Literal["srt", "vtt"] = "srt"
    max_words_per_caption: int = Field(default=8, ge=1, le=20)

class CaptionSegmentOutput(BaseModel):
    index: int = Field(ge=1)
    start_time: float = Field(ge=0)
    end_time: float = Field(gt=0)
    text: str

class CaptionResponse(BaseModel):
    format: Literal["srt", "vtt"]
    total_captions: int
    total_duration_seconds: float
    captions: List[CaptionSegmentOutput]
    subtitle_text: str
    warning: str = "Caption timings are estimated from scene durations; verify against real audio before publishing."
