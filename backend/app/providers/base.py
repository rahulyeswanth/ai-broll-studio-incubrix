from abc import ABC, abstractmethod
from typing import List
from pydantic import BaseModel, Field


class VoiceoverResult(BaseModel):
    audio_path_or_url: str
    duration_seconds: float


class TranscriptSegment(BaseModel):
    start_time: float
    end_time: float
    text: str


class TranscriptResult(BaseModel):
    full_text: str
    segments: List[TranscriptSegment]


class BrollAsset(BaseModel):
    id: str
    source: str = "mock"  # e.g., 'pexels', 'mock', 'pixabay'
    title: str
    download_url: str
    preview_url: str | None = None
    aspect_ratio: str = "16:9"
    duration: float
    width: int | None = None
    height: int | None = None
    commercial_use_allowed: bool = True
    license_type: str = "CC0-Commercial-Safe"
    attribution_required: bool = False
    author_name: str | None = None
    author_url: str | None = None
    # Optional provenance metadata (added for licence/attribution reporting). All default to "unknown".
    page_url: str | None = None
    tags: List[str] = Field(default_factory=list)
    ai_generated: bool | None = None


class TTSProvider(ABC):
    @abstractmethod
    async def synthesize(self, text: str, voice_id: str | None = None) -> VoiceoverResult:
        pass


class TranscriptionProvider(ABC):
    @abstractmethod
    async def transcribe(self, audio_file_path: str) -> TranscriptResult:
        pass


class FootageProvider(ABC):
    @abstractmethod
    async def search_footage(
        self, query: str, aspect_ratio: str = "16:9", max_results: int = 5
    ) -> List[BrollAsset]:
        pass
