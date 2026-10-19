from typing import List
from app.providers.base import (
    TTSProvider,
    TranscriptionProvider,
    FootageProvider,
    VoiceoverResult,
    TranscriptResult,
    TranscriptSegment,
    BrollAsset,
)


class MockTTSProvider(TTSProvider):
    async def synthesize(self, text: str, voice_id: str | None = None) -> VoiceoverResult:
        word_count = len(text.split())
        est_duration = max(1.5, word_count * 0.4)
        return VoiceoverResult(
            audio_path_or_url="mock://storage/audio/mock_narration.mp3",
            duration_seconds=round(est_duration, 2),
        )


class MockTranscriptionProvider(TranscriptionProvider):
    async def transcribe(self, audio_file_path: str) -> TranscriptResult:
        return TranscriptResult(
            full_text="Welcome to AI B-roll Studio automated generation.",
            segments=[
                TranscriptSegment(start_time=0.0, end_time=2.5, text="Welcome to AI B-roll Studio"),
                TranscriptSegment(start_time=2.5, end_time=5.0, text="automated generation."),
            ],
        )


class MockFootageProvider(FootageProvider):
    async def search_footage(
        self, query: str, aspect_ratio: str = "16:9", max_results: int = 5
    ) -> List[BrollAsset]:
        dims = {"16:9": (1920, 1080), "9:16": (1080, 1920), "1:1": (1080, 1080)}.get(
            aspect_ratio, (1920, 1080)
        )
        return [
            BrollAsset(
                id=f"mock_asset_{i}",
                source="mock",
                title=f"Stock clip for query: {query}",
                download_url=f"mock://cdn.assets.local/clips/{query}_{i}.mp4",
                preview_url=f"mock://cdn.assets.local/previews/{query}_{i}.jpg",
                aspect_ratio=aspect_ratio,
                duration=6.0,
                width=dims[0],
                height=dims[1],
                commercial_use_allowed=True,
                license_type="CC0-Commercial-Safe",
                attribution_required=False,
                author_name="Mock Creator",
                author_url="https://mock.local/creator",
            )
            for i in range(1, max_results + 1)
        ]
