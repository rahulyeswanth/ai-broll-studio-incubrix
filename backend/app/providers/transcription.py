import logging
from app.providers.base import TranscriptionProvider, TranscriptResult, TranscriptSegment
from app.providers.mocks import MockTranscriptionProvider

logger = logging.getLogger(__name__)


def get_transcription_provider(provider_name: str = "mock") -> TranscriptionProvider:
    """
    Transcription provider factory.
    Mock provider is active by default to enable immediate offline testing without GPU or paid keys.
    """
    if provider_name.lower() == "mock":
        return MockTranscriptionProvider()
    elif provider_name.lower() == "whisper":
        return FasterWhisperTranscriptionProvider()
    raise NotImplementedError(
        f"Transcription provider '{provider_name}' is not configured. Supported: 'mock', 'whisper'."
    )


class FasterWhisperTranscriptionProvider(TranscriptionProvider):
    """
    Optional local transcription using faster-whisper.
    Requires: pip install faster-whisper
    No API keys or cloud subscriptions needed; runs entirely on local CPU/CUDA.
    """

    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
                logger.info(f"Loading faster-whisper model: {self.model_size}")
                self._model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
            except ImportError as exc:
                raise ImportError(
                    "faster-whisper is not installed. To use real transcription, install it with:\n"
                    "pip install faster-whisper\n"
                    "Alternatively, keep TRANSCRIPTION_PROVIDER=mock for offline development."
                ) from exc

    async def transcribe(self, audio_file_path: str) -> TranscriptResult:
        import asyncio

        self._load_model()

        def _sync_transcribe():
            segments_raw, info = self._model.transcribe(audio_file_path, beam_size=5, word_timestamps=True)
            segments = []
            full_texts = []
            for seg in segments_raw:
                full_texts.append(seg.text.strip())
                segments.append(
                    TranscriptSegment(
                        start_time=round(seg.start, 2),
                        end_time=round(seg.end, 2),
                        text=seg.text.strip(),
                    )
                )
            return TranscriptResult(
                full_text=" ".join(full_texts),
                segments=segments,
            )

        return await asyncio.to_thread(_sync_transcribe)
