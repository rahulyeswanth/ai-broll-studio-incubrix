"""Speech-to-text adapters for uploaded audio (Mode B). Word timestamps are used when the engine provides them."""
from __future__ import annotations

import asyncio
import importlib.metadata
import logging
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .alignment import distribute_words, tokenize
from .config import PipelineSettings
from .errors import AudioValidationError, PipelineError, ProviderUnavailableError, TranscriptionError, TranscriptionTimeoutError
from .media import load_pcm_float32, prepare_audio_for_asr
from .models import WordTiming

log = logging.getLogger(__name__)


@dataclass
class ASRResult:
    text: str
    words: List[WordTiming]
    granularity: str                 # "word" | "segment_interpolated"
    language: Optional[str] = None
    language_probability: Optional[float] = None
    segments: List[tuple] = field(default_factory=list)


class PipelineTranscriber(ABC):
    name = "unknown"
    is_real = False
    model: Optional[str] = None
    version: Optional[str] = None

    @abstractmethod
    async def transcribe(self, audio_path: Path, language: Optional[str]) -> ASRResult:
        ...


class FasterWhisperTranscriber(PipelineTranscriber):
    name = "faster-whisper"
    is_real = True

    def __init__(self, model_size: str = "base", device: str = "cpu", compute_type: str = "int8",
                 timeout_seconds: float = 900.0):
        self.model_size, self.device, self.compute_type = model_size, device, compute_type
        self.timeout_seconds = timeout_seconds
        self.last_prepare_info: dict = {}
        self.model = f"whisper-{model_size}"
        self._model = None
        try:
            self.version = importlib.metadata.version("faster-whisper")
        except importlib.metadata.PackageNotFoundError:
            self.version = None

    def _load(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel  # type: ignore
            except ImportError as exc:
                raise ProviderUnavailableError(
                    "faster-whisper is not installed. Run: pip install faster-whisper "
                    "(first use downloads the Whisper model, which needs internet access).") from exc
            self._model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
        return self._model

    async def transcribe(self, audio_path: Path, language: Optional[str]) -> ASRResult:
        # Decode with FFmpeg ourselves and give Whisper the PCM array. faster-whisper's own file decoder calls
        # av.open(..., metadata_errors=...), which PyAV >= 19 rejects (TypeError); passing an array avoids PyAV decoding.
        def _work() -> ASRResult:
            model = self._load()
            with tempfile.TemporaryDirectory(prefix="asr_") as tmp:
                wav = Path(tmp) / "asr_input.wav"
                self.last_prepare_info = prepare_audio_for_asr(audio_path, wav)
                pcm = load_pcm_float32(wav)
            segs, info = model.transcribe(pcm, word_timestamps=True, vad_filter=True, language=language or None)
            words: List[WordTiming] = []
            segments: List[tuple] = []
            interpolated = False
            for seg in segs:
                text = (seg.text or "").strip()
                segments.append((float(seg.start), float(seg.end), text))
                seg_words = getattr(seg, "words", None) or []
                if seg_words:
                    for w in seg_words:
                        t = (w.word or "").strip()
                        if t:
                            words.append(WordTiming(t, float(w.start), float(w.end)))
                elif text:
                    interpolated = True
                    words.extend(distribute_words(tokenize(text), float(seg.start), float(seg.end)))
            return ASRResult(" ".join(w.word for w in words), words, "segment_interpolated" if interpolated else "word",
                             getattr(info, "language", None), getattr(info, "language_probability", None), segments)
        try:
            return await asyncio.wait_for(asyncio.to_thread(_work), timeout=self.timeout_seconds)
        except asyncio.TimeoutError as exc:
            raise TranscriptionTimeoutError(
                f"Transcription did not finish within {self.timeout_seconds:.0f}s. Try a shorter file or retry later.",
                details={"timeout_seconds": self.timeout_seconds}) from exc
        except (ProviderUnavailableError, AudioValidationError, PipelineError):
            raise
        except Exception as exc:  # noqa: BLE001
            # Full detail goes to the server log only; the public message never carries exception text or paths.
            log.exception("transcription failed (%s)", type(exc).__name__)
            raise TranscriptionError("The audio could not be transcribed. Please retry; if it keeps failing, quote the "
                                     "request ID to the operator.",
                                     details={"cause": type(exc).__name__}) from exc


class MockTranscriber(PipelineTranscriber):
    """Returns canned text unrelated to the audio. Never real."""
    name = "mock"
    is_real = False
    model = "canned-text"

    async def transcribe(self, audio_path: Path, language: Optional[str]) -> ASRResult:
        from .media import audio_duration
        dur = audio_duration(audio_path)
        toks = tokenize("Welcome to AI B-roll Studio. This transcript is a placeholder and does not reflect the audio.")
        return ASRResult(" ".join(toks), distribute_words(toks, 0.0, dur), "segment_interpolated", language, None, [])


def get_pipeline_transcriber(name: str, settings: PipelineSettings) -> PipelineTranscriber:
    name = (name or "").lower()
    if name in ("whisper", "faster-whisper", "faster_whisper"):
        return FasterWhisperTranscriber(settings.whisper_model_size, settings.whisper_device, settings.whisper_compute_type,
                                        settings.transcription_timeout_seconds)
    if name == "mock":
        if not settings.allow_mock_providers:
            raise ProviderUnavailableError(
                "TRANSCRIPTION_PROVIDER=mock returns canned text that does not reflect your audio. Use "
                "TRANSCRIPTION_PROVIDER=whisper (pip install faster-whisper), or set ALLOW_MOCK_PROVIDERS=true.")
        return MockTranscriber()
    raise ProviderUnavailableError(f"Unsupported transcription provider '{name}'. Supported: whisper, mock.")
