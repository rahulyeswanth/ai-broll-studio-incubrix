"""Narration generation for script-only input.

``EdgeTTSProvider`` is the real provider (Microsoft Edge online voices through the ``edge-tts`` package; needs
internet access, no API key). ``PlaceholderToneTTS`` (name ``mock``) produces tone audio of plausible length for
offline development and is *always* flagged ``is_real=False``; the pipeline refuses it unless
``ALLOW_MOCK_PROVIDERS=true`` and then labels the output as placeholder, never as narration.
"""
from __future__ import annotations

import asyncio
import importlib.metadata
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .alignment import distribute_words, map_script_to_timed_words, phrases_to_words, tokenize
from .config import PipelineSettings
from .errors import InputValidationError, ProviderUnavailableError, TTSError
from .media import audio_duration, concat_audio, make_tone_audio, validate_audio_file
from .models import Scene, WordTiming

log = logging.getLogger(__name__)

VOICE_PRESETS: List[Dict[str, str]] = [
    {"id": "en-US-AriaNeural", "label": "English (US) - Aria", "language": "en"},
    {"id": "en-US-GuyNeural", "label": "English (US) - Guy", "language": "en"},
    {"id": "en-GB-SoniaNeural", "label": "English (UK) - Sonia", "language": "en"},
    {"id": "en-IN-NeerjaNeural", "label": "English (India) - Neerja", "language": "en"},
    {"id": "hi-IN-SwaraNeural", "label": "Hindi - Swara", "language": "hi"},
    {"id": "ta-IN-PallaviNeural", "label": "Tamil - Pallavi", "language": "ta"},
    {"id": "es-ES-ElviraNeural", "label": "Spanish (Spain) - Elvira", "language": "es"},
    {"id": "fr-FR-DeniseNeural", "label": "French - Denise", "language": "fr"},
    {"id": "de-DE-KatjaNeural", "label": "German - Katja", "language": "de"},
]


@dataclass
class TTSResult:
    path: Path
    boundaries: List[Tuple[float, float, str, str]] = field(default_factory=list)  # (start,end,text,kind word|phrase)


class PipelineTTS(ABC):
    name = "unknown"
    is_real = False
    model: Optional[str] = None
    version: Optional[str] = None

    @abstractmethod
    async def synthesize(self, text: str, voice: str, language: str, dest: Path) -> TTSResult:
        ...


class EdgeTTSProvider(PipelineTTS):
    name = "edge-tts"
    is_real = True
    model = "Microsoft Edge online neural voices (service-side; no model version exposed)"

    def __init__(self, timeout: float = 60.0):
        self.timeout = timeout
        try:
            self.version = importlib.metadata.version("edge-tts")
        except importlib.metadata.PackageNotFoundError:
            self.version = None

    async def synthesize(self, text: str, voice: str, language: str, dest: Path) -> TTSResult:
        try:
            import edge_tts  # type: ignore
        except ImportError as exc:
            raise ProviderUnavailableError("The 'edge-tts' package is not installed. Run: pip install edge-tts") from exc

        async def _run() -> TTSResult:
            try:
                comm = edge_tts.Communicate(text, voice, boundary="WordBoundary")
            except TypeError:  # older edge-tts without the boundary argument
                comm = edge_tts.Communicate(text, voice)
            bounds: List[Tuple[float, float, str, str]] = []
            got_audio = False
            with dest.open("wb") as fh:
                async for chunk in comm.stream():
                    kind = chunk.get("type")
                    if kind == "audio":
                        data = chunk.get("data") or b""
                        if data:
                            got_audio = True
                            fh.write(data)
                    elif kind in ("WordBoundary", "SentenceBoundary"):
                        off = float(chunk.get("offset", 0)) / 1e7
                        dur = float(chunk.get("duration", 0)) / 1e7
                        bounds.append((off, off + max(dur, 0.01), str(chunk.get("text", "")),
                                       "word" if kind == "WordBoundary" else "phrase"))
            if not got_audio:
                raise TTSError("The TTS service returned no audio (check the voice name and your network).")
            return TTSResult(dest, bounds)

        try:
            return await asyncio.wait_for(_run(), timeout=self.timeout)
        except asyncio.TimeoutError as exc:
            dest.unlink(missing_ok=True)
            raise TTSError(f"TTS request timed out after {self.timeout:.0f}s.") from exc
        except (TTSError, ProviderUnavailableError):
            dest.unlink(missing_ok=True)
            raise
        except Exception as exc:  # network / service errors from edge-tts (NoAudioReceived, aiohttp, ...)
            dest.unlink(missing_ok=True)
            raise TTSError(f"TTS provider failed: {type(exc).__name__}: {exc}") from exc


class PlaceholderToneTTS(PipelineTTS):
    """NOT narration. Emits a tone whose length mimics speech so the offline pipeline can be exercised."""
    name = "mock"
    is_real = False
    model = "placeholder-tone"

    async def synthesize(self, text: str, voice: str, language: str, dest: Path) -> TTSResult:
        words = max(1, len(tokenize(text)))
        make_tone_audio(dest, max(1.5, words * 0.38))
        return TTSResult(dest, [])


def get_pipeline_tts(name: str, settings: PipelineSettings) -> PipelineTTS:
    name = (name or "").lower()
    if name in ("edge", "edge-tts", "edge_tts"):
        return EdgeTTSProvider(timeout=settings.tts_timeout_seconds)
    if name == "mock":
        if not settings.allow_mock_providers:
            raise ProviderUnavailableError(
                "TTS_PROVIDER=mock only produces placeholder tones, not narration. Use TTS_PROVIDER=edge, or set "
                "ALLOW_MOCK_PROVIDERS=true to explicitly accept placeholder audio for development.")
        return PlaceholderToneTTS()
    raise ProviderUnavailableError(f"Unsupported TTS provider '{name}'. Supported: edge, mock.")


@dataclass
class NarrationResult:
    audio_path: Path
    duration_seconds: float
    scenes: List[Scene]
    timing_method: str
    provider: str
    is_real: bool
    warnings: List[str] = field(default_factory=list)


def _scene_words(scene_text: str, res: TTSResult, scene_dur: float) -> Tuple[List[WordTiming], str]:
    tokens = tokenize(scene_text)
    word_b = [WordTiming(t, s, e) for s, e, t, k in res.boundaries if k == "word"]
    phr_b = [(s, e, t) for s, e, t, k in res.boundaries if k == "phrase"]
    if word_b:
        timed, ratio = map_script_to_timed_words(tokens, word_b, 0.0, scene_dur)
        for w in timed:
            w.end = min(w.end, scene_dur)
            w.start = min(w.start, max(0.0, w.end - 0.01))
        return timed, ("tts_word_boundaries" if ratio >= 0.9 else "tts_word_boundaries_partial")
    if phr_b:
        timed, _ = map_script_to_timed_words(tokens, phrases_to_words(phr_b), 0.0, scene_dur)
        return timed, "tts_phrase_boundaries_interpolated"
    return distribute_words(tokens, 0.0, scene_dur), "scene_exact_words_estimated"


async def narrate_scenes(scene_texts: Sequence[str], tts: PipelineTTS, voice: str, language: str, workdir: Path,
                         settings: PipelineSettings, dest_name: str = "narration.mp3",
                         progress=None) -> NarrationResult:
    """Synthesise every scene separately (so scene boundaries are exact), then join the audio gaplessly."""
    if not scene_texts:
        raise InputValidationError("The script contains no narratable text.")
    workdir.mkdir(parents=True, exist_ok=True)
    parts: List[Path] = []
    results: List[TTSResult] = []
    durations: List[float] = []
    for i, text in enumerate(scene_texts, 1):
        dest = workdir / f"scene_{i:03d}.mp3"
        res = await tts.synthesize(text, voice, language, dest)
        if not dest.is_file() or dest.stat().st_size == 0:
            raise TTSError(f"TTS produced no audio file for scene {i}.")
        try:
            dur = audio_duration(dest)
        except ValueError as exc:
            raise TTSError(f"TTS audio for scene {i} is not playable: {exc}") from exc
        parts.append(dest)
        results.append(res)
        durations.append(dur)
        if progress:
            progress(i, len(scene_texts))
    final = concat_audio(parts, workdir / dest_name)
    info = validate_audio_file(final, settings, check_silence=tts.is_real)
    total_est = sum(durations)
    warnings: List[str] = []
    if abs(info.duration_seconds - total_est) > 0.15:
        warnings.append(f"Joined narration is {info.duration_seconds:.2f}s but scene audio sums to {total_est:.2f}s; "
                        "last scene adjusted to the measured length.")
    scenes: List[Scene] = []
    cur = 0.0
    methods = set()
    for i, (text, res, dur) in enumerate(zip(scene_texts, results, durations), 1):
        start = round(cur, 3)
        end = round(cur + dur, 3) if i < len(scene_texts) else round(info.duration_seconds, 3)
        if end <= start:
            end = round(start + 0.05, 3)
        words, method = _scene_words(text, res, end - start)
        methods.add(method)
        words = [WordTiming(w.word, round(start + w.start, 3), round(min(start + w.end, end), 3)) for w in words]
        scenes.append(Scene(f"scene-{i:03d}", i, text, start, end, words, method))
        cur = end
    overall = methods.pop() if len(methods) == 1 else "mixed:" + "+".join(sorted(methods))
    if not tts.is_real:
        warnings.append("PLACEHOLDER AUDIO: the mock TTS provider generated tones, not narration.")
    return NarrationResult(final, info.duration_seconds, scenes, overall, tts.name, tts.is_real, warnings)
