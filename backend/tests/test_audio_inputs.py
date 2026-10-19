"""Audio input handling against REAL speech fixtures and the real faster-whisper engine.

Covers the IncuBrix audio matrix: MP3/WAV/M4A speech, silent, music-only, quiet, corrupted, wrong extension, empty,
over-length, ten-minute, transcription timeout/provider failure. Whisper-dependent tests skip (never silently pass) if the
model cannot be loaded; the skip reason is printed.
"""
from __future__ import annotations

import asyncio
import re
import subprocess
from pathlib import Path

import pytest

from app.pipeline.config import PipelineSettings
from app.pipeline.errors import AudioValidationError, TranscriptionError, TranscriptionTimeoutError
from app.pipeline.media import audio_duration, prepare_audio_for_asr, validate_audio_file
from app.pipeline.transcription import FasterWhisperTranscriber

FIX = Path(__file__).parent / "fixtures" / "audio"
BACKEND = Path(__file__).resolve().parents[1]
SPEECH_TEXT = ("Ocean waves crash on the shore. Scientists study coral reefs to protect marine life. "
               "Plastic waste is a growing threat to the sea.")
S = PipelineSettings()


@pytest.fixture(scope="module")
def whisper():
    t = FasterWhisperTranscriber(S.whisper_model_size, S.whisper_device, S.whisper_compute_type, timeout_seconds=300)
    try:
        t._load()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Whisper model unavailable in this environment: {type(exc).__name__}: {exc}")
    return t


def _transcribe(t, path):
    return asyncio.run(t.transcribe(path, None))


# ---------------------------------------------------------------- validation (FFprobe/FFmpeg only)
@pytest.mark.parametrize("name", ["speech_clear.mp3", "speech_clear.wav", "speech_clear.m4a", "quiet_speech.wav", "music_only.mp3"])
def test_supported_files_pass_validation(name):
    info = validate_audio_file(FIX / name, S)
    assert 4 < info.duration_seconds < 12


@pytest.mark.parametrize("name,fragment", [
    ("silent.wav", "silent"),
    ("empty.mp3", "empty"),
    ("not_audio.mp3", "unreadable"),
    ("corrupt.mp3", "corrupt"),
])
def test_bad_files_rejected_with_clear_message(name, fragment):
    with pytest.raises(AudioValidationError) as ei:
        validate_audio_file(FIX / name, S)
    assert fragment in ei.value.message.lower()
    assert ei.value.code == "invalid_audio"


def test_unsupported_extension_rejected(tmp_path):
    p = tmp_path / "talk.txt"
    p.write_bytes((FIX / "speech_clear.wav").read_bytes())
    with pytest.raises(AudioValidationError, match="Unsupported audio format"):
        validate_audio_file(p, S)


def test_wrong_extension_with_valid_audio_content_is_decoded_by_content(tmp_path):
    """A real MP3 named .wav is still legitimate audio; ffprobe identifies it by content, not by name."""
    p = tmp_path / "actually_mp3.wav"
    p.write_bytes((FIX / "speech_clear.mp3").read_bytes())
    assert validate_audio_file(p, S).duration_seconds > 4


def _tone(path: Path, seconds: float):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", f"sine=frequency=300:duration={seconds}",
                    "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(path)], check=True)
    return path


def test_ten_minute_audio_is_accepted_and_longer_is_rejected(tmp_path):
    ten = _tone(tmp_path / "ten.wav", 600)
    assert validate_audio_file(ten, S).duration_seconds == pytest.approx(600, abs=0.05)
    too_long = _tone(tmp_path / "long.wav", 601)
    with pytest.raises(AudioValidationError, match="maximum is 600s"):
        validate_audio_file(too_long, S)


def test_normalisation_preserves_timing_and_boosts_only_quiet_audio(tmp_path):
    out = tmp_path / "n.wav"
    info = prepare_audio_for_asr(FIX / "quiet_speech.wav", out)
    assert info["gain_db_applied"] > 10
    assert audio_duration(out) == pytest.approx(audio_duration(FIX / "quiet_speech.wav"), abs=0.02)
    loud = prepare_audio_for_asr(FIX / "speech_clear.wav", tmp_path / "l.wav")
    assert loud["gain_db_applied"] == 0.0


# ---------------------------------------------------------------- transcription (real engine)
@pytest.mark.parametrize("name", ["speech_clear.mp3", "speech_clear.wav", "speech_clear.m4a", "quiet_speech.wav"])
def test_real_transcription_all_formats(whisper, name):
    r = _transcribe(whisper, FIX / name)
    text = re.sub(r"[^a-z ]", "", r.text.lower())
    for expected in ("ocean waves", "coral reefs", "plastic waste"):
        assert expected in text, (name, r.text)
    dur = audio_duration(FIX / name)
    starts = [w.start for w in r.words]
    assert starts == sorted(starts)
    assert all(0 <= w.start <= w.end <= dur + 0.5 for w in r.words)
    assert r.granularity == "word"


def test_timestamps_align_with_the_audio(whisper):
    """'Plastic' is the first word of the last sentence; it must be timed in the final third of the recording."""
    r = _transcribe(whisper, FIX / "speech_clear.mp3")
    dur = audio_duration(FIX / "speech_clear.mp3")
    plastic = next(w for w in r.words if w.word.lower().startswith("plastic"))
    assert dur * 0.55 < plastic.start < dur * 0.85
    assert r.words[0].start < 0.6


def test_music_only_yields_no_words(whisper):
    assert _transcribe(whisper, FIX / "music_only.mp3").words == []


def test_regression_pyav_metadata_errors_not_used(whisper, monkeypatch):
    """faster-whisper's own decoder calls av.open(metadata_errors=...), which PyAV>=19 rejects. We must never reach it."""
    import faster_whisper.audio as fwa

    def boom(*a, **k):
        raise AssertionError("faster_whisper.audio.decode_audio must not be used")
    monkeypatch.setattr(fwa, "decode_audio", boom)
    import faster_whisper.transcribe as fwt
    monkeypatch.setattr(fwt, "decode_audio", boom, raising=False)
    assert "coral" in _transcribe(whisper, FIX / "speech_clear.mp3").text.lower()


def test_requirements_pin_pyav_below_19():
    pin = [ln for ln in (BACKEND / "requirements.txt").read_text().splitlines() if re.match(r"\s*av\b", ln)]
    assert pin and "<19" in pin[0].replace(" ", "")


# ---------------------------------------------------------------- failure modes
def test_provider_failure_gives_safe_message_without_internals():
    t = FasterWhisperTranscriber(timeout_seconds=30)

    class Broken:
        def transcribe(self, *a, **k):
            raise TypeError("open() got an unexpected keyword argument 'metadata_errors' at /srv/secret/path")
    t._model = Broken()
    with pytest.raises(TranscriptionError) as ei:
        _transcribe(t, FIX / "speech_clear.wav")
    msg = ei.value.message + str(ei.value.details)
    assert "metadata_errors" not in msg and "/srv/secret" not in msg
    assert ei.value.code == "transcription_failed" and ei.value.retryable


def test_transcription_timeout_is_reported():
    import time
    t = FasterWhisperTranscriber(timeout_seconds=0.3)

    class Slow:
        def transcribe(self, *a, **k):
            time.sleep(2)
    t._model = Slow()
    with pytest.raises(TranscriptionTimeoutError) as ei:
        _transcribe(t, FIX / "speech_clear.wav")
    assert ei.value.code == "transcription_timeout"
