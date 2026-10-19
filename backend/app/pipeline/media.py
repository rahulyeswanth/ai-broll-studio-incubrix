"""FFmpeg/FFprobe helpers: probing, audio validation, silence detection, concatenation."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .config import PipelineSettings
from .errors import AudioValidationError, ProviderUnavailableError, RenderError


def ffmpeg_bin() -> str:
    p = shutil.which("ffmpeg")
    if not p:
        raise ProviderUnavailableError("FFmpeg is not installed or not on PATH.")
    return p


def ffprobe_bin() -> str:
    p = shutil.which("ffprobe")
    if not p:
        raise ProviderUnavailableError("FFprobe is not installed or not on PATH.")
    return p


def run(cmd: Sequence[str], timeout: float = 120, cwd: Optional[str] = None) -> subprocess.CompletedProcess:
    return subprocess.run(list(cmd), capture_output=True, text=True, timeout=timeout, check=False, cwd=cwd)


def probe(path: Path) -> Dict[str, Any]:
    """Return ffprobe JSON (format + streams). Raises ValueError if unreadable."""
    cp = run([ffprobe_bin(), "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)], timeout=60)
    if cp.returncode != 0:
        raise ValueError((cp.stderr or "ffprobe failed").strip()[-500:])
    try:
        return json.loads(cp.stdout or "{}")
    except json.JSONDecodeError as exc:  # pragma: no cover
        raise ValueError("ffprobe returned invalid JSON") from exc


def _first_stream(info: Dict[str, Any], kind: str) -> Optional[Dict[str, Any]]:
    for s in info.get("streams", []):
        if s.get("codec_type") == kind:
            return s
    return None


def audio_duration(path: Path) -> float:
    info = probe(path)
    dur = None
    try:
        dur = float(info.get("format", {}).get("duration"))
    except (TypeError, ValueError):
        st = _first_stream(info, "audio")
        try:
            dur = float(st.get("duration")) if st else None
        except (TypeError, ValueError):
            dur = None
    if dur is None or dur <= 0:
        raise ValueError("Could not determine audio duration.")
    return round(dur, 3)


_VOL_RE = re.compile(r"max_volume:\s*(-?[\d.]+|-inf)\s*dB")
_MEAN_RE = re.compile(r"mean_volume:\s*(-?[\d.]+|-inf)\s*dB")


def measure_volume(path: Path) -> Dict[str, Optional[float]]:
    cp = run([ffmpeg_bin(), "-hide_banner", "-nostats", "-i", str(path), "-vn", "-af", "volumedetect", "-f", "null", "-"], timeout=300)
    text = cp.stderr or ""
    def conv(m):
        if not m:
            return None
        return float("-inf") if m.group(1) == "-inf" else float(m.group(1))
    return {"max_db": conv(_VOL_RE.search(text)), "mean_db": conv(_MEAN_RE.search(text)), "returncode": cp.returncode, "stderr": text}  # type: ignore[dict-item]


def decode_check(path: Path) -> Optional[str]:
    """Fully decode the file; return an error string if the decoder reports problems."""
    cp = run([ffmpeg_bin(), "-v", "error", "-nostats", "-i", str(path), "-f", "null", "-"], timeout=600)
    if cp.returncode != 0:
        return (cp.stderr or "decode failed").strip()[-400:]
    if cp.stderr.strip():
        return cp.stderr.strip()[-400:]
    return None


@dataclass
class AudioInfo:
    duration_seconds: float
    codec: Optional[str]
    sample_rate: Optional[int]
    channels: Optional[int]
    format_name: Optional[str]
    size_bytes: int
    max_volume_db: Optional[float]
    mean_volume_db: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "duration_seconds": self.duration_seconds, "codec": self.codec,
            "sample_rate": self.sample_rate, "channels": self.channels,
            "format_name": self.format_name, "size_bytes": self.size_bytes,
            "max_volume_db": None if self.max_volume_db in (None, float("-inf")) else self.max_volume_db,
            "mean_volume_db": None if self.mean_volume_db in (None, float("-inf")) else self.mean_volume_db,
        }


def validate_audio_file(path: Path, settings: PipelineSettings, *, original_name: Optional[str] = None,
                        check_silence: bool = True) -> AudioInfo:
    """Validate an audio file (uploaded or generated). Raises AudioValidationError with a clear message."""
    name = original_name or path.name
    ext = Path(name).suffix.lower()
    if ext not in settings.allowed_audio_extensions:
        raise AudioValidationError(
            f"Unsupported audio format '{ext or '(none)'}'. Allowed: {', '.join(settings.allowed_audio_extensions)}.")
    if not path.is_file():
        raise AudioValidationError("Audio file does not exist.")
    size = path.stat().st_size
    if size == 0:
        raise AudioValidationError("Audio file is empty (0 bytes).")
    if size > settings.max_audio_mb * 1024 * 1024:
        raise AudioValidationError(f"Audio file is {size/1048576:.1f} MB; the limit is {settings.max_audio_mb} MB.")
    try:
        info = probe(path)
    except ValueError as exc:
        raise AudioValidationError(f"Audio file is unreadable or corrupted: {exc}") from exc
    stream = _first_stream(info, "audio")
    if stream is None:
        raise AudioValidationError("File contains no audio stream.")
    try:
        dur = audio_duration(path)
    except ValueError as exc:
        raise AudioValidationError(str(exc)) from exc
    if dur < settings.min_audio_seconds:
        raise AudioValidationError(f"Audio is only {dur:.2f}s long; minimum is {settings.min_audio_seconds}s.")
    if dur > settings.max_audio_seconds:
        raise AudioValidationError(f"Audio is {dur:.0f}s long; the maximum is {settings.max_audio_seconds:.0f}s.")
    err = decode_check(path)
    if err:
        raise AudioValidationError(f"Audio failed a full decode (corrupted?): {err}")
    vol = measure_volume(path)
    max_db = vol["max_db"]
    if check_silence and (max_db is None or max_db <= settings.silence_max_volume_db):
        raise AudioValidationError("Audio appears to be silent (no audible speech detected).")
    def _int(v):
        try:
            return int(v)
        except (TypeError, ValueError):
            return None
    return AudioInfo(dur, stream.get("codec_name"), _int(stream.get("sample_rate")), _int(stream.get("channels")),
                     info.get("format", {}).get("format_name"), size, max_db, vol["mean_db"])


ASR_SAMPLE_RATE = 16000


def prepare_audio_for_asr(src: Path, dest_wav: Path, *, timeout: float = 600.0) -> Dict[str, Any]:
    """Decode ``src`` with FFmpeg to 16 kHz mono 16-bit PCM WAV for speech recognition.

    Timing is untouched (no trimming, no tempo change, no silence removal), so ASR timestamps map 1:1 onto the original
    file. Only a constant gain is applied, and only when the speech is very quiet. Returns what was done.
    """
    vol = measure_volume(src)
    max_db = vol["max_db"]
    gain_db = 0.0
    if max_db is not None and max_db != float("-inf") and max_db < -20.0:
        gain_db = min(-3.0 - max_db, 30.0)       # bring the peak to about -3 dBFS, never more than +30 dB
    cmd = [ffmpeg_bin(), "-hide_banner", "-nostats", "-y", "-i", str(src), "-vn", "-sn", "-dn", "-ac", "1",
           "-ar", str(ASR_SAMPLE_RATE)]
    if gain_db:
        cmd += ["-af", f"volume={gain_db:.1f}dB,alimiter=limit=0.97"]
    cmd += ["-c:a", "pcm_s16le", str(dest_wav)]
    cp = run(cmd, timeout=timeout)
    if cp.returncode != 0 or not dest_wav.is_file() or dest_wav.stat().st_size <= 44:
        dest_wav.unlink(missing_ok=True)
        raise AudioValidationError("The audio could not be decoded for transcription.",
                                   details={"ffmpeg_tail": (cp.stderr or "")[-300:]})
    return {"sample_rate": ASR_SAMPLE_RATE, "channels": 1, "gain_db_applied": round(gain_db, 1),
            "source_max_volume_db": None if max_db in (None, float("-inf")) else max_db}


def load_pcm_float32(wav_path: Path):
    """Read a 16-bit mono WAV written by :func:`prepare_audio_for_asr` as a float32 array in [-1, 1]."""
    import wave

    import numpy as np
    with wave.open(str(wav_path), "rb") as wf:
        if wf.getsampwidth() != 2 or wf.getnchannels() != 1:
            raise AudioValidationError("Unexpected intermediate audio format.")
        frames = wf.readframes(wf.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


def concat_audio(parts: List[Path], dest: Path, *, bitrate: str = "192k") -> Path:
    """Concatenate audio files (re-encoded, so timing is exact and gapless) into ``dest`` (.mp3 or .wav)."""
    if not parts:
        raise RenderError("No audio parts to concatenate.")
    cmd = [ffmpeg_bin(), "-hide_banner", "-y"]
    for p in parts:
        cmd += ["-i", str(p)]
    n = len(parts)
    pre = "".join(f"[{i}:a:0]aresample=44100,aformat=sample_fmts=fltp:channel_layouts=mono[a{i}];" for i in range(n))
    cmd += ["-filter_complex", pre + "".join(f"[a{i}]" for i in range(n)) + f"concat=n={n}:v=0:a=1[out]", "-map", "[out]"]
    if dest.suffix.lower() == ".mp3":
        cmd += ["-c:a", "libmp3lame", "-b:a", bitrate]
    elif dest.suffix.lower() == ".wav":
        cmd += ["-c:a", "pcm_s16le"]
    else:
        cmd += ["-c:a", "aac", "-b:a", bitrate]
    cmd.append(str(dest))
    cp = run(cmd, timeout=600)
    if cp.returncode != 0 or not dest.is_file() or dest.stat().st_size == 0:
        dest.unlink(missing_ok=True)
        raise RenderError(f"Audio concatenation failed: {(cp.stderr or '')[-400:]}")
    return dest


def make_tone_audio(dest: Path, seconds: float, freq: int = 440, volume: float = 0.3) -> Path:
    """Generate a sine tone file. Used ONLY by placeholder/test providers, never presented as narration."""
    cmd = [ffmpeg_bin(), "-hide_banner", "-y", "-f", "lavfi", "-i", f"sine=frequency={freq}:duration={seconds:.3f}",
           "-af", f"volume={volume}"]
    if dest.suffix.lower() == ".mp3":
        cmd += ["-c:a", "libmp3lame", "-b:a", "96k"]
    cmd.append(str(dest))
    cp = run(cmd, timeout=120)
    if cp.returncode != 0:
        raise RenderError(f"Tone generation failed: {(cp.stderr or '')[-300:]}")
    return dest


def video_info(path: Path) -> Dict[str, Any]:
    info = probe(path)
    v = _first_stream(info, "video")
    if v is None:
        raise ValueError("No video stream.")
    try:
        dur = float(info.get("format", {}).get("duration") or v.get("duration") or 0)
    except (TypeError, ValueError):
        dur = 0.0
    return {"width": int(v.get("width") or 0), "height": int(v.get("height") or 0), "duration": dur,
            "codec": v.get("codec_name"), "pix_fmt": v.get("pix_fmt"), "r_frame_rate": v.get("r_frame_rate")}
