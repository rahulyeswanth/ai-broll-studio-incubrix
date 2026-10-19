"""Independent validation of the finished MP4 (FFprobe + FFmpeg detectors). A zero FFmpeg exit code is
not treated as proof of a good video."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .media import ffmpeg_bin, probe, run

_BLACK = re.compile(r"black_start:([\d.]+)\s+black_end:([\d.]+)\s+black_duration:([\d.]+)")
_FREEZE_DUR = re.compile(r"lavfi\.freezedetect\.freeze_duration:\s*([\d.]+)")
_FREEZE_START = re.compile(r"lavfi\.freezedetect\.freeze_start:\s*([\d.]+)")
_FREEZE_END = re.compile(r"lavfi\.freezedetect\.freeze_end:\s*([\d.]+)")


def validate_output(path: Path, *, expected_w: int, expected_h: int, expected_duration: float,
                    expect_audio: bool, fps: int = 30, duration_tolerance: float = 0.25,
                    run_detectors: bool = True,
                    ignore_ranges: Sequence[Tuple[float, float]] = ()) -> Dict[str, Any]:
    checks: Dict[str, Any] = {}
    errors: List[str] = []
    warnings: List[str] = []
    if not path.is_file() or path.stat().st_size == 0:
        return {"passed": False, "errors": ["output file is missing or empty"], "warnings": [], "checks": {}}
    try:
        info = probe(path)
    except ValueError as exc:
        return {"passed": False, "errors": [f"output is unreadable: {exc}"], "warnings": [], "checks": {}}
    v = next((s for s in info["streams"] if s.get("codec_type") == "video"), None)
    a = next((s for s in info["streams"] if s.get("codec_type") == "audio"), None)
    fmt_dur = float(info.get("format", {}).get("duration") or 0)
    checks["file_size_bytes"] = path.stat().st_size
    checks["duration_seconds"] = round(fmt_dur, 3)
    if v is None:
        errors.append("no video stream")
    else:
        checks.update({"video_codec": v.get("codec_name"), "pix_fmt": v.get("pix_fmt"),
                       "width": v.get("width"), "height": v.get("height"), "r_frame_rate": v.get("r_frame_rate"),
                       "profile": v.get("profile")})
        if v.get("codec_name") != "h264":
            errors.append(f"video codec is {v.get('codec_name')}, expected h264")
        if v.get("pix_fmt") != "yuv420p":
            errors.append(f"pixel format is {v.get('pix_fmt')}, expected yuv420p")
        if (v.get("width"), v.get("height")) != (expected_w, expected_h):
            errors.append(f"resolution is {v.get('width')}x{v.get('height')}, expected {expected_w}x{expected_h}")
        try:
            num, den = (v.get("r_frame_rate") or "0/1").split("/")
            rate = float(num) / float(den)
            if abs(rate - fps) > 0.01:
                errors.append(f"frame rate is {rate:.3f}, expected {fps}")
        except (ValueError, ZeroDivisionError):
            errors.append("could not read frame rate")
    if expect_audio:
        if a is None:
            errors.append("no audio stream")
        else:
            checks.update({"audio_codec": a.get("codec_name"), "audio_sample_rate": a.get("sample_rate"),
                           "audio_channels": a.get("channels")})
            if a.get("codec_name") != "aac":
                errors.append(f"audio codec is {a.get('codec_name')}, expected aac")
            try:
                adur = float(a.get("duration") or fmt_dur)
                checks["audio_duration_seconds"] = round(adur, 3)
                if abs(adur - expected_duration) > duration_tolerance:
                    errors.append(f"audio duration {adur:.2f}s differs from narration {expected_duration:.2f}s (drift/truncation)")
            except (TypeError, ValueError):
                warnings.append("could not read audio stream duration")
    if abs(fmt_dur - expected_duration) > duration_tolerance:
        errors.append(f"video duration {fmt_dur:.2f}s differs from expected {expected_duration:.2f}s")
    if run_detectors and v is not None:
        cp = run([ffmpeg_bin(), "-hide_banner", "-nostats", "-i", str(path), "-an",
                  "-vf", "blackdetect=d=0.5:pix_th=0.10,freezedetect=n=-60dB:d=2", "-f", "null", "-"], timeout=900)
        text = cp.stderr or ""
        blacks = [(float(m.group(1)), float(m.group(3))) for m in _BLACK.finditer(text)]
        starts, ends, durs = _FREEZE_START.findall(text), _FREEZE_END.findall(text), _FREEZE_DUR.findall(text)
        freeze_iv = []   # (start, duration); a freeze that runs to the end of the video never prints a duration
        for i, s0 in enumerate(starts):
            d = float(durs[i]) if i < len(durs) else round(max(0.0, fmt_dur - float(s0)), 3)
            freeze_iv.append((float(s0), d))

        def _ignored(start: float, dur: float) -> bool:
            for a, b in ignore_ranges:   # intentional placeholder (unmatched scene): static/dark by design
                if start >= a - 0.25 and start + dur <= b + 0.25:
                    return True
            return False
        blacks = [(s0, d) for s0, d in blacks if not _ignored(s0, d)]
        freeze_iv = [(s0, d) for s0, d in freeze_iv if not _ignored(s0, d)]
        freezes = [d for _, d in freeze_iv]
        checks["black_intervals"] = [{"start": s0, "duration": d} for s0, d in blacks]
        checks["freeze_intervals_seconds"] = freezes
        if ignore_ranges:
            checks["placeholder_ranges_ignored"] = [[round(a, 2), round(b, 2)] for a, b in ignore_ranges]
        for s0, d in blacks:
            warnings.append(f"black frames detected at {s0:.1f}s for {d:.1f}s")
        for d in freezes:
            warnings.append(f"frozen video detected for {d:.1f}s")
    return {"passed": not errors, "errors": errors, "warnings": warnings, "checks": checks}
