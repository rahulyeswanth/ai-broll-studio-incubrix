"""Per-clip brightness / colour normalisation.

Each clip is sampled (a few frames, downscaled) and two numbers are measured: mean luma (0-1) and mean HSV
saturation (0-1). The job target is the median of all clips measured in the job. A clip is nudged toward the target with
FFmpeg's ``eq`` filter:   brightness = clamp(target_luma - luma, +-MAX_BRIGHTNESS_SHIFT)
                          saturation = clamp(target_sat / sat,  1/MAX_SATURATION_GAIN .. MAX_SATURATION_GAIN)
Small, bounded corrections only: this evens out obviously darker/brighter or washed-out/over-saturated clips, it is
not a colour grade. Measured values and applied corrections are written to timeline.json (``render.normalisation``).
"""
from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .media import ffmpeg_bin, run

MAX_BRIGHTNESS_SHIFT = 0.12
MAX_SATURATION_GAIN = 1.25
DEAD_ZONE_LUMA = 0.02          # differences below this are left alone
DEAD_ZONE_SAT = 0.05
SAMPLE_W, SAMPLE_H = 48, 27


def measure_clip(path: Path, seconds: float = 6.0) -> Optional[Dict[str, float]]:
    """Mean luma and saturation of ~2 frames/second over the first ``seconds`` of the clip; None if unreadable."""
    cmd = [ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-t", f"{seconds:.2f}", "-i", str(path), "-an",
           "-vf", f"fps=2,scale={SAMPLE_W}:{SAMPLE_H}:flags=area", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    try:
        import subprocess
        cp = subprocess.run(cmd, capture_output=True, timeout=60)
    except Exception:  # noqa: BLE001
        return None
    data = cp.stdout
    if cp.returncode != 0 or len(data) < SAMPLE_W * SAMPLE_H * 3:
        return None
    n = len(data) // 3
    lum = sat = 0.0
    r_b = g_b = b_b = 0
    for i in range(0, n * 3, 3):
        r, g, b = data[i], data[i + 1], data[i + 2]
        lum += 0.2126 * r + 0.7152 * g + 0.0722 * b
        mx, mn = max(r, g, b), min(r, g, b)
        sat += (mx - mn) / mx if mx else 0.0
    return {"luma": round(lum / n / 255.0, 4), "saturation": round(sat / n, 4), "frames": n // (SAMPLE_W * SAMPLE_H)}


def target_from(measures: Sequence[Optional[Dict[str, float]]]) -> Optional[Dict[str, float]]:
    ok = [m for m in measures if m]
    if len(ok) < 2:
        return None          # one clip has nothing to be matched against
    return {"luma": round(statistics.median(m["luma"] for m in ok), 4),
            "saturation": round(statistics.median(m["saturation"] for m in ok), 4)}


def adjustment(measure: Optional[Dict[str, float]], target: Optional[Dict[str, float]]) -> Dict[str, Any]:
    rec: Dict[str, Any] = {"measured": measure, "brightness": 0.0, "saturation": 1.0, "applied": False}
    if not measure or not target:
        return rec
    d_l = target["luma"] - measure["luma"]
    b = max(-MAX_BRIGHTNESS_SHIFT, min(MAX_BRIGHTNESS_SHIFT, d_l)) if abs(d_l) >= DEAD_ZONE_LUMA else 0.0
    s = 1.0
    if measure["saturation"] > 0.01 and abs(target["saturation"] - measure["saturation"]) >= DEAD_ZONE_SAT:
        s = max(1 / MAX_SATURATION_GAIN, min(MAX_SATURATION_GAIN, target["saturation"] / measure["saturation"]))
    rec.update({"brightness": round(b, 4), "saturation": round(s, 4), "applied": bool(b or abs(s - 1.0) > 1e-6)})
    return rec


def eq_filter(adj: Dict[str, Any]) -> str:
    return f"eq=brightness={adj['brightness']:.4f}:saturation={adj['saturation']:.4f}," if adj.get("applied") else ""
