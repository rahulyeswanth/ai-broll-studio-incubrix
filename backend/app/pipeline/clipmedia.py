"""Small FFmpeg/Pillow helpers: perceptual fingerprints (near-duplicate clips), placeholder slates, thumbnails."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence

from .errors import RenderError
from .media import ffmpeg_bin, run
from .visual import extract_frames

log = logging.getLogger(__name__)


def _dhash(image_path: Path, size: int = 8) -> int:
    from PIL import Image  # pillow is a core dependency
    with Image.open(image_path) as im:
        px = list(im.convert("L").resize((size + 1, size)).getdata())
    bits = 0
    for row in range(size):
        for col in range(size):
            bits = (bits << 1) | (1 if px[row * (size + 1) + col] > px[row * (size + 1) + col + 1] else 0)
    return bits


def clip_fingerprint(video: Path, work: Path, frames: int = 3) -> List[int]:
    """64-bit difference hashes of evenly spaced frames. Empty list when frames cannot be read."""
    out: List[int] = []
    for f in extract_frames(video, work, frames):
        try:
            out.append(_dhash(f))
        except Exception as exc:  # noqa: BLE001
            log.warning("fingerprint failed for %s: %s", video.name, exc)
    return out


def fingerprint_distance(a: Sequence[int], b: Sequence[int]) -> Optional[int]:
    """Mean Hamming distance over aligned frames (None if either side has no frames). 0 = identical, 64 = opposite."""
    n = min(len(a), len(b))
    if n == 0:
        return None
    return round(sum(bin(x ^ y).count("1") for x, y in zip(a[:n], b[:n])) / n)


def make_slate_clip(dest: Path, width: int, height: int, fps: int, seconds: float) -> Path:
    """Plain dark placeholder used when a scene has NO acceptable footage (never presented as a match)."""
    cmd = [ffmpeg_bin(), "-hide_banner", "-y", "-f", "lavfi", "-i",
           f"color=c=0x0B0F1A:s={width}x{height}:r={fps}:d={max(seconds, 0.1):.3f}",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", str(dest)]
    cp = run(cmd, timeout=120)
    if cp.returncode != 0 or not dest.is_file():
        raise RenderError(f"Could not create placeholder slate: {(cp.stderr or '')[-300:]}")
    return dest


def make_thumbnail(video: Path, dest: Path, at_fraction: float = 0.4) -> bool:
    from .media import video_info
    try:
        dur = video_info(video)["duration"] or 1.0
    except ValueError:
        return False
    cmd = [ffmpeg_bin(), "-hide_banner", "-y", "-ss", f"{dur * at_fraction:.3f}", "-i", str(video), "-frames:v", "1",
           "-vf", "scale=480:-2", "-q:v", "4", str(dest)]
    cp = run(cmd, timeout=60)
    return cp.returncode == 0 and dest.is_file() and dest.stat().st_size > 0
