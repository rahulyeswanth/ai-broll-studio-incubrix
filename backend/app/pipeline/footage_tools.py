"""Small FFmpeg/Pillow helpers: near-duplicate fingerprints, "no footage" slates, scene thumbnails."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List, Optional, Sequence

from .media import ffmpeg_bin, run, video_info
from .visual import extract_frames


def _dhash(img_path: Path, size: int = 8) -> int:
    from PIL import Image  # pillow is a declared dependency
    with Image.open(img_path) as im:
        g = im.convert("L").resize((size + 1, size))
        px = list(g.tobytes())
    bits = 0
    for row in range(size):
        for col in range(size):
            bits = (bits << 1) | (1 if px[row * (size + 1) + col] > px[row * (size + 1) + col + 1] else 0)
    return bits


def clip_fingerprint(path: Path, frames: int = 3) -> List[int]:
    """64-bit difference hashes of evenly spaced frames. Empty list if frames cannot be read (then no dedupe is claimed)."""
    with tempfile.TemporaryDirectory(prefix="fp_") as tmp:
        out: List[int] = []
        for f in extract_frames(path, Path(tmp), frames):
            try:
                out.append(_dhash(f))
            except Exception:  # noqa: BLE001 - unreadable frame: skip it
                continue
        return out


def fingerprint_distance(a: Sequence[int], b: Sequence[int]) -> Optional[int]:
    """Mean Hamming distance over frame pairs, or None when either side has no usable fingerprint."""
    n = min(len(a), len(b))
    if n == 0:
        return None
    return round(sum(bin(a[i] ^ b[i]).count("1") for i in range(n)) / n)


def make_slate(dest: Path, seconds: float, width: int, height: int, fps: int) -> Path:
    """A plain dark clip used for a scene that has NO acceptable footage. Deliberately text-free: the narration
    (and captions, if enabled) carry the content, and the scene is flagged as unmatched everywhere else."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    cp = run([ffmpeg_bin(), "-hide_banner", "-y", "-f", "lavfi", "-i",
              f"color=c=0x0B0F1A:s={width}x{height}:r={fps}:d={max(seconds, 0.2):.3f}",
              "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", str(dest)], timeout=120)
    if cp.returncode != 0 or not dest.is_file():
        raise RuntimeError((cp.stderr or "slate generation failed")[-400:])
    return dest


def make_thumbnail(clip: Path, dest: Path, at_fraction: float = 0.4, width: int = 480) -> Optional[Path]:
    try:
        dur = video_info(clip)["duration"] or 1.0
    except ValueError:
        return None
    cp = run([ffmpeg_bin(), "-hide_banner", "-y", "-ss", f"{dur * at_fraction:.3f}", "-i", str(clip), "-frames:v", "1",
              "-vf", f"scale={width}:-2", "-q:v", "4", str(dest)], timeout=60)
    return dest if cp.returncode == 0 and dest.is_file() and dest.stat().st_size > 0 else None
