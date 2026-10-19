"""FFmpeg renderer: one scene -> one clip segment (looped if too short), concatenated, narration muxed,
optional burned-in captions. Output: H.264 (yuv420p, constant 30 fps) + AAC 48 kHz stereo, faststart."""
from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .caption_layout import font_dir_for_ffmpeg, fit_cues, layout_for, to_ass
from .captions import parse_srt
from .config import PipelineSettings
from .errors import RenderError
from .framing import crop_offsets, estimate_focus
from .media import ffmpeg_bin, run, video_info

DIMENSIONS = {"16:9": (1920, 1080), "9:16": (1080, 1920), "1:1": (1080, 1080)}


@dataclass
class RenderSegment:
    scene_id: str
    clip_path: Path
    duration: float


@dataclass
class RenderOutcome:
    output_path: Path
    width: int
    height: int
    warnings: List[str] = field(default_factory=list)
    segments: List[Dict[str, Any]] = field(default_factory=list)
    ffmpeg_seconds: float = 0.0
    caption_info: Optional[Dict[str, Any]] = None


def _escape_filter_path(name: str) -> str:
    return name.replace("\\", "/").replace(":", r"\:").replace("'", r"\'")


def render_video(segments: Sequence[RenderSegment], audio_path: Optional[Path], output_path: Path, aspect: str,
                 settings: PipelineSettings, *, srt_path: Optional[Path] = None, workdir: Optional[Path] = None) -> RenderOutcome:
    import time
    if aspect not in DIMENSIONS:
        raise RenderError(f"Unsupported aspect ratio '{aspect}'.")
    if not segments:
        raise RenderError("No scenes to render.")
    W, H = DIMENSIONS[aspect]
    warnings: List[str] = []
    seg_info: List[Dict[str, Any]] = []
    cmd: List[str] = [ffmpeg_bin(), "-hide_banner", "-y"]
    filters: List[str] = []
    for i, seg in enumerate(segments):
        if not seg.clip_path.is_file():
            raise RenderError(f"Clip for {seg.scene_id} is missing on disk: {seg.clip_path.name}")
        try:
            info = video_info(seg.clip_path)
        except ValueError as exc:
            raise RenderError(f"Clip for {seg.scene_id} is not a valid video: {exc}") from exc
        if info["width"] <= 0 or info["height"] <= 0:
            raise RenderError(f"Clip for {seg.scene_id} has invalid dimensions.")
        looped = info["duration"] > 0 and info["duration"] + 0.05 < seg.duration
        if looped:
            warnings.append(f"{seg.scene_id}: clip is {info['duration']:.1f}s but the scene needs {seg.duration:.1f}s; clip is looped.")
            cmd += ["-stream_loop", "-1"]
        cmd += ["-i", str(seg.clip_path)]
        cx, cy, method = (0.5, 0.5, "center_default(no_crop_needed)")
        src_ratio = info["width"] / info["height"]
        if abs(src_ratio - W / H) / (W / H) > 0.02:
            cx, cy, method = estimate_focus(seg.clip_path)
        sw, sh, xo, yo = crop_offsets(info["width"], info["height"], W, H, cx, cy)
        filters.append(
            f"[{i}:v:0]trim=duration={seg.duration:.3f},setpts=PTS-STARTPTS,scale={sw}:{sh}:flags=lanczos,"
            f"crop={W}:{H}:{xo}:{yo},setsar=1,fps={settings.fps},format=yuv420p[v{i}]")
        seg_info.append({"scene_id": seg.scene_id, "source_size": f"{info['width']}x{info['height']}",
                         "source_duration": round(info["duration"], 3), "scene_duration": round(seg.duration, 3),
                         "looped": looped, "framing_method": method, "focus": [round(cx, 3), round(cy, 3)],
                         "crop_offset": [xo, yo]})
    n = len(segments)
    filters.append("".join(f"[v{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0[vc]")
    label = "vc"
    total = sum(s.duration for s in segments)

    run_cwd = None
    caption_info: Optional[Dict[str, Any]] = None
    if srt_path is not None:
        # Derive a layout from the real output size, measure + wrap + split the cues, and burn an ASS file whose PlayRes
        # equals the frame so all sizes are real pixels (see caption_layout.py for why force_style on SRT is unsafe).
        layout = layout_for(W, H)
        fitted, fit_report = fit_cues(parse_srt(srt_path.read_text(encoding="utf-8")), layout)
        ass_name = f"captions_{aspect.replace(':', 'x')}.ass"
        (srt_path.parent / ass_name).write_text(to_ass(fitted, layout), encoding="utf-8", newline="\n")
        caption_info = {"layout": layout.to_dict(), "fit": fit_report.to_dict(), "ass_file": ass_name}
        # run FFmpeg in the caption folder with a bare relative name: avoids Windows drive-letter escaping problems
        run_cwd = str(srt_path.parent)
        fonts = font_dir_for_ffmpeg()
        fontsdir = f":fontsdir='{_escape_filter_path(fonts)}'" if fonts else ""
        filters.append(f"[{label}]subtitles=filename='{_escape_filter_path(ass_name)}'{fontsdir}[vs]")
        label = "vs"
    cmd += ["-filter_complex", ";".join(filters)]
    if audio_path is not None:
        cmd += ["-i", str(audio_path)]
    cmd += ["-map", f"[{label}]"]
    if audio_path is not None:
        cmd += ["-map", f"{n}:a:0", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"]
    else:
        cmd += ["-an"]
        warnings.append("No audio track was supplied; the video is silent.")
    cmd += ["-t", f"{total:.3f}", "-c:v", "libx264", "-preset", settings.preset, "-crf", str(settings.crf),
            "-profile:v", "high", "-pix_fmt", "yuv420p", "-r", str(settings.fps), "-fps_mode", "cfr",
            "-movflags", "+faststart", str(output_path)]
    t0 = time.time()
    try:
        cp = run(cmd, timeout=settings.render_timeout_seconds, cwd=run_cwd)
    except Exception as exc:  # subprocess.TimeoutExpired etc.
        output_path.unlink(missing_ok=True)
        raise RenderError(f"FFmpeg did not finish: {type(exc).__name__}") from exc
    if cp.returncode != 0 or not output_path.is_file() or output_path.stat().st_size == 0:
        output_path.unlink(missing_ok=True)
        raise RenderError(f"FFmpeg rendering failed: {(cp.stderr or 'no output produced')[-1500:]}")
    return RenderOutcome(output_path, W, H, warnings, seg_info, round(time.time() - t0, 2), caption_info)
