"""FFmpeg renderer. Stage 1: every scene is encoded on its own into a silent H.264 segment (looped if too short, framed,
colour-normalised, captions burned in at the scene's place on the timeline). Stage 2: the segments are concatenated with
stream copy and the narration is muxed (AAC 48 kHz stereo, faststart). Because scenes are separate files, swapping the clip
of one scene only needs that one segment re-encoded (see swap.py). Segment lengths are cut on a cumulative frame grid, so
the total frame count is exactly round(duration * fps) and no drift builds up over many scenes."""
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
from .normalize import adjustment, eq_filter, measure_clip, target_from

DIMENSIONS = {"16:9": (1920, 1080), "9:16": (1080, 1920), "1:1": (1080, 1080)}


@dataclass
class RenderSegment:
    scene_id: str
    clip_path: Path
    duration: float
    normalise: bool = True        # False for generated placeholder slates


@dataclass
class RenderOutcome:
    output_path: Path
    width: int
    height: int
    warnings: List[str] = field(default_factory=list)
    segments: List[Dict[str, Any]] = field(default_factory=list)
    ffmpeg_seconds: float = 0.0
    caption_info: Optional[Dict[str, Any]] = None
    plan: Optional[Dict[str, Any]] = None
    norm_target: Optional[Dict[str, float]] = None


def _escape_filter_path(name: str) -> str:
    return name.replace("\\", "/").replace(":", r"\:").replace("'", r"\'")


def frame_plan(durations: Sequence[float], fps: int) -> List[int]:
    """Frames per scene on a cumulative grid: boundary_i = round(cumulative_seconds_i * fps)."""
    out, acc, prev = [], 0.0, 0
    for d in durations:
        acc += d
        edge = round(acc * fps)
        out.append(max(edge - prev, 1))
        prev += out[-1]
    return out


def scene_geometry(clip_path: Path, info: Dict[str, Any], W: int, H: int):
    cx, cy, method = (0.5, 0.5, "center_default(no_crop_needed)")
    if abs(info["width"] / info["height"] - W / H) / (W / H) > 0.02:
        cx, cy, method = estimate_focus(clip_path)
    sw, sh, xo, yo = crop_offsets(info["width"], info["height"], W, H, cx, cy)
    return {"sw": sw, "sh": sh, "xo": xo, "yo": yo}, (cx, cy, method)


def _segment_cmd(seg: RenderSegment, info: Dict[str, Any], W: int, H: int, frames: int, start_seconds: float, adj: Dict[str, Any],
                 geometry: Dict[str, Any], settings: PipelineSettings, out: Path, *, ass_name: Optional[str],
                 fontsdir: str) -> List[str]:
    fps = settings.fps
    looped = info["duration"] > 0 and info["duration"] + 0.05 < frames / fps
    cmd = [ffmpeg_bin(), "-hide_banner", "-y"]
    if looped:
        cmd += ["-stream_loop", "-1"]
    cmd += ["-i", str(seg.clip_path)]
    chain = (f"trim=duration={(frames + 0.5) / fps:.4f},setpts=PTS-STARTPTS,scale={geometry['sw']}:{geometry['sh']}:flags=lanczos,"
             f"crop={W}:{H}:{geometry['xo']}:{geometry['yo']},setsar=1,fps={fps},{eq_filter(adj)}format=yuv420p")
    if ass_name:
        chain += (f",setpts=PTS+{start_seconds:.4f}/TB,subtitles=filename='{_escape_filter_path(ass_name)}'{fontsdir},"
                  "setpts=PTS-STARTPTS")
    cmd += ["-filter_complex", f"[0:v:0]{chain}[v]", "-map", "[v]", "-an", "-frames:v", str(frames),
            "-c:v", "libx264", "-preset", settings.preset, "-crf", str(settings.crf), "-profile:v", "high",
            "-pix_fmt", "yuv420p", "-r", str(fps), "-fps_mode", "cfr", str(out)]
    return cmd


def render_segment_file(seg: RenderSegment, aspect: str, frames: int, start_seconds: float, adj: Dict[str, Any],
                        geometry: Dict[str, Any], settings: PipelineSettings, out: Path, *, caption_dir: Optional[Path],
                        ass_name: Optional[str]) -> None:
    W, H = DIMENSIONS[aspect]
    info = video_info(seg.clip_path)
    fonts = font_dir_for_ffmpeg() if ass_name else None
    fontsdir = f":fontsdir='{_escape_filter_path(fonts)}'" if fonts else ""
    cmd = _segment_cmd(seg, info, W, H, frames, start_seconds, adj, geometry, settings, out, ass_name=ass_name, fontsdir=fontsdir)
    try:
        cp = run(cmd, timeout=settings.render_timeout_seconds, cwd=str(caption_dir) if (ass_name and caption_dir) else None)
    except Exception as exc:  # subprocess.TimeoutExpired etc.
        out.unlink(missing_ok=True)
        raise RenderError(f"FFmpeg did not finish: {type(exc).__name__}") from exc
    if cp.returncode != 0 or not out.is_file() or out.stat().st_size == 0:
        out.unlink(missing_ok=True)
        raise RenderError(f"FFmpeg rendering failed: {(cp.stderr or 'no output produced')[-1500:]}")


def assemble_video(segment_files: Sequence[Path], audio_path: Optional[Path], output_path: Path, total_seconds: float,
                   settings: PipelineSettings, workdir: Path) -> None:
    """Concatenate encoded segments with stream copy and mux the narration. No video re-encode."""
    listing = workdir / "segments.txt"
    lines = []
    for f in segment_files:
        p = str(Path(f).resolve()).replace("\\", "/").replace("'", "'\\''")
        lines.append(f"file '{p}'\n")
    listing.write_text("".join(lines), encoding="utf-8")
    cmd = [ffmpeg_bin(), "-hide_banner", "-y", "-f", "concat", "-safe", "0", "-i", str(listing)]
    if audio_path is not None:
        cmd += ["-i", str(audio_path), "-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"]
    else:
        cmd += ["-map", "0:v:0", "-an"]
    cmd += ["-c:v", "copy", "-t", f"{total_seconds:.3f}", "-movflags", "+faststart", str(output_path)]
    try:
        cp = run(cmd, timeout=settings.render_timeout_seconds)
    except Exception as exc:  # noqa: BLE001
        output_path.unlink(missing_ok=True)
        raise RenderError(f"FFmpeg did not finish: {type(exc).__name__}") from exc
    if cp.returncode != 0 or not output_path.is_file() or output_path.stat().st_size == 0:
        output_path.unlink(missing_ok=True)
        raise RenderError(f"FFmpeg assembling failed: {(cp.stderr or 'no output produced')[-1500:]}")


def render_video(segments: Sequence[RenderSegment], audio_path: Optional[Path], output_path: Path, aspect: str,
                 settings: PipelineSettings, *, srt_path: Optional[Path] = None, workdir: Optional[Path] = None,
                 segment_dir: Optional[Path] = None) -> RenderOutcome:
    """Render all scenes. ``segment_dir`` keeps the per-scene segment files (needed for fast clip swaps); without it they
    live in a temporary folder that is removed afterwards. ``outcome.plan`` describes how to rebuild any single scene."""
    import time
    if aspect not in DIMENSIONS:
        raise RenderError(f"Unsupported aspect ratio '{aspect}'.")
    if not segments:
        raise RenderError("No scenes to render.")
    W, H = DIMENSIONS[aspect]
    fps = settings.fps
    warnings: List[str] = []
    seg_info: List[Dict[str, Any]] = []
    infos: List[Dict[str, Any]] = []
    for seg in segments:
        if not seg.clip_path.is_file():
            raise RenderError(f"Clip for {seg.scene_id} is missing on disk: {seg.clip_path.name}")
        try:
            info = video_info(seg.clip_path)
        except ValueError as exc:
            raise RenderError(f"Clip for {seg.scene_id} is not a valid video: {exc}") from exc
        if info["width"] <= 0 or info["height"] <= 0:
            raise RenderError(f"Clip for {seg.scene_id} has invalid dimensions.")
        infos.append(info)

    frames = frame_plan([s.duration for s in segments], fps)
    t_start = time.time()

    # colour normalisation: measure every real clip, aim at the median of the job
    use_norm = getattr(settings, "normalize_color", True)
    measures = [measure_clip(s.clip_path) if (s.normalise and use_norm) else None for s in segments]
    target = target_from(measures)
    adjs = [adjustment(m, target) for m in measures]

    run_cwd: Optional[Path] = None
    ass_name: Optional[str] = None
    caption_info: Optional[Dict[str, Any]] = None
    if srt_path is not None:
        # Derive a layout from the real output size, measure + wrap + split the cues, and burn an ASS file whose PlayRes
        # equals the frame so all sizes are real pixels (see caption_layout.py for why force_style on SRT is unsafe).
        layout = layout_for(W, H)
        fitted, fit_report = fit_cues(parse_srt(srt_path.read_text(encoding="utf-8")), layout)
        ass_name = f"captions_{aspect.replace(':', 'x')}.ass"
        (srt_path.parent / ass_name).write_text(to_ass(fitted, layout), encoding="utf-8", newline="\n")
        caption_info = {"layout": layout.to_dict(), "fit": fit_report.to_dict(), "ass_file": ass_name}
        run_cwd = srt_path.parent    # bare relative name: avoids Windows drive-letter escaping problems

    own_tmp = segment_dir is None
    seg_dir = Path(tempfile.mkdtemp(prefix="segments_")) if own_tmp else segment_dir
    seg_dir.mkdir(parents=True, exist_ok=True)
    plan_segments: List[Dict[str, Any]] = []
    files: List[Path] = []
    try:
        acc_frames = 0
        for i, seg in enumerate(segments):
            info = infos[i]
            geometry, (cx, cy, method) = scene_geometry(seg.clip_path, info, W, H)
            xo, yo = geometry["xo"], geometry["yo"]
            looped = info["duration"] > 0 and info["duration"] + 0.05 < frames[i] / fps
            if looped:
                warnings.append(f"{seg.scene_id}: clip is {info['duration']:.1f}s but the scene needs {seg.duration:.1f}s; clip is looped.")
            out = seg_dir / f"{i:03d}_{seg.scene_id}.mp4"
            render_segment_file(seg, aspect, frames[i], acc_frames / fps, adjs[i], geometry, settings, out,
                                caption_dir=run_cwd, ass_name=ass_name)
            files.append(out)
            seg_info.append({"scene_id": seg.scene_id, "source_size": f"{info['width']}x{info['height']}",
                             "source_duration": round(info["duration"], 3), "scene_duration": round(seg.duration, 3),
                             "frames": frames[i], "looped": looped, "framing_method": method, "focus": [round(cx, 3), round(cy, 3)],
                             "crop_offset": [xo, yo], "color_normalisation": adjs[i]})
            plan_segments.append({"scene_id": seg.scene_id, "file": out.name, "frames": frames[i], "start_frame": acc_frames,
                                  "duration": round(seg.duration, 3), "clip": seg.clip_path.name, "normalise": seg.normalise})
            acc_frames += frames[i]
        wd = Path(tempfile.mkdtemp(prefix="assemble_"))
        try:
            if audio_path is None:
                warnings.append("No audio track was supplied; the video is silent.")
            assemble_video(files, audio_path, output_path, acc_frames / fps, settings, wd)
        finally:
            shutil.rmtree(wd, ignore_errors=True)
    finally:
        if own_tmp:
            shutil.rmtree(seg_dir, ignore_errors=True)
    plan = {"aspect": aspect, "fps": fps, "width": W, "height": H, "total_frames": acc_frames, "segments": plan_segments,
            "norm_target": target, "ass_file": ass_name, "audio_file": audio_path.name if audio_path else None}
    return RenderOutcome(output_path, W, H, warnings, seg_info, round(time.time() - t_start, 2), caption_info, plan, target)
