"""Swap the footage of ONE scene of a finished job and re-render only that scene.

A finished job keeps its per-scene H.264 segments (``segments/``) and ``render_plan.json``. A swap
  1. re-matches that scene (excluding the current clip and clips used by other scenes; an optional query steers it),
  2. downloads the new clip and encodes just that scene's segment (framing, colour normalisation, captions at its place),
  3. re-concatenates all segments with stream copy and re-muxes the narration (no other scene is re-encoded),
  4. rewrites that scene in timeline.json (with ``swap_history``).
The response reports measured timings next to the time of the original full render so the saving is visible, not claimed.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from . import artifacts
from .errors import FootageError, InputValidationError, RenderError
from .footage_tools import make_thumbnail
from .matching import SemanticMatcher
from .media import video_info
from .models import Scene
from .normalize import adjustment, measure_clip
from .orchestrator import Deps, _acquire_clip, _ensure_clip
from .renderer import DIMENSIONS, RenderSegment, assemble_video, render_segment_file, scene_geometry
from .validation import validate_output

_locks: Dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def job_lock(key: str) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(key, threading.Lock())


def _load(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise InputValidationError(f"This job cannot be edited: {path.name} is missing or unreadable.") from exc


async def swap_scene(rd: Path, scene_id: str, deps: Deps, job_id: str, *, query: Optional[str] = None,
                     full_render_seconds: Optional[float] = None) -> Dict[str, Any]:
    t_all = time.perf_counter()
    s = deps.settings
    tl = _load(rd / "timeline.json")
    plan = _load(rd / "render_plan.json")
    idx = next((i for i, x in enumerate(tl["scenes"]) if x["scene_id"] == scene_id), None)
    pidx = next((i for i, x in enumerate(plan["segments"]) if x["scene_id"] == scene_id), None)
    if idx is None or pidx is None:
        raise InputValidationError(f"Unknown scene '{scene_id}'.")
    aspect = plan["aspect"]
    W, H = DIMENSIONS[aspect]
    entry = tl["scenes"][idx]
    if entry.get("placeholder"):
        pass                                    # a placeholder slate can be replaced by real footage
    old_clip = (entry.get("clip") or {})
    old_key = f"{old_clip.get('provider')}:{old_clip.get('clip_id')}" if old_clip else None
    used = {f"{x['clip']['provider']}:{x['clip']['clip_id']}" for i, x in enumerate(tl["scenes"]) if x.get("clip")}
    sc = Scene(scene_id, entry["index"], entry["narration_text"], entry["start"], entry["end"], [], entry.get("timing_method", "unknown"))

    clips_dir = rd / "clips"
    clips_dir.mkdir(exist_ok=True)
    cache: Dict[str, Path] = {}

    def _fetch(cand, dest_dir):
        try:
            return _ensure_clip(cand, clips_dir, cache, s)
        except Exception:  # noqa: BLE001
            return None

    matcher = SemanticMatcher(deps.search, s, verifier=deps.verifier, analyzer=deps.analyzer, fetch_preview=_fetch,
                              avoid=tl.get("avoid") or [])
    hint = (query or "").strip() or entry.get("visual_hint")
    t_match = time.perf_counter()
    dec = await matcher.match_scene(sc, aspect, set(used), hint)
    match_seconds = round(time.perf_counter() - t_match, 3)
    ranked = [x for x in dec.ranked if x.candidate.key not in used and x.passed]
    if not ranked and dec.selected is not None and dec.selected.candidate.key != old_key:
        ranked = [dec.selected]
    if not ranked:
        raise FootageError("No different acceptable clip was found for this scene.",
                           details={"scene_id": scene_id, "queries": [a.query for a in dec.attempts]})
    got = await _acquire_clip(ranked[: 1 + s.max_download_alternates], dec, scene_id, clips_dir, cache, [], s, [])
    if got is None:
        raise FootageError("The alternative clips could not be downloaded.", details={"scene_id": scene_id})
    chosen, path, _fp, _near = got
    dec.selected = chosen
    dec.status = "matched" if chosen.passed else dec.status
    dec.reason = f"manual swap ({'query: ' + hint if hint else 'next-best candidate'}) | " + (dec.reason or "")

    # ---- re-encode only this scene
    seg_dir = rd / "segments"
    ps = plan["segments"][pidx]
    out_seg = seg_dir / ps["file"]
    info = video_info(path)
    geometry, (cx, cy, method) = scene_geometry(path, info, W, H)
    measure = measure_clip(path) if (s.normalize_color and plan.get("norm_target")) else None
    adj = adjustment(measure, plan.get("norm_target"))
    ass_name = plan.get("ass_file")
    tmp_seg = out_seg.with_name(f"_new_{out_seg.name}")
    t_seg = time.perf_counter()
    render_segment_file(RenderSegment(scene_id, path, ps["duration"]), aspect, ps["frames"], ps["start_frame"] / plan["fps"], adj,
                        geometry, s, tmp_seg, caption_dir=rd if ass_name else None, ass_name=ass_name)
    segment_seconds = round(time.perf_counter() - t_seg, 3)

    # ---- stitch (stream copy) + narration
    t_asm = time.perf_counter()
    audio = rd / plan["audio_file"] if plan.get("audio_file") else None
    files = [seg_dir / x["file"] for x in plan["segments"]]
    files[pidx] = tmp_seg
    wd = Path(tempfile.mkdtemp(prefix="swap_"))
    try:
        new_video = wd / "video.mp4"
        assemble_video(files, audio, new_video, plan["total_frames"] / plan["fps"], s, wd)
        vres = validate_output(new_video, expected_w=W, expected_h=H, expected_duration=plan["total_frames"] / plan["fps"],
                               expect_audio=audio is not None, fps=plan["fps"], run_detectors=False)
        if not vres["passed"]:
            raise RenderError("The swapped video failed validation; the original was kept.", details={"errors": vres["errors"]})
        os.replace(tmp_seg, out_seg)
        shutil.move(str(new_video), str(rd / "video.mp4.new"))
        os.replace(rd / "video.mp4.new", rd / "video.mp4")
    finally:
        tmp_seg.unlink(missing_ok=True)
        shutil.rmtree(wd, ignore_errors=True)
    assemble_seconds = round(time.perf_counter() - t_asm, 3)

    # ---- bookkeeping
    seg_info = {scene_id: {"looped": info["duration"] > 0 and info["duration"] + 0.05 < ps["duration"], "framing_method": method,
                           "color_normalisation": adj}}
    thumb = make_thumbnail(path, rd / f"thumb_{scene_id}.jpg")
    new_entry = artifacts.build_timeline(
        job_id, aspect, W, H, plan["fps"], tl["audio"]["file"], tl["audio"]["source"], tl["total_duration"], [sc], [dec],
        {chosen.candidate.key: f"clips/{path.name}"}, seg_info, [], thumbnails={scene_id: thumb.name} if thumb else {},
        settings=s)["scenes"][0]
    new_entry["visual_hint"] = entry.get("visual_hint")
    history = list(entry.get("swap_history") or [])
    history.append({"at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "previous_clip": old_key,
                    "new_clip": chosen.candidate.key, "query": hint, "scene_render_seconds": segment_seconds})
    new_entry["swap_history"] = history
    tl["scenes"][idx] = new_entry
    (rd / "timeline.json").write_text(json.dumps(tl, indent=2, ensure_ascii=False), encoding="utf-8")
    ps["clip"] = path.name
    (rd / "render_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")

    total = round(time.perf_counter() - t_all, 3)
    reencoded = ps["frames"]
    return {"scene_id": scene_id, "aspect_ratio": aspect, "previous_clip": old_key, "new_clip": chosen.candidate.key,
            "relevance_score": round(chosen.final_score, 3), "scoring_method": dec.scoring_method,
            "selection_reason": dec.reason, "provider": chosen.candidate.provider,
            "rendering": {"mode": "scene_only", "frames_reencoded": reencoded, "frames_total": plan["total_frames"],
                          "match_and_download_seconds": match_seconds, "scene_render_seconds": segment_seconds,
                          "assemble_seconds": assemble_seconds, "total_seconds": total,
                          "original_full_render_seconds": full_render_seconds},
            "validation_passed": True}
