"""The single end-to-end workflow: validate -> transcript/scenes -> narration -> match -> download ->
captions -> render -> validate -> timeline.json + manifest.json."""
from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import json
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

from . import artifacts
from .diagnostics import build_diagnostics
from .footage_tools import clip_fingerprint, fingerprint_distance, make_slate, make_thumbnail
from .alignment import map_script_to_timed_words, scenes_from_words, similarity, tokenize
from .captions import build_cues, to_srt, validate_srt
from .config import PipelineSettings
from .downloader import download_clip
from .errors import (DownloadError, FootageError, InputValidationError, PipelineError, ProviderUnavailableError,
                     RenderError, ValidationFailure)
from .matching import SceneDecision, SemanticMatcher, SearchFn
from .media import validate_audio_file
from .models import Scene, WordTiming
from .narration import PipelineTTS, narrate_scenes
from .renderer import DIMENSIONS, RenderSegment, render_video
from .scene_analysis import RuleBasedSceneAnalyzer, SceneAnalyzer
from .text import build_scene_texts, normalize_whitespace
from .transcription import PipelineTranscriber
from .validation import validate_output
from .visual import NullVisualVerifier, VisualVerifier

ProgressFn = Callable[[float, str], None]
SCRIPT_AUDIO_MATCH_THRESHOLD = 0.80   # below this similarity the transcript of the real audio wins over the script


@dataclass
class JobRequest:
    script_text: Optional[str] = None
    audio_path: Optional[Path] = None          # already saved inside the job dir by the API layer
    audio_original_name: Optional[str] = None
    aspect_ratio: str = "16:9"
    voice: Optional[str] = None
    language: Optional[str] = None
    captions_enabled: bool = True
    burn_captions: bool = True
    footage_provider: Optional[str] = None
    scene_hints: List[Optional[str]] = field(default_factory=list)
    aspect_ratios: List[str] = field(default_factory=list)    # >1 entries => multi-ratio job (see multi.py)
    style: Optional[str] = None
    avoid: List[str] = field(default_factory=list)


@dataclass
class Deps:
    settings: PipelineSettings
    search: SearchFn
    footage_provider_name: str
    get_tts: Callable[[], PipelineTTS]
    get_transcriber: Callable[[], PipelineTranscriber]
    verifier: VisualVerifier = field(default_factory=NullVisualVerifier)
    analyzer: SceneAnalyzer = field(default_factory=RuleBasedSceneAnalyzer)
    now: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.timezone.utc)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", s)[:80]


def validate_script(text: Optional[str], settings: PipelineSettings) -> str:
    t = (text or "").strip()
    if not t:
        raise InputValidationError("The script is empty.")
    if len(t) > settings.max_script_chars:
        raise InputValidationError(f"The script has {len(t)} characters; the limit is {settings.max_script_chars}.")
    if not re.search(r"\w", t, re.UNICODE):
        raise InputValidationError("The script contains no readable words.")
    return t


async def run_pipeline(req: JobRequest, deps: Deps, job_dir: Path, job_id: str, progress: Optional[ProgressFn] = None) -> Dict[str, Any]:
    s = deps.settings
    P = progress or (lambda pct, msg: None)
    timings: Dict[str, float] = {}
    warnings: List[str] = []
    events: List[Dict[str, Any]] = []
    created_at = deps.now().isoformat(timespec="seconds")
    job_dir.mkdir(parents=True, exist_ok=True)
    work = job_dir / "tmp"
    clips_dir = job_dir / "clips"
    clips_dir.mkdir(exist_ok=True)
    work.mkdir(exist_ok=True)

    class Stage:
        def __init__(self, name): self.name = name
        def __enter__(self): self.t = time.perf_counter(); return self
        def __exit__(self, *a): timings[self.name] = round(time.perf_counter() - self.t, 3)

    try:
        if req.aspect_ratio not in DIMENSIONS:
            raise InputValidationError(f"Unsupported aspect ratio '{req.aspect_ratio}'. Use 16:9, 9:16 or 1:1.")
        width, height = DIMENSIONS[req.aspect_ratio]
        has_audio = req.audio_path is not None
        has_script = bool((req.script_text or "").strip())
        if not has_audio and not has_script:
            raise InputValidationError("Provide a script, an uploaded audio file, or both.")
        script = validate_script(req.script_text, s) if has_script else None
        language = (req.language or s.default_language)
        if script and sum(1 for ch in script if ord(ch) > 0x24F) > 0.3 * len(script):
            warnings.append("The script is mostly non-Latin text. Scene analysis and stock-footage search are English-only, "
                            "so B-roll relevance will be poor; consider an English visual hint per scene.")

        # ------------------------------------------------------------------ narration / transcript
        asr_meta: Dict[str, Any] = {"name": None}
        tts_meta: Dict[str, Any] = {"name": None}
        input_meta: Dict[str, Any] = {}
        reconcile_note: Optional[Dict[str, Any]] = None
        if has_audio:
            input_type = "uploaded_audio"
            P(4, "Validating uploaded audio")
            with Stage("validate_input"):
                info = validate_audio_file(req.audio_path, s, original_name=req.audio_original_name)
            input_meta = {"filename": req.audio_original_name or req.audio_path.name, "size_bytes": info.size_bytes,
                          "duration_seconds": info.duration_seconds, "sha256": _sha256(req.audio_path),
                          "audio": info.to_dict(), "script_also_provided": has_script,
                          "script_characters": len(script) if script else None}
            P(8, "Transcribing audio (speech-to-text)")
            transcriber = deps.get_transcriber()
            with Stage("transcription"):
                asr = await transcriber.transcribe(req.audio_path, req.language)
            if not asr.words:
                raise InputValidationError("No speech was detected in the uploaded audio.")
            asr_meta = {"name": transcriber.name, "model": transcriber.model, "version": transcriber.version,
                        "is_real": transcriber.is_real, "detected_language": asr.language,
                        "language_probability": asr.language_probability, "timing_granularity": asr.granularity}
            if not transcriber.is_real:
                warnings.append("MOCK TRANSCRIPTION: the transcript is canned text and does NOT reflect your audio.")
                events.append({"type": "mock_provider", "provider": "transcription", "message": "mock transcriber used"})
            timing_method = "asr_word_timestamps" if asr.granularity == "word" else "asr_segment_interpolated"
            if asr.granularity != "word":
                warnings.append("The speech recogniser returned segment-level timing only; word timings inside segments are interpolated.")
                events.append({"type": "timing_fallback", "message": "segment-level ASR timing, words interpolated"})
            words = asr.words
            if script:
                sim = similarity(script, asr.text)
                reconcile_note = {"similarity": round(sim, 3), "threshold": SCRIPT_AUDIO_MATCH_THRESHOLD}
                if sim >= SCRIPT_AUDIO_MATCH_THRESHOLD:
                    mapped, matched = map_script_to_timed_words(tokenize(script), words, 0.0, info.duration_seconds)
                    words = mapped
                    reconcile_note.update({"rule": "script_wording_with_audio_timing", "script_tokens_anchored": round(matched, 3)})
                    if sim < 0.98:
                        warnings.append(f"The script and the speech differ slightly (similarity {sim:.0%}). The script's wording is "
                                        "used for text/captions; all timings come from the uploaded audio.")
                        events.append({"type": "script_audio_mismatch", "similarity": round(sim, 3)})
                else:
                    reconcile_note.update({"rule": "transcript_of_audio_used_script_ignored"})
                    warnings.append(f"The script and the uploaded audio differ substantially (similarity {sim:.0%}). The transcript of "
                                    "the audio is used; the script text was ignored. Uploaded audio is the source of truth.")
                    events.append({"type": "script_audio_mismatch", "similarity": round(sim, 3), "action": "script_ignored"})
            with Stage("scene_building"):
                scenes = scenes_from_words(words, info.duration_seconds, s, timing_method)
            if not scenes:
                raise InputValidationError("Could not build any scenes from the audio.")
            audio_src = req.audio_path
            audio_duration = info.duration_seconds
            audio_ext = req.audio_path.suffix.lower()
            audio_out = job_dir / f"narration{audio_ext}"
            if audio_src.resolve() != audio_out.resolve():
                shutil.copyfile(audio_src, audio_out)
            audio_source, caption_method = "uploaded", timing_method
        else:
            input_type = "script_only"
            P(4, "Validating script")
            scene_texts = build_scene_texts(script)
            if not scene_texts:
                raise InputValidationError("The script contains no narratable text.")
            input_meta = {"script_characters": len(script), "script_words": len(script.split()),
                          "script_sha256": hashlib.sha256(script.encode("utf-8")).hexdigest(),
                          "scene_count": len(scene_texts), "voice": req.voice or s.default_voice, "language": language}
            tts = deps.get_tts()
            P(8, f"Generating narration with {tts.name}")
            with Stage("narration"):
                nar = await narrate_scenes(scene_texts, tts, req.voice or s.default_voice, language, work, s,
                                           progress=lambda i, n: P(8 + 27 * i / n, f"Narrating scene {i}/{n}"))
            tts_meta = {"name": tts.name, "model": tts.model, "version": tts.version, "is_real": tts.is_real,
                        "voice": req.voice or s.default_voice, "language": language}
            warnings += nar.warnings
            if not tts.is_real:
                events.append({"type": "mock_provider", "provider": "tts", "message": "placeholder tones instead of speech"})
            if nar.timing_method not in ("tts_word_boundaries",):
                events.append({"type": "timing_fallback", "message": f"word timing method: {nar.timing_method}"})
                warnings.append(f"Caption word timing method is '{nar.timing_method}' (scene boundaries are exact; "
                                "timing inside scenes is approximate).")
            scenes = nar.scenes
            audio_duration = nar.duration_seconds
            audio_out = job_dir / "narration.mp3"
            shutil.copyfile(nar.audio_path, audio_out)
            audio_source, caption_method = ("generated:" + tts.name), nar.timing_method
            timings.setdefault("validate_input", 0.0)
        input_meta["scene_count"] = len(scenes)
        if reconcile_note:
            input_meta["script_audio_reconciliation"] = reconcile_note

        # ------------------------------------------------------------------ matching
        P(36, "Matching footage to scene meaning")
        clip_cache: Dict[str, Path] = {}

        def _fetch(cand, dest_dir: Path) -> Optional[Path]:
            try:
                return _ensure_clip(cand, clips_dir, clip_cache, s)
            except DownloadError:
                return None

        matcher = SemanticMatcher(deps.search, s, verifier=deps.verifier, analyzer=deps.analyzer, fetch_preview=_fetch,
                                avoid=req.avoid)
        with Stage("matching"):
            decisions = await matcher.match_scenes(scenes, req.aspect_ratio, req.scene_hints or None,
                                                   progress=lambda i, n: P(36 + 24 * i / n, f"Matched scene {i}/{n}"))
        for d in decisions:
            warnings += d.warnings
            if d.status == "fallback_low_confidence":
                events.append({"type": "low_confidence_fallback", "scene_id": d.scene_id, "message": d.reason})
            if d.reused_clip:
                events.append({"type": "clip_reuse", "scene_id": d.scene_id, "message": "clip reused; see scene warnings"})
        unmatched = [d for d in decisions if d.selected is None]

        def _unmatched_error(kind: type) -> PipelineError:
            return kind(
                "No suitable footage could be found for: " + ", ".join(d.scene_id for d in unmatched),
                details={"scenes": [{"scene_id": d.scene_id, "narration": d.narration, "status": d.status, "reason": d.reason,
                                     "queries": [a.query for a in d.attempts],
                                     "errors": [a.error for a in d.attempts if a.error]} for d in unmatched]})
        # "fail" policy keeps the strict legacy behaviour; otherwise an unmatched scene gets an honest placeholder and the
        # rest of the video is still produced. If NOTHING matched there is no B-roll video to make.
        if unmatched and (s.fallback_policy == "fail" or len(unmatched) == len(decisions)):
            raise _unmatched_error(FootageError)

        # ------------------------------------------------------------------ download
        P(60, "Downloading selected clips")
        clip_files: Dict[str, str] = {}
        segs: List[RenderSegment] = []
        slate_ids: List[str] = []
        fingerprints: List[tuple] = []          # (scene_id, clip_key, fingerprint) of clips already placed in the video
        any_download_failure = False
        with Stage("download"):
            for i, (sc, d) in enumerate(zip(scenes, decisions), 1):
                seg_len = max(sc.end - sc.start, 0.05)
                path = None
                if d.selected is not None:
                    used_before = {decisions[j].selected.candidate.key for j in range(i - 1) if decisions[j].selected}
                    tried = [d.selected] + [x for x in d.ranked if x is not d.selected and x.passed
                                            and x.candidate.key not in used_before][: s.max_download_alternates]
                    got = await _acquire_clip(tried, d, sc.scene_id, clips_dir, clip_cache, fingerprints, s, events)
                    any_download_failure = any_download_failure or any(x["outcome"] == "download_failed" for x in d.download_log)
                    if got is not None:
                        chosen, path, chosen_fp, near = got
                        if chosen is not d.selected:
                            d.reason += f" | original choice unusable; replaced by next-best acceptable candidate ({chosen.candidate.key})"
                            d.used_alternate = True
                            d.selected = chosen
                            if chosen.passed and d.status == "fallback_low_confidence":
                                d.status = "matched"
                            warnings.append(f"{sc.scene_id}: top clip could not be used; used next-best candidate.")
                            events.append({"type": "download_fallback", "scene_id": sc.scene_id, "message": f"used {chosen.candidate.key}"})
                        if near:
                            d.warnings.append(f"This clip closely resembles the footage in {near[0]}; no more distinct acceptable clip was found.")
                            events.append({"type": "near_duplicate_footage", "scene_id": sc.scene_id, "message": f"similar to {near[0]}"})
                        fingerprints.append((sc.scene_id, chosen.candidate.key, chosen_fp))
                    else:
                        # every candidate failed: an honest unmatched scene, never a fake match
                        if s.fallback_policy == "fail":
                            raise DownloadError(f"Could not download footage for {sc.scene_id}.",
                                                details={"scene_id": sc.scene_id, "attempts": [x.candidate.key for x in tried]})
                        d.selected, d.status = None, "unmatched"
                        d.reason = f"all {len(tried)} candidate clip(s) failed to download or validate"
                        d.warnings.append(f"Scene {sc.scene_id} has no matching footage: {d.reason}.")
                if d.selected is not None and path is not None:
                    clip_files[d.selected.candidate.key] = f"clips/{path.name}"
                    segs.append(RenderSegment(sc.scene_id, path, seg_len))
                else:
                    slate = await asyncio.to_thread(make_slate, clips_dir / f"slate_{sc.scene_id}.mp4", seg_len, width, height, s.fps)
                    slate_ids.append(sc.scene_id)
                    segs.append(RenderSegment(sc.scene_id, slate, seg_len))
                P(60 + 15 * i / len(scenes), f"Prepared footage for scene {i}/{len(scenes)}")
        if len(slate_ids) == len(decisions):
            unmatched = [d for d in decisions if d.selected is None]
            raise _unmatched_error(DownloadError if any_download_failure else FootageError)

        # ------------------------------------------------------------------ captions
        all_words: List[WordTiming] = [w for sc in scenes for w in sc.words]
        srt_path: Optional[Path] = None
        srt_problems: List[str] = []
        cue_count = 0
        if req.captions_enabled:
            P(76, "Generating synchronized captions")
            with Stage("captions"):
                cues = build_cues(all_words, audio_duration, s)
                srt_text = to_srt(cues)
                srt_problems = validate_srt(srt_text, audio_duration, s)
                if srt_problems:
                    raise ValidationFailure("Generated captions failed validation.", details={"problems": srt_problems})
                srt_path = job_dir / "captions.srt"
                srt_path.write_text(srt_text, encoding="utf-8", newline="\n")
                cue_count = len(cues)
        else:
            timings["captions"] = 0.0

        # ------------------------------------------------------------------ render
        P(80, "Rendering video with FFmpeg")
        out_path = job_dir / "video.mp4"
        with Stage("render"):
            outcome = await asyncio.to_thread(
                render_video, segs, audio_out, out_path, req.aspect_ratio, s,
                srt_path=srt_path if (srt_path and req.burn_captions) else None)
        warnings += outcome.warnings
        thumbs: Dict[str, str] = {}
        for sg in segs:
            if sg.scene_id in slate_ids:
                continue
            th = await asyncio.to_thread(make_thumbnail, sg.clip_path, job_dir / f"thumb_{sg.scene_id}.jpg")
            if th is not None:
                thumbs[sg.scene_id] = th.name
        for seg in outcome.segments:
            if seg["looped"]:
                events.append({"type": "clip_looped", "scene_id": seg["scene_id"], "message": "clip shorter than scene"})

        # ------------------------------------------------------------------ validate + artifacts
        P(94, "Validating output")
        with Stage("validation"):
            vres = await asyncio.to_thread(validate_output, out_path, expected_w=width, expected_h=height,
                                           expected_duration=sum(x.duration for x in segs), expect_audio=True, fps=s.fps,
                                           ignore_ranges=[(sc.start, sc.end) for sc in scenes if sc.scene_id in slate_ids])
        warnings += vres["warnings"]
        seg_by_id = {x["scene_id"]: x for x in outcome.segments}
        sel_keys = [d.selected.candidate.key for d in decisions if d.selected]
        dupes = [k for k in set(sel_keys) if sel_keys.count(k) > 1]
        timeline = artifacts.build_timeline(job_id, req.aspect_ratio, width, height, s.fps, audio_out.name, audio_source,
                                            audio_duration, scenes, decisions, clip_files, seg_by_id, [],
                                            thumbnails=thumbs, slate_ids=slate_ids, settings=s)
        tl_problems = artifacts.validate_timeline(timeline, audio_duration)
        validation = {"video": vres, "srt_problems": srt_problems, "srt_valid": not srt_problems if srt_path else None,
                      "timeline_problems": tl_problems, "timeline_valid": not tl_problems,
                      "duplicate_clips_in_video": dupes,
                      "narration_covers_video": abs(audio_duration - vres["checks"].get("duration_seconds", 0)) <= 0.3}
        providers = {
            "tts": tts_meta, "transcription": asr_meta,
            "footage": {"name": deps.footage_provider_name},
            "visual_verifier": {"name": deps.verifier.name, "version": deps.verifier.version,
                                "performs_visual_analysis": deps.verifier.performs_visual_analysis,
                                "requested": s.visual_verifier,
                                "unavailable_reason": getattr(deps.verifier, "unavailable_reason", None)},
            "scene_analyzer": {"name": decisions[0].analysis.method if decisions else deps.analyzer.__class__.__name__},
        }
        methods = sorted({d.scoring_method for d in decisions})
        matching = {
            "strategy": "analyse scene -> multi-query search -> filter -> score -> threshold -> retry -> disclosed fallback",
            "scoring_methods_used": methods, "threshold": s.match_threshold,
            "weights": {"visual": s.weight_visual, "text": s.weight_text, "technical": s.weight_technical},
            "visual_ai_verification_performed": any(d.selected and d.selected.visual_verified for d in decisions),
            "scenes_total": len(decisions),
            "scenes_matched": sum(d.status == "matched" for d in decisions),
            "scenes_low_confidence_fallback": sum(d.status == "fallback_low_confidence" for d in decisions),
            "scenes_needs_review": sum(d.status == "fallback_low_confidence" for d in decisions),
            "scenes_unmatched": sum(d.selected is None for d in decisions),
            "review_floor": s.review_floor, "high_confidence": s.high_confidence,
            "relevance_note": "scores are uncalibrated relevance rankings, not probabilities; labels use documented thresholds",
            "fallback_policy": s.fallback_policy,
        }
        if getattr(deps.verifier, "unavailable_reason", None):
            msg = f"Visual verification was requested ({s.visual_verifier}) but is unavailable: {deps.verifier.unavailable_reason}"
            warnings.append(msg)
            events.append({"type": "visual_verifier_unavailable", "message": msg})
        if not matching["visual_ai_verification_performed"]:
            msg = ("Footage relevance was estimated from provider titles/tags and scene analysis only. "
                   "No vision-language model inspected the video frames.")
            warnings.append(msg)
            events.append({"type": "metadata_only_matching", "message": msg})
        captions_meta = {"enabled": req.captions_enabled, "file": srt_path.name if srt_path else None,
                         "burned_in": bool(srt_path and req.burn_captions), "cue_count": cue_count,
                         "timing_method": caption_method,
                         "alignment_precision": {"tts_word_boundaries": "word-level (TTS engine boundaries)",
                                                 "asr_word_timestamps": "word-level (ASR timestamps)"}.get(
                             caption_method, "approximate: " + caption_method)}
        outputs = {"video": {"file": out_path.name, "width": width, "height": height, "fps": s.fps,
                             "duration_seconds": vres["checks"].get("duration_seconds"),
                             "size_bytes": out_path.stat().st_size},
                   "audio": {"file": audio_out.name, "duration_seconds": audio_duration, "source": audio_source},
                   "captions": srt_path.name if srt_path else None, "timeline": "timeline.json", "manifest": "manifest.json"}
        status = "completed" if vres["passed"] and not tl_problems else "completed_with_validation_errors"
        P(98, "Writing timeline and manifest")
        timeline["warnings"] = list(dict.fromkeys(warnings))
        (job_dir / "timeline.json").write_text(json.dumps(timeline, indent=2, ensure_ascii=False), encoding="utf-8")
        diag = build_diagnostics(decisions, raw_warnings=warnings, events=events, verifier=deps.verifier,
                                 visual_performed=matching["visual_ai_verification_performed"], slate_scene_ids=slate_ids,
                                 looped_scene_ids=[x["scene_id"] for x in outcome.segments if x["looped"]],
                                 validation_warnings=vres["warnings"])
        manifest = artifacts.build_manifest(
            job_id=job_id, input_type=input_type, input_meta=input_meta, aspect=req.aspect_ratio, outputs=outputs,
            providers=providers, matching=matching, captions=captions_meta, processing=s.public_dict(),
            durations={**timings, "total": round(sum(timings.values()), 3)}, timeline=timeline,
            warnings=warnings, events=events, validation=validation, created_at=created_at, status=status)
        manifest["diagnostics"] = diag
        m_problems = artifacts.validate_manifest(manifest)
        if m_problems:
            raise ValidationFailure("manifest.json is incomplete", details={"problems": m_problems})
        (job_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        if not vres["passed"] or tl_problems:
            raise ValidationFailure("The rendered video failed validation.",
                                    details={"video_errors": vres["errors"], "timeline_problems": tl_problems})
        P(100, "Done")
        return manifest
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _ensure_clip(cand, clips_dir: Path, cache: Dict[str, Path], settings: PipelineSettings) -> Path:
    if cand.key in cache and cache[cand.key].is_file():
        return cache[cand.key]
    dest = clips_dir / f"{_safe_name(cand.provider)}_{_safe_name(cand.clip_id)}.mp4"
    if not dest.is_file():
        download_clip(cand.download_url, dest, settings)
    cache[cand.key] = dest
    return dest


async def _acquire_clip(tried, d, scene_id, clips_dir, cache, fingerprints, s, events):
    """Download candidates in rank order. A download failure moves on to the next one (and is logged). A clip that looks
    like footage already used in another scene is skipped while alternatives remain; if only look-alikes remain the first
    one is used (the caller discloses it). Returns (candidate, path, fingerprint, near) or None."""
    first_dup = None
    for cand_s in tried:
        key = cand_s.candidate.key
        try:
            cpath = await asyncio.to_thread(_ensure_clip, cand_s.candidate, clips_dir, cache, s)
        except DownloadError as exc:
            d.download_log.append({"clip": key, "outcome": "download_failed", "reason": exc.message})
            events.append({"type": "download_failure", "scene_id": scene_id, "clip": key, "message": exc.message})
            continue
        fp = await asyncio.to_thread(clip_fingerprint, cpath)
        near = None
        for (o_sid, o_key, o_fp) in fingerprints:
            if o_key == key:
                continue          # exact reuse is handled (and disclosed) by the matcher
            dist = fingerprint_distance(fp, o_fp)
            if dist is not None and dist <= s.duplicate_hash_distance:
                near = (o_sid, dist)
                break
        if near:
            d.download_log.append({"clip": key, "outcome": "near_duplicate",
                                   "reason": f"looks like the clip used in {near[0]} (hash distance {near[1]})"})
            if first_dup is None:
                first_dup = (cand_s, cpath, fp, near)
            continue
        d.download_log.append({"clip": key, "outcome": "ok", "reason": "downloaded and validated"})
        return cand_s, cpath, fp, None
    if first_dup is not None:
        d.download_log.append({"clip": first_dup[0].candidate.key, "outcome": "ok", "reason": "only near-duplicates remained"})
    return first_dup
