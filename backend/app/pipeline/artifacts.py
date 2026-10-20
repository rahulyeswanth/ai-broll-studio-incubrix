"""timeline.json / manifest.json builders and validators."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .licensing import licence_record, licence_warnings

TIMELINE_SCHEMA_VERSION = "1.0"
MANIFEST_SCHEMA_VERSION = "1.0"

TIMELINE_SCENE_REQUIRED = ["scene_id", "index", "start", "end", "duration", "narration_text", "visual_description",
                           "clip", "license", "relevance", "selection_reason", "match_status", "warnings"]
MANIFEST_REQUIRED = ["schema_version", "job_id", "input_type", "input", "outputs", "requested_aspect_ratio",
                     "providers", "matching", "captions", "processing", "processing_durations_seconds",
                     "cost_estimate", "licensing", "warnings", "fallback_events", "validation"]


def build_timeline(job_id: str, aspect: str, width: int, height: int, fps: int, audio_file: Optional[str],
                   audio_source: str, total_duration: float, scenes, decisions, clip_files: Dict[str, str],
                   segment_info: Dict[str, Dict[str, Any]], warnings: List[str], *, thumbnails: Optional[Dict[str, str]] = None,
                   slate_ids: Optional[List[str]] = None, settings: Any = None,
                   provider_search: Any = None) -> Dict[str, Any]:
    out_scenes: List[Dict[str, Any]] = []
    thumbnails, slate_ids = thumbnails or {}, slate_ids or []
    for sc, dec in zip(scenes, decisions):
        sel = dec.selected
        cand = sel.candidate if sel else None
        lic = licence_record(cand.provider) if cand else {}
        seg = segment_info.get(sc.scene_id, {})
        clip = None
        if cand:
            clip = {
                "provider": cand.provider, "clip_id": cand.clip_id, "title": cand.title, "tags": cand.tags,
                "source_page_url": cand.page_url, "download_url": cand.download_url,
                "local_file": clip_files.get(cand.key), "duration_seconds": cand.duration,
                "width": cand.width, "height": cand.height, "creator": cand.creator, "creator_url": cand.creator_url,
                "ai_generated": cand.ai_generated, "retrieved_at": cand.retrieved_at,
                "looped": seg.get("looped"), "framing_method": seg.get("framing_method"),
                "color_normalisation": seg.get("color_normalisation"),
                "reused_from_earlier_scene": dec.reused_clip, "thumbnail": thumbnails.get(sc.scene_id),
            }
        out_scenes.append({
            "scene_id": sc.scene_id, "index": sc.index, "start": round(sc.start, 3), "end": round(sc.end, 3),
            "duration": round(sc.end - sc.start, 3), "narration_text": sc.text, "timing_method": sc.timing_method,
            "visual_description": dec.analysis.visual_description,
            "visual_analysis": {"subject": dec.analysis.subject, "action": dec.analysis.action,
                                "setting": dec.analysis.setting, "objects": dec.analysis.objects,
                                "avoid_terms": dec.analysis.avoid_terms, "method": dec.analysis.method,
                                "notes": dec.analysis.notes, "frames": dec.analysis.frame_ids,
                                "scene_type": dec.analysis.scene_type, "action_evidence_terms": dec.analysis.action_terms[:12],
                                "object_evidence_terms": dec.analysis.object_terms[:12], "foil_terms": dec.analysis.foil_terms[:10]},
            "search_queries": [a.query for a in dec.attempts], "queries_detail": [a.to_dict() for a in dec.attempts],
            "candidate_count": dec.candidate_count, "retry_attempts": dec.retry_attempts,
            "clip": clip, "license": lic,
            "attribution": ({"required": lic.get("attribution_required"),
                             "text": f"Video by {cand.creator} via {cand.provider.capitalize()}" if cand and cand.creator else None}
                            if cand else None),
            "relevance": {"score": round(sel.final_score, 3) if sel else None,
                          "scoring_method": dec.scoring_method if sel else None,
                          "visual_verified": bool(sel and sel.visual_verified),
                          "threshold": dec.threshold, "passed_threshold": dec.passed_threshold,
                          "text_score": round(sel.text_score, 3) if sel else None,
                          "visual_score": (round(sel.visual_score, 3) if sel and sel.visual_score is not None else None),
                          "label": sel.label if sel else None, "evidence": sel.evidence if sel else None,
                          "gated": sel.gated if sel else None, "visual_detail": sel.visual_detail if sel else None},
            "needs_review": dec.status == "fallback_low_confidence", "placeholder": sc.scene_id in slate_ids,
            "download_attempts": dec.download_log, "used_alternate": dec.used_alternate,
            "selection_reason": dec.reason, "match_status": dec.status,
            "provider_selection": (provider_search.scene_summary([a.query for a in dec.attempts], cand.provider if cand else None)
                                   if provider_search is not None else None),
            "alternatives": [s.to_dict() for s in dec.ranked[1:4]],
            "warnings": list(dec.warnings),
        })
    return {"schema_version": TIMELINE_SCHEMA_VERSION, "job_id": job_id, "aspect_ratio": aspect,
            "width": width, "height": height, "fps": fps,
            "audio": {"file": audio_file, "source": audio_source}, "total_duration": round(total_duration, 3),
            "scenes": out_scenes, "warnings": warnings}


def validate_timeline(tl: Dict[str, Any], audio_duration: Optional[float] = None, tol: float = 0.06) -> List[str]:
    problems: List[str] = []
    scenes = tl.get("scenes") or []
    if not scenes:
        return ["timeline has no scenes"]
    prev_end = 0.0
    for i, s in enumerate(scenes):
        miss = [k for k in TIMELINE_SCENE_REQUIRED if k not in s]
        if miss:
            problems.append(f"scene {i + 1} missing fields: {miss}")
            continue
        if i == 0 and abs(s["start"]) > tol:
            problems.append("first scene does not start at 0")
        if abs(s["start"] - prev_end) > tol:
            problems.append(f"scene {s['scene_id']} starts at {s['start']} but previous ended at {prev_end}")
        if s["end"] <= s["start"]:
            problems.append(f"scene {s['scene_id']} has non-positive duration")
        prev_end = s["end"]
    if audio_duration is not None and abs(prev_end - audio_duration) > tol:
        problems.append(f"timeline ends at {prev_end}s but audio is {audio_duration}s")
    return problems


def cost_estimate(tts_name: Optional[str], asr_name: Optional[str], footage_names: List[str]) -> Dict[str, Any]:
    free_tts = {None, "edge-tts", "mock"}
    free_asr = {None, "faster-whisper", "mock"}
    free_footage = {"pixabay", "pexels", "mock"}
    if tts_name in free_tts and asr_name in free_asr and all(f in free_footage for f in footage_names):
        return {"amount_usd": 0.0, "currency": "USD",
                "basis": "All configured providers are free to call (Edge TTS endpoint, local Whisper, free-tier footage APIs). "
                         "Local compute, bandwidth and any provider rate limits are not priced."}
    return {"amount_usd": None, "currency": "USD", "basis": "unknown: a configured provider has no known pricing in this application"}


def build_manifest(*, job_id: str, input_type: str, input_meta: Dict[str, Any], aspect: str, outputs: Dict[str, Any],
                   providers: Dict[str, Any], matching: Dict[str, Any], captions: Dict[str, Any],
                   processing: Dict[str, Any], durations: Dict[str, float], timeline: Dict[str, Any],
                   warnings: List[str], events: List[Dict[str, Any]], validation: Dict[str, Any],
                   created_at: str, status: str) -> Dict[str, Any]:
    selected = [s for s in timeline["scenes"] if s.get("clip")]
    lic_items = [{"scene_id": s["scene_id"], "provider": s["clip"]["provider"], "clip_id": s["clip"]["clip_id"],
                  "license": s["license"], "creator": s["clip"]["creator"], "source_page_url": s["clip"]["source_page_url"],
                  "ai_generated": s["clip"]["ai_generated"], "retrieved_at": s["clip"]["retrieved_at"]} for s in selected]
    lic_warn = licence_warnings([{"license": i["license"]} for i in lic_items])
    all_warnings = list(dict.fromkeys(list(warnings) + lic_warn))
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION, "job_id": job_id, "status": status, "created_at": created_at,
        "input_type": input_type, "input": input_meta, "requested_aspect_ratio": aspect, "outputs": outputs,
        "providers": providers, "matching": matching, "captions": captions, "processing": processing,
        "processing_durations_seconds": durations,
        "cost_estimate": cost_estimate(providers.get("tts", {}).get("name"), providers.get("transcription", {}).get("name"),
                                       sorted({i["provider"] for i in lic_items}) or [providers.get("footage", {}).get("name", "unknown")]),
        "licensing": {"clips": lic_items,
                      "all_commercial_use_per_provider_policy": bool(lic_items) and all(i["license"].get("commercial_use") for i in lic_items),
                      "attribution_required_for_any": any(i["license"].get("attribution_required") for i in lic_items) or None,
                      "disclaimer": "Licence data reflects provider-level published policy; per-clip rights are not verified."},
        "warnings": all_warnings, "fallback_events": events, "validation": validation,
    }


def validate_manifest(m: Dict[str, Any]) -> List[str]:
    return [f"manifest missing '{k}'" for k in MANIFEST_REQUIRED if k not in m]
