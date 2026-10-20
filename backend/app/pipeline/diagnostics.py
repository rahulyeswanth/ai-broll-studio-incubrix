"""Turns raw pipeline facts into (a) 1-4 short status messages for the main screen and (b) a deduplicated, structured
list for the collapsible "Generation details" panel. Nothing is hidden: the raw ``warnings``/``fallback_events`` stay in
the manifest; this only changes presentation. Severity: info | warning | error."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

ERROR_MESSAGES = {
    "invalid_input": "The script or settings were not accepted. Check the input and try again.",
    "invalid_audio": "The uploaded audio could not be used. Check the file format, length and that it contains speech.",
    "tts_failed": "Narration could not be generated. The text-to-speech service may be unreachable; try again shortly.",
    "transcription_failed": "The uploaded audio could not be transcribed.",
    "provider_unavailable": "The footage or voice provider is not configured or is unreachable. Check the API key in backend/.env.",
    "footage_unavailable": "No suitable footage could be found for any scene. Try a more concrete script or another provider.",
    "download_failed": "Footage was found but none of it could be downloaded.",
    "render_failed": "The video could not be rendered. Details are in the diagnostics.",
    "output_validation_failed": "The video was rendered but failed quality validation. Details are in the diagnostics.",
    "interrupted": "The server restarted while this video was being generated. Please generate it again.",
    "internal_error": "An unexpected error occurred. Details are in the server log.",
}


def friendly_error(code: Optional[str], message: Optional[str] = None) -> str:
    return ERROR_MESSAGES.get((code or "").lower(), message or "The job failed.")


def _item(code: str, severity: str, title: str, message: str, scene_id: Optional[str] = None, **details: Any) -> Dict[str, Any]:
    return {"code": code, "severity": severity, "scene_id": scene_id, "title": title, "message": message,
            "details": {k: v for k, v in details.items() if v not in (None, [], {})}}


def build_diagnostics(decisions: Sequence[Any], *, raw_warnings: Iterable[str], events: Sequence[Dict[str, Any]],
                      verifier: Any, visual_performed: bool, slate_scene_ids: Sequence[str] = (),
                      looped_scene_ids: Sequence[str] = (), validation_warnings: Sequence[str] = ()) -> Dict[str, Any]:
    items: List[Dict[str, Any]] = []
    n_review = n_unmatched = n_alt = 0
    for d in decisions:
        sid = d.scene_id
        if d.status == "fallback_low_confidence":
            n_review += 1
            items.append(_item("needs_review", "warning", "Scene needs review", d.reason, sid,
                               score=round(d.selected.final_score, 3) if d.selected else None,
                               threshold=d.threshold, label=d.label))
        elif d.selected is None:
            n_unmatched += 1
            items.append(_item("no_footage", "warning", "No matching footage", d.reason, sid,
                               queries=[a.query for a in d.attempts], errors=[a.error for a in d.attempts if a.error]))
        if d.reused_clip:
            items.append(_item("clip_reused", "info", "Clip reused", "The same clip appears in more than one scene because no other acceptable candidate was found.", sid))
        for dl in getattr(d, "download_log", []):
            if dl.get("outcome") != "ok":
                items.append(_item("download_failed", "info", "A candidate could not be used", f"{dl.get('clip')}: {dl.get('reason')}", sid, outcome=dl.get("outcome")))
        if getattr(d, "used_alternate", False):
            n_alt += 1
            items.append(_item("alternate_clip", "info", "Alternate clip used", "The top-ranked clip could not be used (download failed or near-duplicate); the next-best acceptable candidate was used.", sid))
    for sid in looped_scene_ids:
        items.append(_item("clip_looped", "info", "Clip looped", "The clip is shorter than the scene and was looped.", sid))
    for sid in slate_scene_ids:
        items.append(_item("slate", "warning", "Placeholder shown", "No acceptable footage was found, so a plain dark placeholder is shown while the narration plays.", sid))

    name = getattr(verifier, "name", "none")
    reason = getattr(verifier, "unavailable_reason", None)
    if reason:
        items.append(_item("visual_unavailable", "info", "Visual verification unavailable", reason))
    if not visual_performed:
        items.append(_item("metadata_only", "info", "Not verified visually",
                           "Relevance was estimated from provider titles/tags and scene analysis. No vision model inspected the frames, "
                           "so the action shown in a clip is not confirmed."))
    items.append(_item("licence_basis", "info", "Licence information", "Commercial-use status reflects the provider's published policy, not a per-clip legal check. Verify before publishing."))
    for w in validation_warnings:
        items.append(_item("output_check", "info", "Output check", w))

    # raw warnings not already represented (e.g. caption timing approximations): keep them, once each
    seen_msgs = {i["message"] for i in items}
    for w in dict.fromkeys(raw_warnings):
        if w in seen_msgs or any(w in (i["message"] or "") or (i["message"] or "") in w for i in items if i["message"]):
            continue
        low = w.lower()
        if "title/tags only" in low or "no vision-language" in low or "licence" in low or "may not accurately" in low or "no footage for scene" in low:
            continue   # represented by the structured items above
        items.append(_item("note", "info", "Note", w))

    uniq: List[Dict[str, Any]] = []
    keys = set()
    for it in items:
        k = (it["code"], it["scene_id"], it["message"])
        if k not in keys:
            keys.add(k)
            uniq.append(it)

    summary: List[Dict[str, str]] = []
    if n_unmatched:
        summary.append({"level": "warning", "text": f"{n_unmatched} scene{'s' if n_unmatched != 1 else ''} ha{'ve' if n_unmatched != 1 else 's'} no matching footage."})
    if n_review:
        summary.append({"level": "warning", "text": f"{n_review} scene{'s' if n_review != 1 else ''} need{'' if n_review != 1 else 's'} review."})
    if n_alt:
        summary.append({"level": "info", "text": "One scene uses an alternate clip." if n_alt == 1 else f"{n_alt} scenes use alternate clips."})
    if not visual_performed:
        summary.append({"level": "info", "text": "Some footage could not be verified visually."})
    if not (n_unmatched or n_review):
        summary.insert(0, {"level": "success", "text": "Video generated successfully."})
    return {"summary": summary, "items": uniq,
            "counts": {"needs_review": n_review, "unmatched": n_unmatched, "alternate_clips": n_alt}}
