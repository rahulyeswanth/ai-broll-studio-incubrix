"""Presentation logic for the Streamlit frontend. No Streamlit import here so it can be unit-tested directly.

Everything that ends up inside raw HTML passes through ``esc`` (HTML-escaped) - long or hostile text (scene
descriptions, clip titles, error messages) can neither break the layout nor inject markup.
"""
from __future__ import annotations

import html
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

STAGES: Tuple[Tuple[str, float, float], ...] = (
    ("Narration", 0, 36), ("Footage matching", 36, 60), ("Downloading", 60, 76),
    ("Captions", 76, 80), ("Rendering", 80, 94), ("Validating", 94, 100.1),
)

STATUS_LABEL = {"matched": "Matched", "review": "Needs review", "unmatched": "No footage"}
STATUS_TONE = {"matched": "ok", "review": "warn", "unmatched": "bad"}
ERROR_FALLBACK = "The video could not be generated."


def esc(x: Any) -> str:
    return html.escape("" if x is None else str(x), quote=True)


def fmt_time(seconds: Any) -> str:
    try:
        s = float(seconds)
    except (TypeError, ValueError):
        return "-"
    m, sec = divmod(s, 60)
    return f"{int(m)}:{sec:04.1f}" if m else f"{sec:.1f}s"


def fmt_range(start: Any, end: Any) -> str:
    try:
        return f"{float(start):.1f}-{float(end):.1f}s"
    except (TypeError, ValueError):
        return "-"


def clamp(text: Any, limit: int) -> str:
    """Shorten with an ellipsis (the full text must stay reachable elsewhere, e.g. in a title attribute / details)."""
    s = " ".join(str(text or "").split())
    return s if len(s) <= limit else s[: max(limit - 1, 1)].rstrip() + "…"


def scene_status(scene: Dict[str, Any]) -> str:
    """matched | review | unmatched - derived from the timeline record, never guessed."""
    if scene.get("placeholder") or not scene.get("clip"):
        return "unmatched"
    if scene.get("needs_review") or scene.get("match_status") == "fallback_low_confidence":
        return "review"
    return "matched"


def counts(scenes: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    out = {"all": len(scenes), "matched": 0, "review": 0, "unmatched": 0}
    for s in scenes:
        out[scene_status(s)] += 1
    return out


def filter_scenes(scenes: Sequence[Dict[str, Any]], which: Optional[str]) -> List[Dict[str, Any]]:
    """``which`` may carry a count suffix as shown on the filter control, e.g. "Needs review (2)"."""
    name = (which or "all").lower().split(" (")[0].strip()
    if name == "all":
        return list(scenes)
    key = {"matched": "matched", "needs review": "review", "review": "review", "unmatched": "unmatched", "no footage": "unmatched"}.get(name)
    return [s for s in scenes if scene_status(s) == key] if key else []


def relevance_text(scene: Dict[str, Any]) -> Tuple[str, str]:
    """(label, small print). A label only exists when there is a selected clip; the number is shown as an uncalibrated
    relevance score, never as a percentage or 'confidence'."""
    rel = scene.get("relevance") or {}
    if scene_status(scene) == "unmatched" or rel.get("score") is None:
        return "None", "No clip selected"
    label = rel.get("label") or ("Medium" if rel.get("passed_threshold") else "Low")
    how = "frames checked" if rel.get("visual_verified") else "from title/tags"
    return label, f"Relevance score {rel['score']:.2f} · {how}"


def why_selected(scene: Dict[str, Any], limit: int = 170) -> str:
    st = scene_status(scene)
    reason = scene.get("selection_reason") or ""
    if st == "unmatched":
        return clamp(reason or "No acceptable clip was found for this scene.", limit)
    ev = (scene.get("relevance") or {}).get("evidence") or {}
    matched = ev.get("matched_terms") or {}
    if st == "matched" and matched:
        parts = [f"{k} '{v}'" for k, v in matched.items()]
        return clamp("Clip tags match " + ", ".join(parts) + ".", limit)
    return clamp(reason, limit)


def timeline_html(scenes: Sequence[Dict[str, Any]], selected: Optional[int] = None) -> str:
    """Proportional segment bar. ``selected`` is a 1-based scene index. Segment labels are short and ellipsised; the
    full scene text is the segment's title attribute. Fully self-contained (no overlap with following content)."""
    segs: List[str] = []
    total = 0.0
    spans: List[Tuple[Dict[str, Any], float]] = []
    for s in scenes:
        try:
            d = max(float(s.get("end", s.get("end_time", 0))) - float(s.get("start", s.get("start_time", 0))), 0.001)
        except (TypeError, ValueError):
            d = 0.001
        spans.append((s, d))
        total += d
    if not spans or total <= 0:
        return ""
    for i, (s, d) in enumerate(spans, 1):
        tone = STATUS_TONE.get(scene_status(s), "ok") if ("clip" in s or "match_status" in s or "placeholder" in s) else "ok"
        sel = " is-selected" if selected == i else ""
        text = s.get("narration_text") or s.get("text") or ""
        segs.append(f'<div class="tl-seg tone-{tone}{sel}" style="flex:{d:.3f} 1 0" title="Scene {i}: {esc(clamp(text, 160))}">'
                    f'<span class="tl-label">{i}</span><span class="tl-dur">{d:.1f}s</span></div>')
    return f'<div class="tl-bar" role="list" aria-label="Scene timeline">{"".join(segs)}</div>'


def stage_for(progress: float, step: str = "") -> Tuple[int, str]:
    """Index of the active stage and a short, non-technical status line."""
    p = max(0.0, min(100.0, float(progress or 0)))
    idx = next((i for i, (_, lo, hi) in enumerate(STAGES) if lo <= p < hi), len(STAGES) - 1)
    return idx, (step or STAGES[idx][0])


def scene_progress(step: str) -> Optional[Tuple[int, int]]:
    """Parse '... scene 3/5' from backend step text when present."""
    import re
    m = re.search(r"(\d+)\s*/\s*(\d+)", step or "")
    return (int(m.group(1)), int(m.group(2))) if m else None


def stages_html(progress: float, step: str, failed: bool = False) -> str:
    idx, _ = stage_for(progress, step)
    out = []
    for i, (name, _, _) in enumerate(STAGES):
        cls = "done" if i < idx else ("active" if i == idx else "")
        if failed and i == idx:
            cls = "failed"
        out.append(f'<div class="stg {cls}"><span class="stg-dot"></span><span class="stg-name">{esc(name)}</span></div>')
    return f'<div class="stages">{"".join(out)}</div>'


def elapsed_text(seconds: float) -> str:
    s = int(max(0, seconds))
    return f"{s // 60}:{s % 60:02d}"


def notice_html(level: str, text: str) -> str:
    icon = {"success": "✓", "info": "i", "warning": "!", "error": "×"}.get(level, "i")
    return f'<div class="notice n-{esc(level)}"><span class="n-ico">{icon}</span><span class="n-txt">{esc(text)}</span></div>'


def job_error_text(job: Dict[str, Any]) -> Tuple[str, str]:
    """(user-facing message, technical message) for a failed job."""
    err = job.get("error") or {}
    return err.get("user_message") or err.get("message") or ERROR_FALLBACK, err.get("message") or ""


def group_diagnostics(items: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group structured diagnostic items by severity, preserving order and removing exact duplicates."""
    seen = set()
    groups: Dict[str, List[Dict[str, Any]]] = {"error": [], "warning": [], "info": []}
    for it in items or []:
        k = (it.get("code"), it.get("scene_id"), it.get("message"))
        if k in seen:
            continue
        seen.add(k)
        groups.setdefault(it.get("severity", "info"), []).append(it)
    return groups


def badge_html(status: str) -> str:
    return f'<span class="badge tone-{STATUS_TONE.get(status, "ok")}">{esc(STATUS_LABEL.get(status, status))}</span>'
