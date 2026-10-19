"""Candidate filtering and scoring.

Final score = w_visual*visual + w_text*text + w_tech*technical, where the visual term is only present when
a real visual verifier (e.g. CLIP) actually looked at frames. Without it the weight is redistributed to the
text and technical terms and the method is reported as ``metadata_text_fallback`` -- never as visual AI.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from urllib.parse import urlparse

from .config import PipelineSettings
from .models import Candidate, SceneAnalysis, ScoredCandidate

_WORD = re.compile(r"[a-z]+")
TARGET_RATIO = {"16:9": 16 / 9, "9:16": 9 / 16, "1:1": 1.0}


def _stem(w: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


def candidate_terms(c: Candidate) -> Set[str]:
    text = " ".join([c.title or ""] + list(c.tags or []))
    return {_stem(w) for w in _WORD.findall(text.lower()) if len(w) > 2}


# --------------------------------------------------------------------------- filtering
@dataclass
class FilterResult:
    kept: List[Candidate]
    rejected: List[Tuple[str, str]]  # (candidate key, reason)


def filter_candidates(cands: Iterable[Candidate], settings: PipelineSettings, *, seen_keys: Optional[Set[str]] = None,
                      require_commercial: bool = True) -> FilterResult:
    """Remove unusable, duplicate, inaccessible and policy-incompatible candidates, with a reason for each."""
    kept: List[Candidate] = []
    rejected: List[Tuple[str, str]] = []
    keys: Set[str] = set(seen_keys or ())
    urls: Set[str] = set()
    for c in cands:
        if not c.clip_id or not c.download_url:
            rejected.append((c.key, "missing id or download URL"))
            continue
        if c.key in keys or c.download_url in urls:
            rejected.append((c.key, "duplicate candidate"))
            continue
        u = urlparse(c.download_url.strip())
        if u.scheme not in ("http", "https") or not u.netloc or " " in c.download_url.strip():
            rejected.append((c.key, "invalid download URL"))
            continue
        if settings.require_https and u.scheme != "https":
            rejected.append((c.key, "download URL is not HTTPS"))
            continue
        host = (u.hostname or "").lower()
        if settings.allowed_download_hosts and host not in settings.allowed_download_hosts and settings.require_https:
            rejected.append((c.key, f"host '{host}' is not on the download allow-list"))
            continue
        if c.duration and c.duration < settings.min_clip_seconds:
            rejected.append((c.key, f"clip too short ({c.duration:.1f}s)"))
            continue
        if c.width and c.height and min(c.width, c.height) < 360:
            rejected.append((c.key, f"resolution too low ({c.width}x{c.height})"))
            continue
        if require_commercial and c.commercial_use is False:
            rejected.append((c.key, "provider licence metadata says commercial use is NOT allowed"))
            continue
        if c.ai_generated is True:
            rejected.append((c.key, "flagged as AI-generated media (excluded by policy)"))
            continue
        keys.add(c.key)
        urls.add(c.download_url)
        kept.append(c)
    return FilterResult(kept, rejected)


# --------------------------------------------------------------------------- evidence-based text relevance
PERSON_WORDS = {"patient", "patients", "student", "students", "people", "person", "man", "woman", "men", "women", "child",
                "children", "kid", "kids", "crowd", "team", "colleague", "colleagues", "customer", "customers", "worker",
                "workers", "audience", "family", "group", "tourist", "tourists", "visitor", "visitors"}
_SW = {"a", "an", "the", "of", "in", "on", "at", "to", "for", "and", "with", "by", "from", "into"}


def _toks(text: str) -> List[str]:
    return [_stem(w) for w in _WORD.findall((text or "").lower()) if w not in _SW and len(w) >= 2]


def _candidate_token_list(c: Candidate) -> List[str]:
    out: List[str] = []
    for chunk in [c.title or ""] + list(c.tags or []):
        out += _toks(chunk)
        out.append("|")   # tag boundary: phrases never span two tags
    return out


def phrase_match(term: str, ctoks: Sequence[str]) -> float:
    """0..1: 1.0 contiguous phrase (or single word) present; 0.75 all words present but separated; partial words <= 0.5."""
    words = _toks(term)
    if not words:
        return 0.0
    cset = set(ctoks)
    present = [w for w in words if w in cset]
    if len(present) == len(words):
        if len(words) == 1:
            return 1.0
        n = len(words)
        for i in range(len(ctoks) - n + 1):
            if list(ctoks[i:i + n]) == words:
                return 1.0
        return 0.75
    return 0.5 * len(present) / len(words) if len(words) > 1 else 0.0


def _best(terms: Iterable[str], ctoks: Sequence[str]) -> Tuple[float, Optional[str]]:
    best, which = 0.0, None
    for term in terms:
        s = phrase_match(term, ctoks)
        if s > best:
            best, which = s, term
    return best, which


HIT = 0.75   # coverage at/above which an element counts as "evidenced"


@dataclass
class Evidence:
    scores: Dict[str, float] = field(default_factory=dict)         # subject/action/object/setting -> 0..1 (present elements only)
    matched: Dict[str, str] = field(default_factory=dict)          # element -> the term that matched
    weights: Dict[str, float] = field(default_factory=dict)
    foil_hits: List[str] = field(default_factory=list)
    negative_hits: List[str] = field(default_factory=list)
    score: float = 0.0
    action_evidenced: bool = False
    object_evidenced: bool = False
    gated: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"elements": {k: round(v, 3) for k, v in self.scores.items()}, "matched_terms": self.matched,
                "weights": self.weights, "foil_hits": self.foil_hits, "negative_hits": self.negative_hits,
                "action_evidenced": self.action_evidenced, "object_evidenced": self.object_evidenced}


def has_evidence_model(a: SceneAnalysis) -> bool:
    return bool(a.subject_terms or a.action_terms or a.object_terms or a.target_terms or a.setting_terms)


def evidence_relevance(c: Candidate, a: SceneAnalysis, settings: PipelineSettings) -> Evidence:
    """Score how well title/tags support each element of the scene, separately.

    Elements: subject (+ role synonyms), action, key object, setting. Only elements present in the scene take part;
    weights (``evidence_weight_*``) are renormalised over those. All element scores are 0..1 phrase coverages, so the
    result is a documented *relevance score*, not a calibrated probability.
    """
    ctoks = _candidate_token_list(c)
    ev = Evidence()
    inferred_setting = any("inferred" in n for n in a.notes)

    person_objs = [o for o in a.target_terms if all(w in PERSON_WORDS for w in _WORD.findall(o.lower()))]
    target_objs = [o for o in a.target_terms if o not in person_objs]
    strong_objs = list(a.object_terms) + target_objs          # instruments + targets both feed the score ...
    subj_terms = list(a.subject_terms) + [o for o in person_objs]

    groups = (("subject", subj_terms, settings.evidence_weight_subject),
              ("action", a.action_terms, settings.evidence_weight_action),
              ("object", strong_objs, settings.evidence_weight_object),
              ("setting", a.setting_terms, settings.evidence_weight_setting * (0.5 if inferred_setting else 1.0)))
    for name, terms, w in groups:
        if not terms:
            continue
        s, which = _best(terms, ctoks)
        ev.scores[name], ev.weights[name] = s, w
        if which and s > 0:
            ev.matched[name] = which
    total_w = sum(ev.weights.values())
    ev.score = sum(ev.scores[k] * ev.weights[k] for k in ev.scores) / total_w if total_w else 0.0
    ev.action_evidenced = ev.scores.get("action", 0.0) >= HIT
    # ... but only instruments (cuff, beaker, camera) can *prove* the action; a passive target (the tower) cannot
    ev.object_evidenced = bool(a.object_terms) and _best(a.object_terms, ctoks)[0] >= HIT

    for term in a.foil_terms:
        if phrase_match(term, ctoks) >= 1.0:
            ev.foil_hits.append(term)
    asked_for = " ".join(a.term_weights)
    for term in dict.fromkeys(list(a.avoid_terms) + list(a.negative_terms)):
        if term.lower() in asked_for:
            continue                         # the narration itself asks for it (e.g. a scene about an illustration)
        if phrase_match(term, ctoks) >= 1.0:
            ev.negative_hits.append(term)
    return ev


def apply_gates(ev: Evidence, a: SceneAnalysis) -> Optional[str]:
    """Reasons a clip must not be accepted on metadata alone (None = may pass).

    Gates need a known activity frame: only then do we know what proof of the action looks like. Scenes without one
    (abstract or unusual narration) are scored by weighted coverage alone."""
    if not a.gated_frame:
        return None
    if ev.foil_hits and not ev.action_evidenced:
        return f"title/tags indicate a different activity ({', '.join(ev.foil_hits[:3])}) and nothing indicates the scene's action"
    if not (ev.action_evidenced or ev.object_evidenced):
        who = ev.matched.get("subject") or ev.matched.get("setting") or ev.matched.get("object")
        kind = "subject" if ev.matched.get("subject") else ("setting" if ev.matched.get("setting") else "scene element")
        what = (a.action_terms[0] if a.action_terms else a.action) or "the described action"
        head = f"matches the {kind} ('{who}')" if who else "matches no scene element"
        return f"{head} but nothing in its title/tags shows {what}"
    return None


# --------------------------------------------------------------------------- scoring
def text_relevance(c: Candidate, analysis: SceneAnalysis) -> Tuple[float, List[str], List[str]]:
    """Weighted coverage of the scene's terms by the candidate's title/tags (0..1) minus avoid-term penalty."""
    weights = analysis.term_weights or {t: 1.0 for t in analysis.search_terms}
    if not weights:
        return 0.0, [], []
    cterms = candidate_terms(c)
    total = sum(weights.values())
    hit = 0.0
    matched: List[str] = []
    for term, w in weights.items():
        if _stem(term) in cterms:
            hit += w
            matched.append(term)
    score = hit / total if total else 0.0
    penalties: List[str] = []
    for av in analysis.avoid_terms:
        if all(_stem(p) in cterms for p in _WORD.findall(av.lower())):
            # an avoid term the narration itself asks for is not a penalty
            if av.lower() in " ".join(analysis.term_weights).lower():
                continue
            score -= 0.25
            penalties.append(f"avoid-term '{av}' present in clip tags/title")
    return max(0.0, min(1.0, score)), matched, penalties


def technical_fit(c: Candidate, aspect: str, scene_seconds: float) -> Tuple[float, List[str]]:
    notes: List[str] = []
    score = 0.0
    if c.width and c.height:
        ratio = c.width / c.height
        target = TARGET_RATIO.get(aspect, 16 / 9)
        diff = abs(ratio - target) / target
        s = max(0.0, 1.0 - diff)
        score += 0.5 * s
        if diff > 0.25:
            notes.append(f"orientation {c.width}x{c.height} differs from {aspect}; will be cropped")
    else:
        score += 0.25
    if c.duration:
        score += 0.3 * min(1.0, c.duration / max(scene_seconds, 0.1))
        if c.duration < scene_seconds:
            notes.append(f"clip ({c.duration:.1f}s) shorter than scene ({scene_seconds:.1f}s); will loop")
    else:
        score += 0.15
    if c.height:
        score += 0.2 * min(1.0, c.height / 1080) if c.width and c.width >= c.height else 0.2 * min(1.0, c.width / 1080 if c.width else 0.5)
    else:
        score += 0.1
    return min(1.0, score), notes


def relevance_label(sc: ScoredCandidate, settings: PipelineSettings) -> str:
    """High / Medium / Low with documented thresholds (this is a ranking aid, not a probability):
    High   = passed AND score >= high_confidence AND (action/object evidence in metadata, or frames inspected)
    Medium = passed the match threshold
    Low    = below the threshold (shown as 'needs review' or unmatched)"""
    if not sc.passed:
        return "Low"
    ev = sc.evidence or {}
    evidenced = ev.get("action_evidenced") or ev.get("object_evidenced") or sc.visual_verified
    return "High" if sc.final_score >= settings.high_confidence and evidenced else "Medium"


def score_candidates(cands: Sequence[Candidate], analysis: SceneAnalysis, aspect: str, scene_seconds: float,
                     settings: PipelineSettings, visual_scores: Optional[Dict[str, float]] = None,
                     visual_details: Optional[Dict[str, Dict[str, Any]]] = None) -> List[ScoredCandidate]:
    visual_scores = visual_scores or {}
    visual_details = visual_details or {}
    out: List[ScoredCandidate] = []
    use_evidence = has_evidence_model(analysis)
    cap_value = max(0.0, settings.match_threshold - 0.02)
    for c in cands:
        ev: Optional[Evidence] = None
        if use_evidence:
            ev = evidence_relevance(c, analysis, settings)
            t = ev.score
            matched = [f"{k}: {v}" for k, v in ev.matched.items()]
            penalties: List[str] = []
            if ev.foil_hits and not ev.action_evidenced:
                t = max(0.0, t - settings.foil_penalty)
                penalties.append("related-but-different activity in title/tags: " + ", ".join(ev.foil_hits[:3]))
            for n in ev.negative_hits:
                if n in ev.foil_hits:      # already penalised as a foil: do not count the same word twice
                    continue
                t = max(0.0, t - 0.15)
                penalties.append(f"avoid-term '{n}' present in clip tags/title")
        else:
            t, matched, penalties = text_relevance(c, analysis)
        if analysis.sense_good_terms or analysis.sense_bad_terms:
            words = set(re.findall(r"[a-z]+", " ".join([c.title or ""] + list(c.tags or [])).lower()))
            good, bad = sorted(words & set(analysis.sense_good_terms)), sorted(words & set(analysis.sense_bad_terms))
            if bad and not good:
                t = max(0.0, t - settings.sense_mismatch_penalty)
                penalties.append("clip metadata indicates a different meaning of an ambiguous word: " + ", ".join(bad[:3]))
            elif good and t > 0:
                t = min(1.0, t + settings.sense_match_bonus)
                matched = list(matched) + ["word-sense confirmed: " + ", ".join(good[:3])]
        tech, tech_notes = technical_fit(c, aspect, scene_seconds)
        v = visual_scores.get(c.key)
        vd = dict(visual_details.get(c.key) or {})
        if v is not None:
            wsum = settings.weight_visual + settings.weight_text + settings.weight_technical
            final = (settings.weight_visual * v + settings.weight_text * t + settings.weight_technical * tech) / wsum
            method = "visual_clip+metadata_text"
            verified = True
        else:
            # no visual evidence: re-normalise; technical fit may not rescue an irrelevant clip
            wsum = settings.weight_text + settings.weight_technical
            final = (settings.weight_text * t + settings.weight_technical * tech * (0.5 + 0.5 * t)) / wsum
            method = "metadata_text_fallback"
            verified = False
        final = max(0.0, min(1.0, final))

        gated: Optional[str] = None
        if ev is not None:
            if v is None:
                gated = apply_gates(ev, analysis)          # metadata alone cannot prove the action
            elif vd.get("foil_leading"):
                gated = "frame analysis found a different activity more likely than the described one"
            elif v < 0.30:
                gated = "frame analysis found little visual agreement with the scene"
        elif v is not None and vd.get("foil_leading"):
            gated = "frame analysis found a different activity more likely than the described one"
        if gated and final > cap_value:
            final = cap_value

        reasons = [("evidence - " + ", ".join(matched)) if matched else "no scene terms found in clip title/tags"]
        if gated:
            reasons.append("NOT accepted automatically: " + gated)
        reasons += tech_notes
        sc = ScoredCandidate(c, t, v, tech, final, method, reasons, False, verified, penalties,
                             evidence=ev.to_dict() if ev else {}, gated=gated, visual_detail=vd)
        sc.passed = (final >= settings.match_threshold) and gated is None
        sc.label = relevance_label(sc, settings)
        out.append(sc)
    out.sort(key=lambda s: (-s.final_score, s.candidate.key))
    return out


@dataclass
class Selection:
    chosen: Optional[ScoredCandidate]
    status: str          # matched | fallback_low_confidence | unmatched
    reason: str
    reused: bool = False


def select_best(scored: Sequence[ScoredCandidate], used_keys: Set[str], settings: PipelineSettings) -> Selection:
    """Pick the strongest acceptable candidate, avoiding clips already used in this video."""
    passing = [s for s in scored if s.passed]
    fresh = [s for s in passing if s.candidate.key not in used_keys]
    if fresh:
        top = fresh[0]
        return Selection(top, "matched", f"highest-scoring unused candidate ({top.final_score:.2f} >= threshold {settings.match_threshold:.2f}; {top.method})")
    if passing and settings.allow_clip_reuse:
        top = passing[0]
        return Selection(top, "matched", f"only already-used clips passed the threshold; reused with disclosure ({top.final_score:.2f})", reused=True)
    return Selection(None, "unmatched", "no candidate reached the relevance threshold")
