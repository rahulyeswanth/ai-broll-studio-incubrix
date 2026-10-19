"""Semantic B-roll matching: analyse scene -> query -> retrieve -> filter -> score -> select -> retry.

Every step is recorded in a ``SceneDecision`` which feeds timeline.json / manifest.json.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Set

from .config import PipelineSettings
from .models import Candidate, Scene, SceneAnalysis, ScoredCandidate
from .provider_adapter import candidate_from_asset
from .ranking import filter_candidates, score_candidates, select_best
from .scene_analysis import RuleBasedSceneAnalyzer, SceneAnalyzer
from .visual import NullVisualVerifier, VisualVerifier

log = logging.getLogger(__name__)

SearchFn = Callable[[str, str, int], Awaitable[Sequence[Any]]]
PreviewFn = Callable[[Candidate, Path], Optional[Path]]


@dataclass
class QueryAttempt:
    query: str
    returned: int = 0
    kept: int = 0
    rejected: List[str] = field(default_factory=list)
    best_score: Optional[float] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"query": self.query, "returned": self.returned, "kept": self.kept, "rejected": self.rejected,
                "best_score": None if self.best_score is None else round(self.best_score, 3), "error": self.error}


@dataclass
class SceneDecision:
    scene_id: str
    narration: str
    analysis: SceneAnalysis
    attempts: List[QueryAttempt] = field(default_factory=list)
    candidate_count: int = 0
    ranked: List[ScoredCandidate] = field(default_factory=list)
    selected: Optional[ScoredCandidate] = None
    # matched | fallback_low_confidence (== "needs review": usable but not proven) | unmatched | provider_error | no_candidates
    status: str = "unmatched"
    reason: str = ""
    reused_clip: bool = False
    warnings: List[str] = field(default_factory=list)
    threshold: float = 0.0
    download_log: List[Dict[str, Any]] = field(default_factory=list)   # every clip actually tried, with its outcome
    used_alternate: bool = False
    scoring_method: str = "metadata_text_fallback"
    visual_verifier: str = "none"

    @property
    def passed_threshold(self) -> bool:
        return bool(self.selected and self.selected.passed)

    @property
    def needs_review(self) -> bool:
        return self.status == "fallback_low_confidence"

    @property
    def label(self) -> str:
        return self.selected.label if self.selected else "None"

    @property
    def retry_attempts(self) -> int:
        return max(0, len(self.attempts) - 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene_id": self.scene_id, "narration": self.narration, "analysis": self.analysis.to_dict(),
            "queries_tried": [a.to_dict() for a in self.attempts], "candidate_count": self.candidate_count,
            "status": self.status, "needs_review": self.needs_review, "label": self.label,
            "reason": self.reason, "passed_threshold": self.passed_threshold,
            "threshold": self.threshold, "scoring_method": self.scoring_method, "visual_verifier": self.visual_verifier,
            "download_attempts": self.download_log, "used_alternate": self.used_alternate,
            "retry_attempts": self.retry_attempts, "reused_clip": self.reused_clip, "warnings": self.warnings,
            "top_candidates": [s.to_dict() for s in self.ranked[:5]],
            "selected": self.selected.to_dict() if self.selected else None,
        }


_AVOID_EXPANSIONS = {
    "face": {"face", "faces", "portrait", "selfie", "man", "woman", "boy", "girl", "person", "people", "child", "children",
             "businessman", "businesswoman", "couple", "family", "smiling", "crowd"},
    "people": {"people", "person", "man", "woman", "crowd", "men", "women", "human", "humans", "boy", "girl", "child", "children"},
    "alcohol": {"alcohol", "beer", "wine", "whiskey", "vodka", "cocktail", "liquor", "bar", "champagne", "drunk", "brewery"},
    "violence": {"violence", "violent", "fight", "fighting", "gun", "weapon", "blood", "war", "explosion"},
    "smoking": {"smoking", "cigarette", "cigar", "tobacco", "vape", "vaping"},
}


def expand_avoid(terms: Sequence[str]) -> Set[str]:
    """Turn free-text avoid rules ("people's faces", "alcohol") into a set of lower-case metadata words to reject.
    Metadata (title/tags) only: this cannot see frames, so it reduces but does not eliminate the risk."""
    out: Set[str] = set()
    for t in terms or []:
        for w in re.findall(r"[a-z]+", (t or "").lower().replace("'", " ")):
            if w in ("s", "the", "of", "and", "or", "any", "no", "with", "in"):
                continue
            stem = w[:-1] if w.endswith("s") and len(w) > 3 else w
            out |= _AVOID_EXPANSIONS.get(stem, set()) | _AVOID_EXPANSIONS.get(w, set()) | {w, stem}
    return out


def violates_avoid(c: Candidate, avoid_words: Set[str]) -> Optional[str]:
    if not avoid_words:
        return None
    words = set(re.findall(r"[a-z]+", " ".join([c.title or ""] + list(c.tags or [])).lower()))
    hit = sorted(words & avoid_words)
    return f"matches avoid rule ({', '.join(hit[:3])})" if hit else None


class SemanticMatcher:
    def __init__(self, search: SearchFn, settings: PipelineSettings, *, verifier: Optional[VisualVerifier] = None,
                 analyzer: Optional[SceneAnalyzer] = None, fetch_preview: Optional[PreviewFn] = None,
                 to_candidate: Callable[[Any], Candidate] = candidate_from_asset, avoid: Sequence[str] = ()):
        self.avoid_words = expand_avoid(avoid)
        self.search, self.settings = search, settings
        self.verifier = verifier or NullVisualVerifier()
        self.analyzer = analyzer or RuleBasedSceneAnalyzer()
        self.fetch_preview = fetch_preview
        self.to_candidate = to_candidate

    async def match_scene(self, scene: Scene, aspect: str, used_keys: Set[str], visual_hint: Optional[str] = None) -> SceneDecision:
        s = self.settings
        analysis = self.analyzer.analyze(scene.text, visual_hint)
        dec = SceneDecision(scene.scene_id, scene.text, analysis, threshold=s.match_threshold,
                            visual_verifier=self.verifier.name if self.verifier.performs_visual_analysis else "none")
        pool: Dict[str, ScoredCandidate] = {}
        seen_in_scene: Set[str] = set()
        errors = 0
        scene_seconds = max(scene.duration, 0.1)

        tried_queries: Set[str] = set()
        for query in analysis.queries:
            norm_q = " ".join(query.lower().split())
            if norm_q in tried_queries:
                continue                      # never spend a provider call on an identical query
            if len(tried_queries) >= s.max_query_attempts:
                break
            tried_queries.add(norm_q)
            att = QueryAttempt(query)
            dec.attempts.append(att)
            try:
                assets = await self.search(query, aspect, s.candidates_per_query)
            except Exception as exc:  # provider/network failure must not crash the job
                att.error = f"{type(exc).__name__}: {exc}"
                errors += 1
                log.warning("Footage search failed for %r: %s", query, exc)
                continue
            att.returned = len(assets)
            cands = []
            for a in assets:
                try:
                    cands.append(self.to_candidate(a))
                except Exception as exc:  # malformed asset
                    att.rejected.append(f"unparseable asset: {exc}")
            if self.avoid_words:
                kept_c = []
                for c in cands:
                    why = violates_avoid(c, self.avoid_words)
                    if why:
                        att.rejected.append(f"{c.key}: {why}")
                    else:
                        kept_c.append(c)
                cands = kept_c
            fr = filter_candidates(cands, s, seen_keys=seen_in_scene)
            att.rejected += [f"{k}: {why}" for k, why in fr.rejected]
            att.kept = len(fr.kept)
            if not fr.kept:
                continue
            visual: Dict[str, float] = {}
            if self.verifier.performs_visual_analysis and self.fetch_preview:
                pre = score_candidates(fr.kept, analysis, aspect, scene_seconds, s)
                shortlist = [x.candidate for x in pre[:5]]
                try:
                    visual = self.verifier.score(analysis, shortlist, self.fetch_preview)
                except Exception as exc:  # noqa: BLE001
                    dec.warnings.append(f"Visual verifier failed ({exc}); metadata fallback used for this query.")
            scored = score_candidates(fr.kept, analysis, aspect, scene_seconds, s, visual,
                                      getattr(self.verifier, "details", None))
            for sc in scored:
                seen_in_scene.add(sc.candidate.key)
                pool[sc.candidate.key] = sc
            att.best_score = scored[0].final_score if scored else None
            ranked_now = sorted(pool.values(), key=lambda x: (-x.final_score, x.candidate.key))
            sel = select_best(ranked_now, used_keys, s)
            if sel.chosen is not None and not sel.reused and sel.chosen.final_score >= s.early_stop_score:
                break   # a fresh, clearly good clip was found; weaker/reused results keep trying alternative phrasings

        dec.ranked = sorted(pool.values(), key=lambda x: (-x.final_score, x.candidate.key))
        dec.candidate_count = len(pool)
        if dec.ranked:
            dec.scoring_method = dec.ranked[0].method
        sel = select_best(dec.ranked, used_keys, s)

        if sel.chosen is not None:
            dec.selected, dec.status, dec.reason, dec.reused_clip = sel.chosen, "matched", sel.reason, sel.reused
            if sel.reused:
                dec.warnings.append("Clip already used in an earlier scene; reused because no other candidate passed the threshold.")
            if not sel.chosen.visual_verified:
                dec.warnings.append("Relevance based on clip title/tags only (no visual verification of frames).")
            return dec

        if not pool:
            dec.status = "provider_error" if errors and errors == len(dec.attempts) else "no_candidates"
            dec.reason = ("all footage searches failed" if dec.status == "provider_error"
                          else "no usable candidates returned for any query")
            dec.warnings.append(f"No footage for scene {scene.scene_id}: {dec.reason}.")
            return dec

        # Nothing passed the threshold. What happens now depends on the policy; a clearly wrong clip is never used silently.
        policy = s.fallback_policy
        best_unused = next((x for x in dec.ranked if x.candidate.key not in used_keys), dec.ranked[0])
        if policy == "best_available":
            # legacy behaviour: always use the best candidate, however poor (disclosed as low-confidence)
            dec.selected, dec.status = best_unused, "fallback_low_confidence"
            dec.reused_clip = best_unused.candidate.key in used_keys
            dec.reason = (f"no candidate reached threshold {s.match_threshold:.2f}; best available "
                          f"({best_unused.final_score:.2f}) used as an explicit LOW-CONFIDENCE fallback")
            dec.warnings.append(f"Scene {scene.scene_id}: footage may NOT accurately depict the narration "
                                f"(score {best_unused.final_score:.2f} < {s.match_threshold:.2f}).")
        elif policy == "review":
            reviewable = [x for x in dec.ranked if x.final_score >= s.review_floor]
            pick = next((x for x in reviewable if x.candidate.key not in used_keys), None)
            if pick is None and reviewable and s.allow_clip_reuse:
                pick = reviewable[0]
            if pick is not None:
                dec.selected, dec.status = pick, "fallback_low_confidence"
                dec.reused_clip = pick.candidate.key in used_keys
                why = pick.gated or f"score {pick.final_score:.2f} is below the match threshold {s.match_threshold:.2f}"
                dec.reason = f"needs review: {why}"
                dec.warnings.append(f"Scene {scene.scene_id}: footage may NOT accurately depict the narration ({why}).")
            else:
                dec.status = "unmatched"
                top = dec.ranked[0]
                dec.reason = (f"no candidate reached the minimum relevance floor {s.review_floor:.2f} "
                              f"(best was {top.final_score:.2f}: {top.gated or 'weak title/tag match'})")
                dec.warnings.append(f"Scene {scene.scene_id} has no matching footage: {dec.reason}.")
        else:
            dec.status = "unmatched"
            dec.reason = f"no candidate reached threshold {s.match_threshold:.2f} and fallback_policy=fail"
            dec.warnings.append(f"Scene {scene.scene_id} is unmatched.")
        return dec

    async def match_scenes(self, scenes: Sequence[Scene], aspect: str, hints: Optional[Sequence[Optional[str]]] = None,
                           progress: Optional[Callable[[int, int], None]] = None) -> List[SceneDecision]:
        used: Set[str] = set()
        out: List[SceneDecision] = []
        for i, sc in enumerate(scenes):
            hint = hints[i] if hints and i < len(hints) else None
            d = await self.match_scene(sc, aspect, used, hint)
            if d.selected is not None:
                used.add(d.selected.candidate.key)
            out.append(d)
            if progress:
                progress(i + 1, len(scenes))
        return out
