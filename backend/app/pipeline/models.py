"""Plain dataclasses shared across the pipeline (no third-party dependencies)."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class WordTiming:
    word: str
    start: float
    end: float

    def to_dict(self) -> Dict[str, Any]:
        return {"word": self.word, "start": round(self.start, 3), "end": round(self.end, 3)}


@dataclass
class Scene:
    scene_id: str
    index: int                 # 1-based
    text: str                  # original narration text, never rewritten
    start: float = 0.0
    end: float = 0.0
    words: List[WordTiming] = field(default_factory=list)
    timing_method: str = "unknown"

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 3)


@dataclass
class SceneAnalysis:
    """Structured visual understanding of one narration scene."""
    subject: Optional[str]
    action: Optional[str]
    setting: Optional[str]
    objects: List[str]
    visual_description: str
    queries: List[str]
    avoid_terms: List[str]
    search_terms: List[str]
    method: str = "rule_based_v1"
    notes: List[str] = field(default_factory=list)
    term_weights: Dict[str, float] = field(default_factory=dict)   # term -> importance for ranking
    context: Optional[str] = None
    # --- action-aware evidence (filled from the activity lexicon; empty for scenes with no known frame) ---------
    scene_type: str = "activity"           # activity | place | object | abstract
    frame_ids: List[str] = field(default_factory=list)
    gated_frame: bool = False               # primary frame may hard-gate clips that lack evidence of the action
    subject_terms: List[str] = field(default_factory=list)   # subject + role synonyms
    action_terms: List[str] = field(default_factory=list)    # evidence that the ACTION is shown
    object_terms: List[str] = field(default_factory=list)    # instruments/equipment whose presence evidences the action (cuff, beaker, camera)
    target_terms: List[str] = field(default_factory=list)    # passive targets named in the narration (tomato plants, Eiffel Tower): score, never proof
    setting_terms: List[str] = field(default_factory=list)
    foil_terms: List[str] = field(default_factory=list)      # related-but-different actions to penalise
    negative_terms: List[str] = field(default_factory=list)  # terms that make a clip wrong for this scene
    sense_good_terms: List[str] = field(default_factory=list)  # metadata words that confirm the resolved word sense
    sense_bad_terms: List[str] = field(default_factory=list)   # metadata words that indicate a different sense of the same word
    content_words: List[str] = field(default_factory=list)     # narration words a clip should reflect (proper names excluded)
    grounding_terms: List[str] = field(default_factory=list)   # content words + brand stand-in words, used by the grounding gate
    strict_grounding: bool = False                             # named brand/product in the scene: clip must cover >= 2 scene terms
    sensitive: bool = False                                    # sad/serious story: sombre queries, upbeat clips rejected

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Candidate:
    """Provider-neutral footage candidate."""
    provider: str
    clip_id: str
    download_url: str
    title: str = ""
    tags: List[str] = field(default_factory=list)
    page_url: Optional[str] = None
    preview_url: Optional[str] = None
    duration: float = 0.0
    width: Optional[int] = None
    height: Optional[int] = None
    creator: Optional[str] = None
    creator_url: Optional[str] = None
    license_name: Optional[str] = None
    license_url: Optional[str] = None
    license_basis: str = "unknown"            # provider_policy_documented | per_clip_verified | unknown
    commercial_use: Optional[bool] = None       # None == unknown
    attribution_required: Optional[bool] = None
    ai_generated: Optional[bool] = None         # None == provider does not say
    raw_commercial_flag: Optional[bool] = None
    retrieved_at: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.provider}:{self.clip_id}"


@dataclass
class ScoredCandidate:
    candidate: Candidate
    text_score: float
    visual_score: Optional[float]
    technical_score: float
    final_score: float
    method: str
    reasons: List[str] = field(default_factory=list)
    passed: bool = False
    visual_verified: bool = False
    penalties: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)   # per-element coverage 0..1 + which terms matched
    gated: Optional[str] = None             # reason the score was capped below the threshold (e.g. subject-only match)
    visual_detail: Dict[str, Any] = field(default_factory=dict)
    label: str = "Low"                      # High | Medium | Low  (documented thresholds, NOT a probability)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clip": self.candidate.key, "title": self.candidate.title,
            "text_score": round(self.text_score, 3),
            "visual_score": None if self.visual_score is None else round(self.visual_score, 3),
            "technical_score": round(self.technical_score, 3),
            "final_score": round(self.final_score, 3), "method": self.method,
            "passed": self.passed, "visual_verified": self.visual_verified,
            "reasons": self.reasons, "penalties": self.penalties,
            "label": self.label, "gated": self.gated, "evidence": self.evidence, "visual_detail": self.visual_detail,
        }
