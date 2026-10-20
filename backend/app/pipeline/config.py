"""Pipeline settings, read from environment variables and an optional ``.env`` file.

Deliberately independent from ``app.core.config`` (Pydantic) so that the pipeline can be
imported and tested in minimal environments. Secrets are never logged or serialised.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Dict, Optional, Tuple

BACKEND_DIR = Path(__file__).resolve().parents[2]


def _parse_dotenv(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def _bool(v: Optional[str], default: bool) -> bool:
    if v is None or v == "":
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class PipelineSettings:
    # --- storage -----------------------------------------------------------------
    storage_root: Path = field(default_factory=lambda: BACKEND_DIR / "storage")
    jobs_subdir: str = "jobs"

    # --- providers ---------------------------------------------------------------
    tts_provider: str = "edge"              # edge | mock (placeholder tones, never "real")
    transcription_provider: str = "whisper"  # whisper | mock
    footage_provider: str = "pixabay"
    allow_mock_providers: bool = False
    default_voice: str = "en-US-AriaNeural"
    default_language: str = "en"
    whisper_model_size: str = "base"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    tts_timeout_seconds: float = 60.0
    transcription_timeout_seconds: float = 900.0

    # --- input limits ------------------------------------------------------------
    max_script_chars: int = 10000
    max_audio_mb: int = 50
    max_audio_seconds: float = 600.0       # assessment limit: 10 minutes
    min_audio_seconds: float = 1.0
    allowed_audio_extensions: Tuple[str, ...] = (".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac")
    silence_max_volume_db: float = -55.0  # audio whose loudest sample is below this is "silent"

    # --- scene building ----------------------------------------------------------
    min_scene_seconds: float = 2.0
    max_scene_seconds: float = 12.0

    # --- matching ----------------------------------------------------------------
    candidates_per_query: int = 12
    max_query_attempts: int = 6
    match_threshold: float = 0.45
    weight_visual: float = 0.55
    weight_text: float = 0.35
    weight_technical: float = 0.10
    visual_verifier: str = "metadata"       # metadata | clip
    clip_model_name: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"
    fallback_policy: str = "review"          # review | best_available | fail   (see README "Relevance thresholds")
    review_floor: float = 0.30               # below this a scene is UNMATCHED (never filled with a clearly wrong clip)
    high_confidence: float = 0.70            # label "High" needs a passing score >= this AND action/object evidence
    min_grounded_terms: int = 2              # a scene with >= 5 content words needs a clip whose metadata covers at least this many
    early_stop_score: float = 0.62           # stop trying alternate queries once a fresh clip scores this well
    # evidence weights inside the text/metadata score (only elements present in the scene count)
    evidence_weight_subject: float = 0.30
    evidence_weight_action: float = 0.30
    evidence_weight_object: float = 0.25
    evidence_weight_setting: float = 0.15
    foil_penalty: float = 0.10               # clip shows a related-but-different action (e.g. harvesting vs watering)
    sense_mismatch_penalty: float = 0.35     # clip metadata shows a different sense of an ambiguous word (python the snake vs the language)
    sense_match_bonus: float = 0.20          # clip metadata confirms the resolved sense
    max_download_alternates: int = 4         # extra passing candidates tried when the chosen clip cannot be downloaded
    duplicate_hash_distance: int = 6         # dHash Hamming distance (of 64) at/below which two clips count as near-duplicates
    allow_clip_reuse: bool = True            # reuse only if nothing else passes; always disclosed
    min_clip_seconds: float = 1.5

    # --- footage providers ---------------------------------------------------------
    footage_fallback_providers: Tuple[str, ...] = ()      # empty = every other provider that has an API key; ("none",) = off
    footage_min_primary_results: int = 3

    # --- download ----------------------------------------------------------------
    max_download_mb: int = 100
    download_timeout_seconds: float = 60.0
    allowed_download_hosts: Tuple[str, ...] = (
        "cdn.pixabay.com", "videos.pexels.com", "player.vimeo.com",
        "storage.coverr.co", "cdn.coverr.co", "api.coverr.co", "coverr.co",
    )
    require_https: bool = True

    # --- render ------------------------------------------------------------------
    fps: int = 30
    crf: int = 21
    preset: str = "veryfast"
    render_timeout_seconds: int = 1800
    normalize_color: bool = True
    caption_max_chars_per_line: int = 38
    caption_max_lines: int = 2
    caption_max_seconds: float = 6.0
    caption_min_seconds: float = 0.6

    # --- secrets (never serialised) ---------------------------------------------
    pixabay_api_key: Optional[str] = field(default=None, repr=False)
    pexels_api_key: Optional[str] = field(default=None, repr=False)
    coverr_api_key: Optional[str] = field(default=None, repr=False)

    @property
    def jobs_dir(self) -> Path:
        return self.storage_root / self.jobs_subdir

    def public_dict(self) -> Dict[str, object]:
        """Settings safe to embed in manifests (no secrets, no absolute paths)."""
        skip = {"pixabay_api_key", "pexels_api_key", "coverr_api_key", "storage_root"}
        out: Dict[str, object] = {}
        for f in fields(self):
            if f.name in skip:
                continue
            v = getattr(self, f.name)
            out[f.name] = list(v) if isinstance(v, tuple) else v
        return out


def load_settings(env: Optional[Dict[str, str]] = None, dotenv_path: Optional[Path] = None) -> PipelineSettings:
    """Build settings. Precedence: explicit ``env`` > process environment > ``.env`` file."""
    merged: Dict[str, str] = {}
    merged.update(_parse_dotenv(dotenv_path or (BACKEND_DIR / ".env")))
    merged.update({k: v for k, v in os.environ.items()})
    if env:
        merged.update(env)
    g = merged.get
    s = PipelineSettings()

    root = g("STORAGE_ROOT")
    if root:
        p = Path(root)
        s.storage_root = (p if p.is_absolute() else BACKEND_DIR / p).resolve()

    s.tts_provider = (g("TTS_PROVIDER") or s.tts_provider).lower()
    s.transcription_provider = (g("TRANSCRIPTION_PROVIDER") or s.transcription_provider).lower()
    s.footage_provider = (g("FOOTAGE_PROVIDER") or s.footage_provider).lower()
    s.allow_mock_providers = _bool(g("ALLOW_MOCK_PROVIDERS"), s.allow_mock_providers)
    s.default_voice = g("TTS_DEFAULT_VOICE") or s.default_voice
    s.default_language = g("DEFAULT_LANGUAGE") or s.default_language
    s.whisper_model_size = g("WHISPER_MODEL_SIZE") or s.whisper_model_size
    s.whisper_device = g("WHISPER_DEVICE") or s.whisper_device
    s.whisper_compute_type = g("WHISPER_COMPUTE_TYPE") or s.whisper_compute_type
    s.max_audio_mb = int(g("MAX_AUDIO_FILE_SIZE_MB") or s.max_audio_mb)
    s.max_audio_seconds = float(g("MAX_AUDIO_SECONDS") or s.max_audio_seconds)
    s.transcription_timeout_seconds = float(g("TRANSCRIPTION_TIMEOUT_SECONDS") or s.transcription_timeout_seconds)
    s.max_script_chars = int(g("MAX_SCRIPT_CHAR_LENGTH") or s.max_script_chars)
    s.match_threshold = float(g("MATCH_THRESHOLD") or s.match_threshold)
    s.visual_verifier = (g("VISUAL_VERIFIER") or s.visual_verifier).lower()
    s.fallback_policy = (g("MATCH_FALLBACK_POLICY") or s.fallback_policy).lower()
    s.review_floor = float(g("MATCH_REVIEW_FLOOR") or s.review_floor)
    s.min_grounded_terms = int(g("MATCH_MIN_GROUNDED_TERMS") or s.min_grounded_terms)
    s.high_confidence = float(g("MATCH_HIGH_CONFIDENCE") or s.high_confidence)
    s.max_download_alternates = int(g("MAX_DOWNLOAD_ALTERNATES") or s.max_download_alternates)
    s.duplicate_hash_distance = int(g("DUPLICATE_HASH_DISTANCE") or s.duplicate_hash_distance)
    s.min_scene_seconds = float(g("MIN_SCENE_DURATION_SECONDS") or s.min_scene_seconds)
    s.max_scene_seconds = float(g("MAX_SCENE_DURATION_SECONDS") or s.max_scene_seconds)
    s.max_download_mb = int(g("MAX_VIDEO_DOWNLOAD_MB") or s.max_download_mb)
    fb = (g("FOOTAGE_FALLBACK_PROVIDERS") or "").strip().lower()
    s.footage_fallback_providers = tuple(x.strip() for x in fb.split(",") if x.strip())
    s.footage_min_primary_results = int(g("FOOTAGE_MIN_PRIMARY_RESULTS") or s.footage_min_primary_results)
    s.normalize_color = _bool(g("NORMALIZE_COLOR"), s.normalize_color)
    s.pixabay_api_key = g("PIXABAY_API_KEY") or None
    s.pexels_api_key = g("PEXELS_API_KEY") or None
    s.coverr_api_key = g("COVERR_API_KEY") or None
    return s
