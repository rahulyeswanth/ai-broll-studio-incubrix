"""Visual verification adapters (replaceable).

``VisualVerifier.score`` returns {candidate.key: score 0..1} only for candidates it actually inspected.
* ``NullVisualVerifier``        - does nothing; the pipeline then reports the metadata fallback honestly.
* ``UnavailableVisualVerifier`` - verification was REQUESTED but cannot run (missing package / model); behaves like
  Null for scoring and carries the reason so the manifest and UI can say so.
* ``ContrastiveVisualVerifier`` - action-aware zero-shot check on frames sampled from the downloaded clip, using any
  ``EmbeddingBackend`` (image + text embeddings). Frames are compared with *positive* prompts (the scene) AND with
  *foil* prompts (related-but-different activities such as "doctor consultation" for a blood-pressure scene), so a
  clip that merely contains the right people scores low when a foil explains the frames better.
* ``ClipVisualVerifier``        - ContrastiveVisualVerifier on OpenCLIP. Needs ``open_clip_torch`` + ``torch`` + model
  weights (optional; not installed by default). NOT exercised with real weights in this repository's test suite.

The score is a heuristic ranking signal; it is not a calibrated probability.
"""
from __future__ import annotations

import logging
import math
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .media import ffmpeg_bin, run
from .models import Candidate, SceneAnalysis

log = logging.getLogger(__name__)


def extract_frames(video_path: Path, out_dir: Path, count: int = 3) -> List[Path]:
    """Sample ``count`` evenly spaced frames using FFmpeg. Returns the frame image paths that exist."""
    from .media import video_info
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        dur = video_info(video_path)["duration"] or 1.0
    except ValueError:
        return []
    paths: List[Path] = []
    for i in range(count):
        t = dur * (i + 1) / (count + 1)
        dest = out_dir / f"frame_{i}.jpg"
        cp = run([ffmpeg_bin(), "-hide_banner", "-y", "-ss", f"{t:.3f}", "-i", str(video_path), "-frames:v", "1", "-q:v", "3", str(dest)], timeout=60)
        if cp.returncode == 0 and dest.is_file() and dest.stat().st_size > 0:
            paths.append(dest)
    return paths


class VisualVerifier(ABC):
    name = "none"
    version: Optional[str] = None
    performs_visual_analysis = False
    unavailable_reason: Optional[str] = None
    details: Dict[str, Dict] = {}

    @abstractmethod
    def score(self, analysis: SceneAnalysis, candidates: Sequence[Candidate],
              fetch_preview: Callable[[Candidate, Path], Optional[Path]]) -> Dict[str, float]:
        ...


class NullVisualVerifier(VisualVerifier):
    def score(self, analysis, candidates, fetch_preview):
        return {}


class UnavailableVisualVerifier(NullVisualVerifier):
    """Visual verification was requested but cannot run. Scoring falls back to metadata, disclosed everywhere."""
    name = "unavailable"

    def __init__(self, requested: str, reason: str):
        self.requested, self.unavailable_reason = requested, reason


class EmbeddingBackend(ABC):
    """Anything that can embed text and images into one shared, L2-normalised vector space (e.g. CLIP)."""
    name = "embedding"
    version: Optional[str] = None

    @abstractmethod
    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        ...

    @abstractmethod
    def embed_images(self, paths: Sequence[Path]) -> List[List[float]]:
        ...


def _softmax(xs: Sequence[float]) -> List[float]:
    m = max(xs)
    ex = [math.exp(x - m) for x in xs]
    s = sum(ex)
    return [e / s for e in ex]


def build_prompts(analysis: SceneAnalysis, max_foils: int = 4) -> Tuple[List[str], List[str]]:
    """Positive prompts describe the scene; foil prompts describe confusable alternatives (same subject, other action)."""
    positives = [analysis.visual_description] + [f"a photo of {q}" for q in analysis.queries[:2]]
    who = analysis.subject or "a person"
    foils = [f"a photo of {who} {f}" for f in analysis.foil_terms[:max_foils]]
    if analysis.subject:
        foils.append("an empty place with nobody in it")
    return list(dict.fromkeys(p for p in positives if p)), foils


class ContrastiveVisualVerifier(VisualVerifier):
    performs_visual_analysis = True
    # CLIP cosine similarity of a matching image/text pair is typically 0.20-0.35; this maps it onto 0..1 (heuristic).
    SIM_LOW, SIM_SPAN, TEMPERATURE = 0.12, 0.23, 100.0

    def __init__(self, backend: EmbeddingBackend, frames: int = 3):
        self.backend, self.frames = backend, frames
        self.name = getattr(backend, "name", "embedding")
        self.version = getattr(backend, "version", None)
        self.details = {}

    def score(self, analysis, candidates, fetch_preview):
        out: Dict[str, float] = {}
        pos, foils = build_prompts(analysis)
        prompts = pos + foils
        text = self.backend.embed_texts(prompts)           # may raise: the matcher records the failure and falls back
        for c in candidates:
            with tempfile.TemporaryDirectory(prefix="visv_") as tmp:
                try:
                    clip_path = fetch_preview(c, Path(tmp))
                    if not clip_path:
                        continue
                    frames = [clip_path] if clip_path.suffix.lower() in (".jpg", ".jpeg", ".png") else \
                        extract_frames(clip_path, Path(tmp) / "f", self.frames)
                    if not frames:
                        continue
                    imgs = self.backend.embed_images(frames)
                    per_frame_mass: List[float] = []
                    per_frame_abs: List[float] = []
                    top_foil: Optional[str] = None
                    foil_votes = 0
                    for v in imgs:
                        sims = [sum(a * b for a, b in zip(v, tv)) for tv in text]
                        best_pos = max(sims[: len(pos)])
                        per_frame_abs.append(min(1.0, max(0.0, (best_pos - self.SIM_LOW) / self.SIM_SPAN)))
                        if foils:
                            probs = _softmax([self.TEMPERATURE * s for s in sims])
                            mass = sum(probs[: len(pos)])
                            per_frame_mass.append(mass)
                            if mass < 0.5:
                                foil_votes += 1
                                k = max(range(len(pos), len(sims)), key=lambda i: sims[i])
                                top_foil = prompts[k]
                    abs_score = sum(per_frame_abs) / len(per_frame_abs)
                    mass = (sum(per_frame_mass) / len(per_frame_mass)) if per_frame_mass else 1.0
                    leading = bool(foils) and foil_votes > len(imgs) / 2
                    out[c.key] = float(min(1.0, max(0.0, abs_score * (0.5 + 0.5 * mass))))
                    self.details[c.key] = {"frames_inspected": len(imgs), "positive_mass": round(mass, 3),
                                           "best_positive_similarity": round(abs_score, 3), "foil_leading": leading,
                                           "top_foil": top_foil if leading else None, "backend": self.name}
                except Exception as exc:  # noqa: BLE001 - one bad clip must not cost the others
                    log.warning("Visual scoring failed for %s: %s", c.key, exc)
        return out


class OpenClipBackend(EmbeddingBackend):
    def __init__(self, model_name: str, pretrained: str, device: str = "cpu"):
        import open_clip  # type: ignore   (ImportError is handled by get_visual_verifier)
        import torch  # type: ignore
        self._torch, self._device = torch, device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()
        self.name = "openclip"
        self.version = f"open_clip {getattr(open_clip, '__version__', 'unknown')} / {model_name}:{pretrained}"

    def embed_texts(self, texts):
        torch = self._torch
        with torch.no_grad():
            f = self.model.encode_text(self.tokenizer(list(texts)).to(self._device))
            f = f / f.norm(dim=-1, keepdim=True)
        return f.cpu().tolist()

    def embed_images(self, paths):
        from PIL import Image  # type: ignore
        torch = self._torch
        with torch.no_grad():
            imgs = torch.stack([self.preprocess(Image.open(p).convert("RGB")) for p in paths]).to(self._device)
            f = self.model.encode_image(imgs)
            f = f / f.norm(dim=-1, keepdim=True)
        return f.cpu().tolist()


class ClipVisualVerifier(ContrastiveVisualVerifier):
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k", device: str = "cpu", frames: int = 3):
        try:
            backend = OpenClipBackend(model_name, pretrained, device)
        except ImportError as exc:
            raise ImportError("ClipVisualVerifier needs `pip install open_clip_torch torch pillow`. "
                              "Set VISUAL_VERIFIER=metadata to use the transparent metadata fallback.") from exc
        super().__init__(backend, frames)


def get_visual_verifier(name: str, settings) -> VisualVerifier:
    """Never raises for a missing/failed model: returns an ``UnavailableVisualVerifier`` carrying the reason so the job
    still runs on metadata (and says so). Unknown names are reported the same way."""
    name = (name or "metadata").lower()
    if name in ("metadata", "none", ""):
        return NullVisualVerifier()
    if name == "clip":
        try:
            return ClipVisualVerifier(settings.clip_model_name, settings.clip_pretrained)
        except ImportError as exc:
            return UnavailableVisualVerifier(name, str(exc))
        except Exception as exc:  # noqa: BLE001 - weights download / device errors
            return UnavailableVisualVerifier(name, f"the vision model could not be loaded ({type(exc).__name__}: {exc})")
    return UnavailableVisualVerifier(name, f"unknown VISUAL_VERIFIER value '{name}' (use 'metadata' or 'clip')")
