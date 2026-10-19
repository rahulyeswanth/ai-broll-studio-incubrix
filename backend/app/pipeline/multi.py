"""Multi-ratio jobs: one run of the pipeline per requested ratio, merged into one job directory.

Honest limitation: each ratio runs its own narration/transcription and footage search (they share the input and the
configuration, not a cached scene timeline), so scene boundaries can differ slightly between ratios when TTS or
matching is non-deterministic. Every ratio's own timeline/manifest is kept (timeline_{ratio}.json, manifest_{ratio}.json).
"""
from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .orchestrator import Deps, JobRequest, run_pipeline


def ratio_slug(ratio: str) -> str:
    return ratio.replace(":", "x")


async def run_multi_ratio(req: JobRequest, deps: Deps, job_dir: Path, job_id: str,
                          progress: Optional[Callable[[float, str], None]] = None) -> Dict[str, Any]:
    ratios: List[str] = list(dict.fromkeys(req.aspect_ratios or [req.aspect_ratio]))
    n = len(ratios)
    merged_outputs: Dict[str, Any] = {"videos": {}}
    manifests: Dict[str, Dict[str, Any]] = {}
    warnings: List[str] = []
    for i, ratio in enumerate(ratios):
        sub = job_dir / f"r_{ratio_slug(ratio)}"
        shutil.rmtree(sub, ignore_errors=True)

        def P(pct: float, msg: str, i=i, ratio=ratio):
            if progress:
                progress((i + pct / 100.0) / n * 98.0, f"[{ratio}] {msg}")

        sub_req = dataclasses.replace(req, aspect_ratio=ratio, aspect_ratios=[])
        m = await run_pipeline(sub_req, deps, sub, job_id, P)
        manifests[ratio] = m
        slug = ratio_slug(ratio)
        shutil.copy2(sub / "video.mp4", job_dir / f"video_{slug}.mp4")
        shutil.copy2(sub / "timeline.json", job_dir / f"timeline_{slug}.json")
        shutil.copy2(sub / "manifest.json", job_dir / f"manifest_{slug}.json")
        merged_outputs["videos"][ratio] = f"video_{slug}.mp4"
        if i == 0:
            for name in ("captions.srt", "timeline.json"):
                if (sub / name).is_file():
                    shutil.copy2(sub / name, job_dir / name)
            for p in sub.glob("narration.*"):
                shutil.copy2(p, job_dir / p.name)
        warnings += [f"[{ratio}] {w}" for w in m.get("warnings", [])]
        shutil.rmtree(sub, ignore_errors=True)
    first = manifests[ratios[0]]
    merged = {
        "job_id": job_id, "kind": "multi_ratio", "aspect_ratios": ratios, "status": "completed",
        "outputs": {**merged_outputs, "captions": "captions.srt" if (job_dir / "captions.srt").is_file() else None,
                    "timeline": "timeline.json", "manifest": "manifest.json",
                    "per_ratio": {r: {"video": f"video_{ratio_slug(r)}.mp4", "timeline": f"timeline_{ratio_slug(r)}.json",
                                      "manifest": f"manifest_{ratio_slug(r)}.json"} for r in ratios}},
        "warnings": list(dict.fromkeys(warnings)),
        "limitations": ["Each ratio is produced by its own pipeline run; scene timing can differ slightly between ratios."],
        "per_ratio": {r: {"status": m.get("status"), "durations": m.get("durations"), "validation": m.get("validation")}
                      for r, m in manifests.items()},
        "reference_manifest_ratio": ratios[0],
        "reference_manifest": first,
    }
    (job_dir / "manifest.json").write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    progress and progress(100, "Done")
    return merged
