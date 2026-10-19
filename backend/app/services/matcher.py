import logging
from typing import List, Optional, Tuple

from app.providers.base import FootageProvider, BrollAsset
from app.providers.footage import get_footage_provider
from app.schemas.domain import SceneMatchResult, SceneSegment

logger = logging.getLogger(__name__)


class BrollMatchingService:
    """Match each narration scene to a suitable candidate from a footage provider."""

    def __init__(self, footage_provider: Optional[FootageProvider] = None):
        self.provider = footage_provider or get_footage_provider()

    @staticmethod
    def _determine_search_query(scene: SceneSegment) -> str:
        if scene.visual_hint and scene.visual_hint.strip():
            return scene.visual_hint.strip()
        if scene.broll_query and scene.broll_query.strip():
            return scene.broll_query.strip()
        return scene.narration_text.strip() or "cinematic backdrop"

    @staticmethod
    def _select_best_candidate(
        candidates: List[BrollAsset], target_aspect_ratio: str, scene_duration: float
    ) -> Tuple[Optional[BrollAsset], Optional[str]]:
        if not candidates:
            return None, "No candidate footage returned by provider."

        # Fail closed: only select assets whose provider metadata explicitly allows commercial use.
        commercial_safe = [c for c in candidates if c.commercial_use_allowed is True]
        if not commercial_safe:
            return None, "Candidate footage was found, but none is cleared for commercial use."

        exact_ratio = [c for c in commercial_safe if c.aspect_ratio == target_aspect_ratio]
        pool = exact_ratio or commercial_safe
        long_enough = [c for c in pool if c.duration >= scene_duration]
        best = long_enough[0] if long_enough else pool[0]
        warning = None
        if best.aspect_ratio != target_aspect_ratio:
            warning = (f"No exact {target_aspect_ratio} clip available; selected {best.aspect_ratio} "
                       "for letterbox/center-crop.")
        if best.duration < scene_duration:
            duration_warning = (f"Selected clip is {best.duration:.2f}s but scene duration is "
                                f"{scene_duration:.2f}s; looping or trimming may be required.")
            warning = f"{warning} {duration_warning}" if warning else duration_warning
        return best, warning

    async def match_scene(self, scene: SceneSegment, target_aspect_ratio: str) -> SceneMatchResult:
        query = self._determine_search_query(scene)
        scene_copy = scene.model_copy(deep=True)
        try:
            candidates = await self.provider.search_footage(
                query=query, aspect_ratio=target_aspect_ratio, max_results=5
            )
        except Exception as exc:
            logger.warning("Footage provider error for query %r: %s", query, exc, exc_info=True)
            scene_copy.selected_asset = None
            return SceneMatchResult(scene=scene_copy, match_status="provider_error",
                                    candidates_found=0, warning=f"Footage provider error: {exc}")

        if not candidates:
            scene_copy.selected_asset = None
            return SceneMatchResult(scene=scene_copy, match_status="no_candidates", candidates_found=0,
                                    warning=f"No stock footage found for query: '{query}'.")

        selected, warning = self._select_best_candidate(candidates, target_aspect_ratio, scene.duration_seconds)
        scene_copy.selected_asset = selected
        if selected is not None:
            return SceneMatchResult(scene=scene_copy, match_status="matched",
                                    candidates_found=len(candidates), warning=warning)
        return SceneMatchResult(scene=scene_copy, match_status="no_candidates",
                                candidates_found=len(candidates), warning=warning or
                                "No suitable candidate met commercial-use constraints.")

    async def match_scenes(self, scenes: List[SceneSegment], target_aspect_ratio: str):
        results: List[SceneMatchResult] = []
        warnings: List[str] = []
        for scene in scenes:
            result = await self.match_scene(scene, target_aspect_ratio)
            results.append(result)
            if result.warning:
                warnings.append(f"Scene {scene.scene_index} ({scene.scene_id}): {result.warning}")
        return results, warnings
