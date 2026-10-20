import logging
from typing import Any, Dict, List, Optional

import httpx

from app.providers.base import BrollAsset, FootageProvider

logger = logging.getLogger(__name__)


class CoverrFootageProvider(FootageProvider):
    """Footage search through the Coverr API (https://api.coverr.co/docs/).

    Licence: Coverr's free licence allows commercial use without attribution; resale of the clips and AI-model
    training are not allowed. Coverr's API terms also ask apps to show a Coverr link/logo and to ping
    ``PATCH /videos/:id/stats/downloads`` when a clip is downloaded (see ``track_download``).
    """

    BASE_URL = "https://api.coverr.co"

    def __init__(self, api_key: Optional[str] = None):
        if not api_key:
            raise ValueError("Coverr API key is not configured. Set COVERR_API_KEY.")
        self.api_key = api_key

    @property
    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    async def search_footage(
        self, query: str, aspect_ratio: str = "16:9", max_results: int = 5
    ) -> List[BrollAsset]:
        params = {"query": query, "page_size": max(1, min(int(max_results) * 2, 50)), "urls": "true"}
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.get(f"{self.BASE_URL}/videos", headers=self._headers, params=params)
            except httpx.RequestError as exc:
                logger.error("Network error querying Coverr API: %s", type(exc).__name__)
                raise RuntimeError(f"Failed to connect to Coverr API: {type(exc).__name__}") from exc

        if response.status_code in (401, 403):
            raise PermissionError(
                f"Coverr API authentication failed (status {response.status_code}). Verify COVERR_API_KEY.")
        if response.status_code != 200:
            raise RuntimeError(f"Coverr API returned HTTP {response.status_code}")
        return self.to_assets(response.json(), aspect_ratio, max_results)

    @staticmethod
    def to_assets(data: Dict[str, Any], aspect_ratio: str, max_results: int) -> List[BrollAsset]:
        hits = list(data.get("hits") or [])
        vertical = [h for h in hits if h.get("is_vertical")]
        landscape = [h for h in hits if not h.get("is_vertical")]
        if aspect_ratio == "9:16":
            ordered = vertical + landscape          # prefer true vertical clips, fall back to cropping landscape
        else:
            ordered = landscape
        assets: List[BrollAsset] = []
        for hit in ordered:
            urls = hit.get("urls") or {}
            download_url = urls.get("mp4") or urls.get("mp4_download")
            if not hit.get("id") or not download_url:
                continue
            tags = [str(t) for t in (hit.get("tags") or []) if t]
            assets.append(BrollAsset(
                id=str(hit["id"]),
                source="coverr",
                title=str(hit.get("title") or f"Coverr video {hit['id']}"),
                download_url=download_url,
                preview_url=hit.get("thumbnail") or hit.get("poster"),
                aspect_ratio=aspect_ratio,
                duration=float(hit.get("duration") or 0.0),
                width=hit.get("max_width"),
                height=hit.get("max_height"),
                commercial_use_allowed=True,
                license_type="Coverr License (commercial use allowed, attribution not required)",
                attribution_required=False,
                author_name="Coverr",
                author_url="https://coverr.co",
                tags=tags,
            ))
            if len(assets) >= max_results:
                break
        return assets

    async def track_download(self, asset_id: str) -> bool:
        """Best-effort download ping requested by Coverr's API terms. Never raises."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.patch(f"{self.BASE_URL}/videos/{asset_id}/stats/downloads", headers=self._headers)
            return r.status_code == 204
        except Exception:  # noqa: BLE001
            return False
