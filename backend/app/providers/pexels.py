import logging
from typing import List, Optional
import httpx

from app.providers.base import FootageProvider, BrollAsset

logger = logging.getLogger(__name__)


class PexelsFootageProvider(FootageProvider):
    """
    Real footage search provider using the official Pexels Video Search API.
    All footage retrieved under the Pexels license is cleared for commercial use.
    Attribution is preserved and tracked in returned metadata.
    """

    BASE_URL = "https://api.pexels.com/videos/search"

    def __init__(self, api_key: Optional[str] = None):
        if not api_key:
            raise ValueError(
                "Pexels API key is not configured. Set PEXELS_API_KEY environment variable or select 'mock' provider."
            )
        self.api_key = api_key

    def _map_orientation(self, aspect_ratio: str) -> str:
        mapping = {
            "16:9": "landscape",
            "9:16": "portrait",
            "1:1": "square",
        }
        return mapping.get(aspect_ratio, "landscape")

    async def search_footage(
        self, query: str, aspect_ratio: str = "16:9", max_results: int = 5
    ) -> List[BrollAsset]:
        orientation = self._map_orientation(aspect_ratio)
        headers = {"Authorization": self.api_key}
        params = {
            "query": query,
            "per_page": max_results,
            "orientation": orientation,
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.get(self.BASE_URL, headers=headers, params=params)
            except httpx.RequestError as exc:
                logger.error(f"Network error querying Pexels API: {exc}")
                raise RuntimeError(f"Failed to connect to Pexels API: {exc}") from exc

        if response.status_code in (401, 403):
            raise PermissionError(
                f"Pexels API authentication failed (status {response.status_code}). Verify your PEXELS_API_KEY."
            )
        if response.status_code != 200:
            raise RuntimeError(f"Pexels API returned HTTP {response.status_code}: {response.text}")

        data = response.json()
        raw_videos = data.get("videos", [])
        assets: List[BrollAsset] = []

        for video in raw_videos:
            video_files = video.get("video_files", [])
            if not video_files:
                continue

            # Pick the best HD MP4 stream
            selected_file = None
            for vf in video_files:
                if vf.get("file_type") == "video/mp4":
                    if vf.get("quality") == "hd":
                        selected_file = vf
                        break
                    elif selected_file is None:
                        selected_file = vf

            if not selected_file:
                selected_file = video_files[0]

            download_url = selected_file.get("link")
            if not download_url:
                continue

            user_info = video.get("user", {})
            author_name = user_info.get("name")
            author_url = user_info.get("url")

            # Extract clean title from Pexels video URL or fallback
            url_slug = video.get("url", "").rstrip("/").split("/")[-1]
            title = url_slug.replace("-", " ").title() if url_slug else f"Pexels Video {video.get('id')}"

            asset = BrollAsset(
                id=str(video.get("id")),
                source="pexels",
                title=title,
                download_url=download_url,
                preview_url=video.get("image"),
                aspect_ratio=aspect_ratio,
                duration=float(video.get("duration", 0.0)),
                width=selected_file.get("width"),
                height=selected_file.get("height"),
                # Commercial terms per Pexels License:
                commercial_use_allowed=True,
                license_type="Pexels License (Commercial Use Allowed)",
                attribution_required=False,
                author_name=author_name,
                author_url=author_url,
                page_url=video.get("url"),
            )
            assets.append(asset)

        return assets
