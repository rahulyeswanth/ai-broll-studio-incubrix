
import logging
from typing import List

import httpx

from app.providers.base import FootageProvider, BrollAsset

logger = logging.getLogger(__name__)


class PixabayFootageProvider(FootageProvider):
    """Search Pixabay's public video API for B-roll footage."""

    BASE_URL = "https://pixabay.com/api/videos/"

    def __init__(self, api_key: str | None = None, strict_orientation: bool = True):
        if not api_key:
            raise ValueError("PIXABAY_API_KEY is not configured.")
        self.api_key = api_key
        # strict_orientation=True keeps the legacy behaviour (drop clips whose orientation differs from the
        # request). The semantic pipeline passes False and lets the ranker penalise/crop instead, which matters
        # for 9:16 where Pixabay has few portrait clips.
        self.strict_orientation = strict_orientation

    @staticmethod
    def _actual_ratio(width, height, fallback: str) -> str:
        if not width or not height:
            return fallback
        r = width / height
        return min({"16:9": 16 / 9, "9:16": 9 / 16, "1:1": 1.0}.items(), key=lambda kv: abs(kv[1] - r))[0]

    async def search_footage(
        self,
        query: str,
        aspect_ratio: str = "16:9",
        max_results: int = 5,
    ) -> List[BrollAsset]:

        params = {
            "key": self.api_key,
            "q": query,
            "per_page": min(max(max_results * 3, 3), 50),
            "safesearch": "true",
        }

        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.get(self.BASE_URL, params=params)
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in (401, 403):
                raise PermissionError(
                    "Pixabay authentication failed. Check PIXABAY_API_KEY."
                ) from exc
            raise RuntimeError(
                f"Pixabay API returned HTTP {exc.response.status_code}."
            ) from exc
        except httpx.RequestError as exc:
            raise RuntimeError(
                f"Could not connect to Pixabay: {exc}"
            ) from exc

        data = response.json()
        assets: List[BrollAsset] = []

        for video in data.get("hits", []):
            renditions = video.get("videos", {})

            # Prefer large, then medium, then small MP4 renditions.
            selected = None
            for quality in ("large", "medium", "small", "tiny"):
                candidate = renditions.get(quality, {})
                if candidate.get("url"):
                    selected = candidate
                    break

            if not selected:
                continue

            width = selected.get("width")
            height = selected.get("height")

            # Filter orientation locally because the Pixabay video
            # endpoint does not provide an orientation parameter.
            if width and height and self.strict_orientation:
                if aspect_ratio == "16:9" and width < height:
                    continue
                if aspect_ratio == "9:16" and height < width:
                    continue
                if aspect_ratio == "1:1":
                    ratio = width / height
                    if not 0.85 <= ratio <= 1.15:
                        continue

            user = video.get("user", "Unknown")
            page_url = video.get("pageURL")
            tags = video.get("tags", "")

            assets.append(
                BrollAsset(
                    id=str(video.get("id")),
                    source="pixabay",
                    title=tags or f"Pixabay video {video.get('id')}",
                    download_url=selected["url"],
                    preview_url=selected.get("thumbnail"),
                    aspect_ratio=self._actual_ratio(width, height, aspect_ratio),
                    duration=float(video.get("duration", 0)),
                    width=width,
                    height=height,
                    commercial_use_allowed=True,
                    license_type="Pixabay Content License - verify asset eligibility",
                    attribution_required=False,
                    page_url=page_url,
                    tags=[t.strip() for t in tags.split(",") if t.strip()],
                    author_name=user,
                    author_url=(
                        f"https://pixabay.com/users/{user}-{video.get('user_id')}/"
                        if video.get("user_id")
                        else None
                    ),
                )
            )

            if len(assets) >= max_results:
                break

        return assets
