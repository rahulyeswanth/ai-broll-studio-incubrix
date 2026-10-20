import asyncio

from app.pipeline.multi_provider import MultiProviderSearch
from app.providers.base import BrollAsset
from app.providers.coverr import CoverrFootageProvider

SAMPLE = {"hits": [
    {"id": "abc", "title": "Ocean waves", "tags": ["ocean", "waves"], "is_vertical": False, "duration": 11.6,
     "max_width": 2048, "max_height": 1152, "thumbnail": "https://x/t.jpg",
     "urls": {"mp4": "https://storage.coverr.co/v/abc.mp4"}},
    {"id": "vert", "title": "Phone beach", "tags": ["beach"], "is_vertical": True, "duration": 8.0,
     "urls": {"mp4": "https://storage.coverr.co/v/vert.mp4"}},
    {"id": "nourl", "title": "No url", "tags": [], "is_vertical": False, "duration": 5.0, "urls": {}},
]}


class FailingProvider:
    async def search_footage(self, query, aspect_ratio="16:9", max_results=5):
        raise RuntimeError("Pixabay is down")


class FakeCoverr:
    async def search_footage(self, query, aspect_ratio="16:9", max_results=5):
        return CoverrFootageProvider.to_assets(SAMPLE, aspect_ratio, max_results)


def test_coverr_parsing_and_licence_fields():
    assets = CoverrFootageProvider.to_assets(SAMPLE, "16:9", 5)
    assert [a.id for a in assets] == ["abc"]
    a = assets[0]
    assert a.source == "coverr" and a.commercial_use_allowed and not a.attribution_required
    assert a.tags == ["ocean", "waves"] and a.download_url.endswith("abc.mp4")


def test_coverr_prefers_vertical_for_9x16():
    assets = CoverrFootageProvider.to_assets(SAMPLE, "9:16", 5)
    assert [a.id for a in assets] == ["vert", "abc"]


def test_job_still_gets_footage_when_primary_source_is_down():
    search = MultiProviderSearch([("pixabay", FailingProvider()), ("coverr", FakeCoverr())], min_results=3)
    got = asyncio.run(search("ocean waves", "16:9", 5))
    assert [a.source for a in got] == ["coverr"]
    entry = search.log["ocean waves"]
    assert entry["errors"] == {"pixabay": "RuntimeError"} and entry["fallback_used"] is True


def test_all_sources_down_raises():
    search = MultiProviderSearch([("pixabay", FailingProvider()), ("coverr", FailingProvider())], min_results=3)
    try:
        asyncio.run(search("ocean", "16:9", 5))
    except RuntimeError:
        return
    raise AssertionError("expected the error to surface when every provider fails")
