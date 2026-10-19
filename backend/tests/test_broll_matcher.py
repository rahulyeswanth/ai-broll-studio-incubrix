import pytest
from unittest.mock import AsyncMock

from app.providers.base import BrollAsset, FootageProvider
from app.providers.mocks import MockFootageProvider
from app.schemas.domain import SceneSegment
from app.services.matcher import BrollMatchingService


def make_scene(**kwargs):
    values = dict(scene_id="scene-001", scene_index=1, start_time=0.0, end_time=3.0,
                  duration_seconds=3.0, narration_text="Ocean wildlife.", broll_query="ocean wildlife")
    values.update(kwargs)
    return SceneSegment(**values)


class EmptyProvider(FootageProvider):
    async def search_footage(self, query, aspect_ratio="16:9", max_results=5):
        return []


class FailingProvider(FootageProvider):
    async def search_footage(self, query, aspect_ratio="16:9", max_results=5):
        raise RuntimeError("network timeout")


class NonCommercialProvider(FootageProvider):
    async def search_footage(self, query, aspect_ratio="16:9", max_results=5):
        return [BrollAsset(id="bad", source="test", title="Editorial clip", download_url="https://example.test/a.mp4",
                           aspect_ratio=aspect_ratio, duration=8, commercial_use_allowed=False, license_type="Editorial-Only")]


@pytest.mark.asyncio
async def test_mock_provider_matches_scene_and_aspect_ratio():
    result = await BrollMatchingService(MockFootageProvider()).match_scene(make_scene(), "9:16")
    assert result.match_status == "matched"
    assert result.scene.selected_asset.aspect_ratio == "9:16"
    assert result.scene.selected_asset.commercial_use_allowed is True


@pytest.mark.asyncio
async def test_visual_hint_is_preferred():
    provider = MockFootageProvider()
    provider.search_footage = AsyncMock(wraps=provider.search_footage)
    scene = make_scene(visual_hint="sea turtle underwater")
    await BrollMatchingService(provider).match_scene(scene, "16:9")
    provider.search_footage.assert_awaited_once_with(query="sea turtle underwater", aspect_ratio="16:9", max_results=5)


@pytest.mark.asyncio
async def test_empty_provider_result_is_graceful():
    result = await BrollMatchingService(EmptyProvider()).match_scene(make_scene(), "16:9")
    assert result.match_status == "no_candidates" and result.scene.selected_asset is None


@pytest.mark.asyncio
async def test_provider_failure_is_graceful():
    result = await BrollMatchingService(FailingProvider()).match_scene(make_scene(), "16:9")
    assert result.match_status == "provider_error" and result.scene.selected_asset is None


@pytest.mark.asyncio
async def test_noncommercial_assets_are_rejected():
    result = await BrollMatchingService(NonCommercialProvider()).match_scene(make_scene(), "16:9")
    assert result.match_status == "no_candidates" and result.scene.selected_asset is None
    assert "none is cleared" in result.warning


@pytest.mark.asyncio
async def test_api_matches_scenes(async_client):
    response = await async_client.post("/api/v1/broll/match", json={"scenes": [{
        "scene_id": "scene-001", "scene_index": 1, "start_time": 0, "end_time": 3,
        "duration_seconds": 3, "narration_text": "Ocean wildlife", "broll_query": "ocean wildlife",
        "visual_hint": "sea turtle"}], "target_aspect_ratio": "16:9", "provider": "mock"})
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["matched_scenes"] == 1 and data["scenes"][0]["selected_asset"] is not None


@pytest.mark.asyncio
async def test_api_matches_raw_script(async_client):
    response = await async_client.post("/api/v1/broll/match", json={"script": {
        "text": "Whales swim in the ocean. Coral reefs are colorful.", "scene_hints": [
            {"visual_hint": "whale underwater"}, {"visual_hint": "colorful coral reef"}]},
        "target_aspect_ratio": "9:16", "provider": "mock"})
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["total_scenes"] == 2 and data["matched_scenes"] == 2
    assert all(s["selected_asset"]["aspect_ratio"] == "9:16" for s in data["scenes"])


@pytest.mark.asyncio
async def test_api_rejects_missing_source(async_client):
    response = await async_client.post("/api/v1/broll/match", json={"target_aspect_ratio": "16:9"})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_api_rejects_invalid_aspect_ratio(async_client):
    response = await async_client.post("/api/v1/broll/match", json={"scenes": [{
        "scene_id": "scene-001", "scene_index": 1, "start_time": 0, "end_time": 2,
        "narration_text": "Sample", "broll_query": "sample"}], "target_aspect_ratio": "4:3"})
    assert response.status_code == 422
