import pytest


@pytest.mark.asyncio
async def test_narration_endpoint_creates_sequential_scene_timeline(async_client):
    response = await async_client.post(
        "/api/v1/audio/narrate",
        json={
            "script": "Coral reefs shelter tropical fish. Oceans support life.",
            "provider": "mock",
            "target_aspect_ratio": "16:9",
        },
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["provider_used"] == "mock"
    assert data["total_scenes"] == 2
    assert data["audio_is_placeholder"] is True
    assert data["total_duration_seconds"] > 0
    first, second = data["scenes"]
    assert first["start_time"] == 0
    assert first["end_time"] == second["start_time"]
    assert second["end_time"] == data["total_duration_seconds"]
    assert data["warnings"]


@pytest.mark.asyncio
async def test_narration_endpoint_rejects_unknown_provider(async_client):
    response = await async_client.post(
        "/api/v1/audio/narrate",
        json={"script": "Hello world.", "provider": "unknown"},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_narration_endpoint_rejects_blank_script(async_client):
    response = await async_client.post(
        "/api/v1/audio/narrate",
        json={"script": "   "},
    )
    assert response.status_code == 422
