import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_ingest_script_valid_payload(async_client: AsyncClient):
    payload = {
        "text": "Welcome to AI B-roll Studio. We turn scripts into captivating videos automatically.",
        "target_aspect_ratio": "16:9",
        "scene_hints": [
            {"visual_hint": "futuristic studio camera"},
            {"visual_hint": "video editor timeline workspace"},
        ],
    }
    response = await async_client.post("/api/v1/scripts/ingest", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["target_aspect_ratio"] == "16:9"
    assert data["total_scenes"] == 2
    assert data["estimated_total_duration"] > 0.0

    scenes = data["scenes"]
    assert len(scenes) == 2
    assert scenes[0]["scene_id"] == "scene-001"
    assert scenes[0]["scene_index"] == 1
    assert scenes[0]["narration_text"] == "Welcome to AI B-roll Studio."
    assert scenes[0]["broll_query"] == "futuristic studio camera"

    assert scenes[1]["scene_id"] == "scene-002"
    assert scenes[1]["scene_index"] == 2
    assert scenes[1]["start_time"] == scenes[0]["end_time"]


@pytest.mark.asyncio
async def test_ingest_script_empty_and_whitespace_rejected(async_client: AsyncClient):
    response = await async_client.post("/api/v1/scripts/ingest", json={"text": "   \n\t  "})
    assert response.status_code == 422
    assert "Script text cannot be empty" in response.text


@pytest.mark.asyncio
async def test_ingest_script_invalid_aspect_ratio_rejected(async_client: AsyncClient):
    response = await async_client.post(
        "/api/v1/scripts/ingest",
        json={"text": "A valid script.", "target_aspect_ratio": "21:9"},
    )
    assert response.status_code == 422
    assert "Invalid aspect ratio '21:9'" in response.text


@pytest.mark.asyncio
async def test_ingest_script_exceeding_max_length_rejected(async_client: AsyncClient):
    oversized_text = "word " * 2500  # > 10,000 characters
    response = await async_client.post(
        "/api/v1/scripts/ingest",
        json={"text": oversized_text},
    )
    assert response.status_code == 422
    assert "exceeds maximum limit" in response.text


@pytest.mark.asyncio
async def test_ingest_script_malformed_json(async_client: AsyncClient):
    response = await async_client.post(
        "/api/v1/scripts/ingest",
        content="not a json",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422
