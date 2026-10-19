import pytest

@pytest.mark.asyncio
async def test_generate_srt_captions_splits_and_formats_timeline(async_client):
    response = await async_client.post("/api/v1/captions/generate", json={
        "scenes": [{"scene_id": "scene-001", "start_time": 0, "end_time": 4, "narration_text": "Colorful coral reefs provide shelter for tropical fish."}],
        "format": "srt", "max_words_per_caption": 4
    })
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["total_captions"] == 2
    assert data["captions"][0]["start_time"] == 0
    assert data["captions"][-1]["end_time"] == 4
    assert "00:00:00,000 -->" in data["subtitle_text"]
    assert "WEBVTT" not in data["subtitle_text"]

@pytest.mark.asyncio
async def test_generate_vtt_has_header_and_dot_milliseconds(async_client):
    response = await async_client.post("/api/v1/captions/generate", json={
        "scenes": [{"scene_id": "scene-001", "start_time": 1.25, "end_time": 3, "narration_text": "Ocean life matters."}],
        "format": "vtt"
    })
    assert response.status_code == 200, response.text
    text = response.json()["subtitle_text"]
    assert text.startswith("WEBVTT\n")
    assert "00:00:01.250 --> 00:00:03.000" in text

@pytest.mark.asyncio
async def test_caption_endpoint_rejects_empty_scenes(async_client):
    response = await async_client.post("/api/v1/captions/generate", json={"scenes": []})
    assert response.status_code == 422
