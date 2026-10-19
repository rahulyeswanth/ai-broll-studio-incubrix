import pytest
import httpx
from unittest.mock import patch, MagicMock

from app.providers.mocks import MockFootageProvider
from app.providers.pexels import PexelsFootageProvider
from app.providers.footage import get_footage_provider
from app.core.config import Settings


@pytest.mark.asyncio
async def test_mock_footage_provider_commercial_license():
    provider = MockFootageProvider()
    assets = await provider.search_footage(query="city skyline", aspect_ratio="16:9", max_results=2)
    assert len(assets) == 2
    for asset in assets:
        assert asset.commercial_use_allowed is True
        assert asset.license_type == "CC0-Commercial-Safe"
        assert asset.attribution_required is False
        assert asset.source == "mock"
        assert asset.author_name == "Mock Creator"


def test_pexels_provider_missing_key_raises_error():
    with pytest.raises(ValueError) as exc:
        PexelsFootageProvider(api_key=None)
    assert "Pexels API key is not configured" in str(exc.value)


@pytest.mark.asyncio
async def test_pexels_provider_search_footage_mocked_network():
    mock_payload = {
        "page": 1,
        "per_page": 2,
        "total_results": 10,
        "videos": [
            {
                "id": 855564,
                "url": "https://www.pexels.com/video/drone-shot-of-a-modern-city-855564/",
                "image": "https://images.pexels.com/videos/855564/preview.jpg",
                "duration": 12,
                "user": {"name": "Jane Cinematographer", "url": "https://www.pexels.com/@janecine"},
                "video_files": [
                    {
                        "id": 101,
                        "quality": "hd",
                        "file_type": "video/mp4",
                        "width": 1920,
                        "height": 1080,
                        "link": "https://video.pexels.com/download/855564_hd.mp4",
                    }
                ],
            }
        ],
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_payload

    with patch.object(httpx.AsyncClient, "get", return_value=mock_resp):
        provider = PexelsFootageProvider(api_key="test_fake_api_key")
        assets = await provider.search_footage(query="modern city", aspect_ratio="16:9", max_results=1)

    assert len(assets) == 1
    asset = assets[0]
    assert asset.id == "855564"
    assert asset.source == "pexels"
    assert asset.download_url == "https://video.pexels.com/download/855564_hd.mp4"
    assert asset.commercial_use_allowed is True
    assert asset.license_type == "Pexels License (Commercial Use Allowed)"
    assert asset.attribution_required is False
    assert asset.author_name == "Jane Cinematographer"
    assert asset.author_url == "https://www.pexels.com/@janecine"


@pytest.mark.asyncio
async def test_pexels_provider_handles_auth_failure():
    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_resp.text = "Unauthorized"

    with patch.object(httpx.AsyncClient, "get", return_value=mock_resp):
        provider = PexelsFootageProvider(api_key="invalid_key")
        with pytest.raises(PermissionError) as exc:
            await provider.search_footage(query="technology")
    assert "Pexels API authentication failed" in str(exc.value)


def test_footage_provider_factory_fallback():
    # 1. Mock provider explicitly requested
    provider = get_footage_provider("mock")
    assert isinstance(provider, MockFootageProvider)

    # 2. Pexels requested without key configured raises ValueError
    with patch("app.providers.footage.get_settings", return_value=Settings(PEXELS_API_KEY=None)):
        with pytest.raises(ValueError) as exc:
            get_footage_provider("pexels")
        assert "Pexels API key is not configured" in str(exc.value)

    # 3. Pexels requested with key configured
    with patch("app.providers.footage.get_settings", return_value=Settings(PEXELS_API_KEY="valid_mock_key")):
        provider = get_footage_provider("pexels")
        assert isinstance(provider, PexelsFootageProvider)
