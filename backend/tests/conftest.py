import pytest
import pytest_asyncio
import os
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.core.config import Settings, get_settings


@pytest_asyncio.fixture
def test_settings() -> Settings:
    return Settings(
        APP_ENV="test",
        STORAGE_TYPE="local",
        STORAGE_LOCAL_DIR="./test_storage_data",
        REDIS_URL="redis://localhost:6379/15",
    )


@pytest_asyncio.fixture
async def async_client(test_settings: Settings):
    app.dependency_overrides[get_settings] = lambda: test_settings
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client
    app.dependency_overrides.clear()
