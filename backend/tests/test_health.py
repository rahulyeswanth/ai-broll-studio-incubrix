import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_endpoint(async_client: AsyncClient):
    response = await async_client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert data["version"] == "1.0.0"


@pytest.mark.asyncio
async def test_readiness_endpoint_when_dependencies_down(async_client: AsyncClient):
    # Local test settings will point to an unreachable Redis port unless Redis is running
    response = await async_client.get("/api/v1/ready")
    assert response.status_code in [200, 503]
    data = response.json()
    assert "ready" in data
    assert "dependencies" in data
    assert "redis" in data["dependencies"]
    assert "storage" in data["dependencies"]
