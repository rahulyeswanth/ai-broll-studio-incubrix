import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_correlation_id_generated(async_client: AsyncClient):
    response = await async_client.get("/api/v1/health")
    assert response.status_code == 200
    assert "X-Correlation-Id" in response.headers
    assert len(response.headers["X-Correlation-Id"]) > 0


@pytest.mark.asyncio
async def test_correlation_id_propagated(async_client: AsyncClient):
    custom_cid = "custom-test-trace-id-12345"
    response = await async_client.get(
        "/api/v1/health",
        headers={"X-Correlation-Id": custom_cid},
    )
    assert response.status_code == 200
    assert response.headers["X-Correlation-Id"] == custom_cid
