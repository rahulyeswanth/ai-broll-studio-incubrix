import logging
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, status, Response
import redis.asyncio as redis

from app.core.config import Settings, get_settings
from app.schemas.health import HealthResponse, ReadinessResponse, DependencyStatus
from app.storage.minio_adapter import MinioStorageAdapter
from app.storage.local_adapter import LocalStorageAdapter

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness check",
    description="Returns service liveness status.",
)
async def get_health() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc).isoformat(),
        version="1.0.0",
    )


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Readiness check",
    description="Accurately verifies availability of external dependencies (Redis, Storage).",
)
async def get_readiness(
    response: Response,
    settings: Settings = Depends(get_settings),
) -> ReadinessResponse:
    dependencies: dict[str, DependencyStatus] = {}
    is_ready = True

    # 1. Check Redis
    try:
        r = redis.from_url(settings.REDIS_URL, socket_timeout=1.5)
        ping_ok = await r.ping()
        await r.aclose()
        if ping_ok:
            dependencies["redis"] = DependencyStatus(status="up", details="Connected and ping succeeded")
        else:
            dependencies["redis"] = DependencyStatus(status="down", details="Ping failed")
            is_ready = False
    except Exception as exc:
        logger.warning(f"Readiness check failed for Redis: {str(exc)}")
        dependencies["redis"] = DependencyStatus(status="down", details=str(exc))
        is_ready = False

    # 2. Check Storage
    try:
        if settings.STORAGE_TYPE == "minio":
            adapter = MinioStorageAdapter(
                endpoint=settings.MINIO_ENDPOINT,
                access_key=settings.MINIO_ACCESS_KEY,
                secret_key=settings.MINIO_SECRET_KEY,
                bucket_name=settings.MINIO_BUCKET_NAME,
                secure=settings.MINIO_SECURE,
            )
            storage_ok = await adapter.check_health()
        else:
            adapter = LocalStorageAdapter(base_dir=settings.STORAGE_LOCAL_DIR)
            storage_ok = await adapter.check_health()

        if storage_ok:
            dependencies["storage"] = DependencyStatus(
                status="up", details=f"Storage type '{settings.STORAGE_TYPE}' verified"
            )
        else:
            dependencies["storage"] = DependencyStatus(status="down", details="Storage adapter inaccessible")
            is_ready = False
    except Exception as exc:
        logger.warning(f"Readiness check failed for Storage: {str(exc)}")
        dependencies["storage"] = DependencyStatus(status="down", details=str(exc))
        is_ready = False

    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return ReadinessResponse(
        ready=is_ready,
        status="ready" if is_ready else "not_ready",
        timestamp=datetime.now(timezone.utc).isoformat(),
        dependencies=dependencies,
    )
