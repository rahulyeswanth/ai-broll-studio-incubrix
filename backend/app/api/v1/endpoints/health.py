import logging
import os
import shutil
import subprocess
import uuid
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

    # 1. Redis: the job queue is file-based, so Redis is optional unless READINESS_REQUIRE_REDIS=true
    redis_required = os.environ.get("READINESS_REQUIRE_REDIS", "").lower() in ("1", "true", "yes")
    try:
        r = redis.from_url(settings.REDIS_URL, socket_timeout=1.5)
        ping_ok = await r.ping()
        await r.aclose()
        dependencies["redis"] = DependencyStatus(status="up" if ping_ok else "down", required=redis_required,
                                                 details="Connected and ping succeeded" if ping_ok else "Ping failed")
        if not ping_ok and redis_required:
            is_ready = False
    except Exception as exc:
        logger.warning(f"Readiness check failed for Redis: {str(exc)}")
        dependencies["redis"] = DependencyStatus(status="down", required=redis_required, details=type(exc).__name__)
        if redis_required:
            is_ready = False

    # 1b. FFmpeg / FFprobe (rendering and audio validation cannot work without them)
    for tool in ("ffmpeg", "ffprobe"):
        path = shutil.which(os.environ.get(tool.upper() + "_BINARY", tool))
        ok = False
        if path:
            try:
                ok = subprocess.run([path, "-version"], capture_output=True, timeout=5).returncode == 0
            except (OSError, subprocess.SubprocessError):
                ok = False
        dependencies[tool] = DependencyStatus(status="up" if ok else "down", details="runs" if ok else "not found or not executable")
        is_ready = is_ready and ok

    # 1c. The jobs directory must be writable
    try:
        from app.pipeline.config import load_settings
        jd = load_settings().jobs_dir
        jd.mkdir(parents=True, exist_ok=True)
        probe = jd / f".ready_{uuid.uuid4().hex}"
        probe.write_text("ok")
        probe.unlink()
        dependencies["jobs_dir"] = DependencyStatus(status="up", details="writable")
    except Exception as exc:
        logger.warning(f"Readiness check failed for jobs dir: {type(exc).__name__}")
        dependencies["jobs_dir"] = DependencyStatus(status="down", details=type(exc).__name__)
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
