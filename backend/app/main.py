from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.core.middleware import CorrelationIdMiddleware
from app.api.v1.router import api_v1_router
from app.api.v1.endpoints import health as health_endpoints
from app.core.errors import install_error_handlers
from app.core.security import RateLimitMiddleware
from app.core.idempotency import IdempotencyMiddleware

settings = get_settings()
setup_logging(settings.LOG_LEVEL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Lifecycle startup hooks
    yield
    # Lifecycle shutdown hooks


app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    description="Automated narration, B-roll matching, captioning, and video synthesis engine.",
    openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
    docs_url=f"{settings.API_V1_PREFIX}/docs",
    redoc_url=f"{settings.API_V1_PREFIX}/redoc",
    lifespan=lifespan,
)

# Middleware
import os

_origins = [o.strip() for o in os.environ.get(
    "CORS_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501").split(",") if o.strip()]
install_error_handlers(app)
app.add_middleware(IdempotencyMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(CorrelationIdMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH", "OPTIONS"],
    allow_headers=["Content-Type", "Idempotency-Key", "X-API-Key", "X-Correlation-Id"],
    expose_headers=["X-Correlation-Id", "Retry-After", "Location"],
)

# API Routes
app.include_router(api_v1_router, prefix=settings.API_V1_PREFIX)

# Liveness/readiness at the service root as well (the /api/v1 copies stay for existing clients).
app.include_router(health_endpoints.router, include_in_schema=True, tags=["System Health"])
