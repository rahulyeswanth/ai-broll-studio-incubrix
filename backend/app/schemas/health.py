from typing import Dict, Literal
from app.schemas.common import BaseSchema


class HealthResponse(BaseSchema):
    status: Literal["healthy", "degraded", "unhealthy"]
    timestamp: str
    version: str


class DependencyStatus(BaseSchema):
    status: Literal["up", "down", "not_configured"]
    details: str | None = None
    required: bool = True


class ReadinessResponse(BaseSchema):
    ready: bool
    status: Literal["ready", "not_ready"]
    timestamp: str
    dependencies: Dict[str, DependencyStatus]
