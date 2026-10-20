import logging
import re
import time
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from app.core.logging import correlation_id_ctx, job_id_ctx

CORRELATION_ID_HEADER = "X-Correlation-Id"
VALID_CORRELATION_ID_REGEX = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_JOB_PATH = re.compile(r"/jobs/([A-Za-z0-9_-]{6,64})")
access_log = logging.getLogger("app.access")


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Assigns/propagates X-Correlation-Id, binds job_id from /jobs/{id} paths, and writes one JSON access-log line."""

    async def dispatch(self, request: Request, call_next) -> Response:
        incoming_id = request.headers.get(CORRELATION_ID_HEADER)
        cid = incoming_id if incoming_id and VALID_CORRELATION_ID_REGEX.match(incoming_id) else str(uuid.uuid4())
        m = _JOB_PATH.search(request.url.path)
        token = correlation_id_ctx.set(cid)
        jtoken = job_id_ctx.set(m.group(1) if m else "")
        started = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers[CORRELATION_ID_HEADER] = cid
            return response
        finally:
            if not request.url.path.endswith(("/health", "/ready")):
                access_log.info("%s %s -> %s in %.0f ms", request.method, request.url.path, status,
                                (time.perf_counter() - started) * 1000)
            job_id_ctx.reset(jtoken)
            correlation_id_ctx.reset(token)
