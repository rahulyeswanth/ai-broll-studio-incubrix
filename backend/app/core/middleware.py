import uuid
import re
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from app.core.logging import correlation_id_ctx

CORRELATION_ID_HEADER = "X-Correlation-Id"
VALID_CORRELATION_ID_REGEX = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        incoming_id = request.headers.get(CORRELATION_ID_HEADER)
        if incoming_id and VALID_CORRELATION_ID_REGEX.match(incoming_id):
            cid = incoming_id
        else:
            cid = str(uuid.uuid4())

        token = correlation_id_ctx.set(cid)
        try:
            response = await call_next(request)
            response.headers[CORRELATION_ID_HEADER] = cid
            return response
        finally:
            correlation_id_ctx.reset(token)
