"""Standard error envelope: {"error": {"code", "message", "retryable", "details"}}.

The legacy ``detail`` key is kept next to ``error`` so the existing Streamlit frontend keeps working.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.logging import correlation_id_ctx

log = logging.getLogger(__name__)

# Documented in README.md ("Error codes"); tests/test_error_envelope.py checks this table against the README.
ERROR_CODES: Dict[str, Dict[str, Any]] = {
    "invalid_request": {"status": 422, "retryable": False, "meaning": "Body or parameters failed validation."},
    "invalid_input": {"status": 422, "retryable": False, "meaning": "Script, scenes or options are not acceptable."},
    "invalid_audio": {"status": 422, "retryable": False, "meaning": "Audio is unsupported, empty, corrupt, too long or too large."},
    "invalid_audio_url": {"status": 422, "retryable": False, "meaning": "audio_url is not a public http(s) URL or could not be fetched."},
    "payload_too_large": {"status": 413, "retryable": False, "meaning": "Upload exceeds the size limit."},
    "unauthorized": {"status": 401, "retryable": False, "meaning": "API key missing or invalid (only when API_KEYS is configured)."},
    "not_found": {"status": 404, "retryable": False, "meaning": "Unknown job or file (also returned for jobs owned by another key)."},
    "idempotency_conflict": {"status": 409, "retryable": False, "meaning": "Idempotency-Key reused with a different request body."},
    "rate_limited": {"status": 429, "retryable": True, "meaning": "Too many requests; honour Retry-After."},
    "transcription_failed": {"status": 502, "retryable": True, "meaning": "Speech-to-text failed."},
    "transcription_timeout": {"status": 504, "retryable": True, "meaning": "Speech-to-text exceeded TRANSCRIPTION_TIMEOUT_SECONDS."},
    "tts_failed": {"status": 502, "retryable": True, "meaning": "Narration could not be synthesised."},
    "provider_unavailable": {"status": 503, "retryable": True, "meaning": "A required provider is unreachable or not configured."},
    "footage_unavailable": {"status": 422, "retryable": False, "meaning": "No acceptable footage and no fallback allowed."},
    "download_failed": {"status": 502, "retryable": True, "meaning": "A footage download failed."},
    "render_failed": {"status": 500, "retryable": False, "meaning": "FFmpeg could not render the video."},
    "output_validation_failed": {"status": 500, "retryable": False, "meaning": "Rendered output failed automated checks."},
    "internal_error": {"status": 500, "retryable": False, "meaning": "Unexpected server error; quote the request ID."},
}
_STATUS_CODE = {400: "invalid_request", 401: "unauthorized", 403: "unauthorized", 404: "not_found", 405: "invalid_request",
                409: "idempotency_conflict", 413: "payload_too_large", 422: "invalid_request", 429: "rate_limited",
                503: "provider_unavailable"}


def envelope(code: str, message: str, details: Optional[Dict[str, Any]] = None, retryable: Optional[bool] = None,
             **extra: Any) -> Dict[str, Any]:
    if retryable is None:
        retryable = bool(ERROR_CODES.get(code, {}).get("retryable", False))
    body = {"error": {"code": code, "message": message, "retryable": retryable, "details": details or {},
                      "request_id": correlation_id_ctx.get() or None}}
    body.update(extra)
    return body


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException):
        d = exc.detail
        if isinstance(d, dict) and "code" in d:
            body = envelope(d["code"], d.get("message", "Request failed."), d.get("details"), d.get("retryable"), detail=d)
        else:
            code = _STATUS_CODE.get(exc.status_code, "invalid_request" if exc.status_code < 500 else "internal_error")
            body = envelope(code, str(d), detail=d)
        return JSONResponse(body, status_code=exc.status_code, headers=getattr(exc, "headers", None))

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError):
        errors = [{"field": ".".join(str(x) for x in e.get("loc", ())), "problem": e.get("msg", "")} for e in exc.errors()]
        return JSONResponse(envelope("invalid_request", "The request did not pass validation.", {"errors": errors},
                                     detail=errors), status_code=422)

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception):
        log.exception("Unhandled error")
        return JSONResponse(envelope("internal_error", "Unexpected server error. Quote the request ID to the operator.",
                                     detail="internal_error"), status_code=500)
