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
    "INVALID_REQUEST": {"status": 422, "retryable": False, "meaning": "Body or parameters failed validation."},
    "INVALID_INPUT": {"status": 422, "retryable": False, "meaning": "Script, scenes or options are not acceptable."},
    "INVALID_AUDIO": {"status": 422, "retryable": False, "meaning": "Audio is unsupported, empty, corrupt, too long or too large."},
    "INVALID_AUDIO_URL": {"status": 422, "retryable": False, "meaning": "audio_url is not a public http(s) URL or could not be fetched."},
    "PAYLOAD_TOO_LARGE": {"status": 413, "retryable": False, "meaning": "Upload exceeds the size limit."},
    "UNAUTHORIZED": {"status": 401, "retryable": False, "meaning": "API key missing or invalid (only when API_KEYS is configured)."},
    "NOT_FOUND": {"status": 404, "retryable": False, "meaning": "Unknown job or file (also returned for jobs owned by another key)."},
    "IDEMPOTENCY_CONFLICT": {"status": 409, "retryable": False, "meaning": "Idempotency-Key reused with a different request body."},
    "RATE_LIMITED": {"status": 429, "retryable": True, "meaning": "Too many requests; honour Retry-After."},
    "TRANSCRIPTION_FAILED": {"status": 502, "retryable": True, "meaning": "Speech-to-text failed."},
    "TRANSCRIPTION_TIMEOUT": {"status": 504, "retryable": True, "meaning": "Speech-to-text exceeded TRANSCRIPTION_TIMEOUT_SECONDS."},
    "TTS_FAILED": {"status": 502, "retryable": True, "meaning": "Narration could not be synthesised."},
    "PROVIDER_UNAVAILABLE": {"status": 503, "retryable": True, "meaning": "A required provider is unreachable or not configured."},
    "FOOTAGE_UNAVAILABLE": {"status": 422, "retryable": False, "meaning": "No acceptable footage and no fallback allowed."},
    "DOWNLOAD_FAILED": {"status": 502, "retryable": True, "meaning": "A footage download failed."},
    "RENDER_FAILED": {"status": 500, "retryable": False, "meaning": "FFmpeg could not render the video."},
    "OUTPUT_VALIDATION_FAILED": {"status": 500, "retryable": False, "meaning": "Rendered output failed automated checks."},
    "INTERNAL_ERROR": {"status": 500, "retryable": False, "meaning": "Unexpected server error; quote the request ID."},
}
_STATUS_CODE = {400: "INVALID_REQUEST", 401: "UNAUTHORIZED", 403: "UNAUTHORIZED", 404: "NOT_FOUND", 405: "INVALID_REQUEST",
                409: "IDEMPOTENCY_CONFLICT", 413: "PAYLOAD_TOO_LARGE", 422: "INVALID_REQUEST", 429: "RATE_LIMITED",
                503: "PROVIDER_UNAVAILABLE"}


def wire_code(code: Optional[str]) -> str:
    """Public error codes are UPPER_SNAKE_CASE; internal pipeline codes are lower-case and mapped here."""
    return (code or "INTERNAL_ERROR").upper()


def public_error(err: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Copy of a stored job error with the code in its public (UPPER_SNAKE_CASE) form."""
    if not err:
        return err
    return {**err, "code": wire_code(err.get("code"))}


def envelope(code: str, message: str, details: Optional[Dict[str, Any]] = None, retryable: Optional[bool] = None,
             **extra: Any) -> Dict[str, Any]:
    code = wire_code(code)
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
            body = envelope(d["code"], d.get("message", "Request failed."), d.get("details"), d.get("retryable"), detail={**d, "code": wire_code(d["code"])})
        else:
            code = _STATUS_CODE.get(exc.status_code, "INVALID_REQUEST" if exc.status_code < 500 else "INTERNAL_ERROR")
            body = envelope(code, str(d), detail=d)
        return JSONResponse(body, status_code=exc.status_code, headers=getattr(exc, "headers", None))

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError):
        errors = [{"field": ".".join(str(x) for x in e.get("loc", ())), "problem": e.get("msg", "")} for e in exc.errors()]
        return JSONResponse(envelope("INVALID_REQUEST", "The request did not pass validation.", {"errors": errors},
                                     detail=errors), status_code=422)

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception):
        log.exception("Unhandled error")
        return JSONResponse(envelope("INTERNAL_ERROR", "Unexpected server error. Quote the request ID to the operator.",
                                     detail="internal_error"), status_code=500)
