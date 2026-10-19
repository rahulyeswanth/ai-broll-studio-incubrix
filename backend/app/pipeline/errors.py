"""Structured pipeline errors. Every error carries a stable ``code`` and a human message."""
from __future__ import annotations

from typing import Any, Dict, Optional


class PipelineError(Exception):
    code = "pipeline_error"
    http_status = 500
    retryable = False

    def __init__(self, message: str, *, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        return {"code": self.code, "message": self.message, "retryable": self.retryable, "details": self.details}


class InputValidationError(PipelineError):
    code = "invalid_input"
    http_status = 422


class AudioValidationError(InputValidationError):
    code = "invalid_audio"


class TTSError(PipelineError):
    code = "tts_failed"
    http_status = 502
    retryable = True


class TranscriptionError(PipelineError):
    code = "transcription_failed"
    http_status = 502
    retryable = True


class TranscriptionTimeoutError(TranscriptionError):
    code = "transcription_timeout"
    http_status = 504


class ProviderUnavailableError(PipelineError):
    code = "provider_unavailable"
    http_status = 503
    retryable = True


class FootageError(PipelineError):
    code = "footage_unavailable"
    http_status = 422


class DownloadError(PipelineError):
    code = "download_failed"
    http_status = 502
    retryable = True


class RenderError(PipelineError):
    code = "render_failed"
    http_status = 500


class ValidationFailure(PipelineError):
    code = "output_validation_failed"
    http_status = 500
