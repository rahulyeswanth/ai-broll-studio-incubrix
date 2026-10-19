from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class StorageError(RuntimeError):
    """Raised when an object cannot be stored or retrieved (maps to STORAGE_FAILURE)."""


class StorageAdapter(ABC):
    @abstractmethod
    async def check_health(self) -> bool: ...

    @abstractmethod
    async def put_file(self, key: str, path: Path, content_type: str = "application/octet-stream") -> str:
        """Store a local file under ``key``; returns a storage URI."""

    @abstractmethod
    async def get_download_url(self, key: str, expires_seconds: int = 3600) -> str: ...


def safe_key(key: str) -> str:
    """Reject absolute paths and traversal in object keys."""
    parts = [p for p in key.replace("\\", "/").split("/") if p not in ("", ".")]
    if not parts or any(p == ".." for p in parts):
        raise StorageError(f"Unsafe storage key: {key!r}")
    return "/".join(parts)
