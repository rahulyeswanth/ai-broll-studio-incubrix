from __future__ import annotations

import asyncio
import shutil
import uuid
from pathlib import Path

from .base import StorageAdapter, StorageError, safe_key


class LocalStorageAdapter(StorageAdapter):
    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir).resolve()

    async def check_health(self) -> bool:
        def _probe() -> bool:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            probe = self.base_dir / f".health-{uuid.uuid4().hex}"
            probe.write_text("ok")
            ok = probe.read_text() == "ok"
            probe.unlink()
            return ok
        try:
            return await asyncio.to_thread(_probe)
        except OSError:
            return False

    async def put_file(self, key: str, path: Path, content_type: str = "application/octet-stream") -> str:
        dest = (self.base_dir / safe_key(key)).resolve()
        if self.base_dir not in dest.parents:
            raise StorageError("Storage key escapes the storage directory.")
        try:
            await asyncio.to_thread(lambda: (dest.parent.mkdir(parents=True, exist_ok=True), shutil.copyfile(path, dest)))
        except OSError as exc:
            raise StorageError(f"Local storage write failed: {exc.__class__.__name__}") from exc
        return dest.as_uri()

    async def get_download_url(self, key: str, expires_seconds: int = 3600) -> str:
        return (self.base_dir / safe_key(key)).as_uri()
