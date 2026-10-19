from __future__ import annotations

import asyncio
from datetime import timedelta
from pathlib import Path

from .base import StorageAdapter, StorageError, safe_key


class MinioStorageAdapter(StorageAdapter):
    """S3-compatible storage (MinIO, AWS S3, R2...) via the minio SDK."""

    def __init__(self, endpoint: str, access_key: str, secret_key: str, bucket_name: str, secure: bool = False,
                 public_endpoint: str | None = None):
        from minio import Minio  # local import keeps the dependency optional for local-storage deployments
        self.bucket = bucket_name
        self.client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
        # Presigned URLs embed the host, so a browser-reachable endpoint may differ from the in-network one.
        self._signer = (Minio(public_endpoint, access_key=access_key, secret_key=secret_key, secure=secure,
                              region="us-east-1") if public_endpoint else self.client)

    async def check_health(self) -> bool:
        def _probe() -> bool:
            if not self.client.bucket_exists(self.bucket):
                self.client.make_bucket(self.bucket)
            return True
        try:
            return await asyncio.wait_for(asyncio.to_thread(_probe), timeout=3)
        except Exception:  # noqa: BLE001 - any failure means "not ready"
            return False

    async def put_file(self, key: str, path: Path, content_type: str = "application/octet-stream") -> str:
        key = safe_key(key)
        try:
            await asyncio.to_thread(self.client.fput_object, self.bucket, key, str(path), content_type=content_type)
        except Exception as exc:  # noqa: BLE001
            raise StorageError(f"Object storage upload failed: {exc.__class__.__name__}") from exc
        return f"s3://{self.bucket}/{key}"

    async def get_download_url(self, key: str, expires_seconds: int = 3600) -> str:
        try:
            return await asyncio.to_thread(self._signer.presigned_get_object, self.bucket, safe_key(key),
                                           expires=timedelta(seconds=expires_seconds))
        except Exception as exc:  # noqa: BLE001
            raise StorageError(f"Could not create a download link: {exc.__class__.__name__}") from exc
