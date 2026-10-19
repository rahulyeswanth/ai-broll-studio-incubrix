"""SSRF-safe URL fetching, in-memory rate limiting and API-key based job ownership."""
from __future__ import annotations

import hashlib
import hmac
import ipaddress
import os
import socket
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, Optional
from urllib.parse import urljoin, urlparse

import httpx
from fastapi import Header, HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.core.errors import envelope


class UnsafeURLError(ValueError):
    pass


def _allow_private() -> bool:
    return os.environ.get("ALLOW_PRIVATE_URLS", "false").lower() in ("1", "true", "yes")


def assert_public_url(url: str) -> str:
    """Only http(s) to hosts whose every resolved address is public. (DNS can change between this check and the
    connection; the fetch below re-checks on each redirect hop but does not pin the IP.)"""
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        raise UnsafeURLError("Only http(s) URLs are accepted.")
    if p.username or p.password:
        raise UnsafeURLError("URLs with embedded credentials are not accepted.")
    if _allow_private():
        return url
    try:
        infos = socket.getaddrinfo(p.hostname, p.port or (443 if p.scheme == "https" else 80), proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        raise UnsafeURLError("The host could not be resolved.")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global:
            raise UnsafeURLError("The URL points to a non-public address.")
    return url


def fetch_to_file(url: str, dest: Path, max_bytes: int, timeout: float = 30.0, max_redirects: int = 3) -> int:
    """Stream ``url`` to ``dest`` with a hard byte limit; redirects are followed manually and re-validated."""
    current = assert_public_url(url)
    with httpx.Client(timeout=timeout, follow_redirects=False) as client:
        for _ in range(max_redirects + 1):
            with client.stream("GET", current) as r:
                if r.status_code in (301, 302, 303, 307, 308) and r.headers.get("location"):
                    current = assert_public_url(urljoin(current, r.headers["location"]))
                    continue
                if r.status_code != 200:
                    raise UnsafeURLError(f"The audio URL returned HTTP {r.status_code}.")
                declared = r.headers.get("content-length")
                if declared and declared.isdigit() and int(declared) > max_bytes:
                    raise OverflowError("too large")
                written = 0
                with dest.open("wb") as fh:
                    for chunk in r.iter_bytes(1024 * 256):
                        written += len(chunk)
                        if written > max_bytes:
                            raise OverflowError("too large")
                        fh.write(chunk)
                return written
    raise UnsafeURLError("Too many redirects.")


def sign_body(secret: str, body: bytes) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-client sliding window for job creation (POST .../jobs). RATE_LIMIT_PER_MINUTE=0 disables it.
    Single-process memory only; put a gateway or Redis limiter in front when running several replicas."""

    def __init__(self, app, per_minute: Optional[int] = None):
        super().__init__(app)
        self.per_minute = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "60")) if per_minute is None else per_minute
        self.hits: Dict[str, Deque[float]] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next):
        if self.per_minute > 0 and request.method == "POST" and request.url.path.rstrip("/").endswith("/jobs"):
            key = request.headers.get("x-api-key") or (request.client.host if request.client else "unknown")
            key = hashlib.sha256(key.encode()).hexdigest()
            now, q = time.monotonic(), self.hits[key]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= self.per_minute:
                retry = max(1, int(60 - (now - q[0])))
                return JSONResponse(envelope("rate_limited", "Too many job submissions. Retry later.",
                                             {"retry_after_seconds": retry}), status_code=429,
                                    headers={"Retry-After": str(retry)})
            q.append(now)
        return await call_next(request)


def configured_api_keys() -> set:
    return {k.strip() for k in os.environ.get("API_KEYS", "").split(",") if k.strip()}


def current_owner(x_api_key: Optional[str] = Header(default=None)) -> Optional[str]:
    """None when API_KEYS is not configured (open mode). Otherwise a stable id derived from the key."""
    keys = configured_api_keys()
    if not keys:
        return None
    if not x_api_key or not any(hmac.compare_digest(x_api_key, k) for k in keys):
        raise HTTPException(401, detail={"code": "unauthorized", "message": "A valid X-API-Key header is required.", "details": {}})
    return hashlib.sha256(x_api_key.encode()).hexdigest()[:16]
