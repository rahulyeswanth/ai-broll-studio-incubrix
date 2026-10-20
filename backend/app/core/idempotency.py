"""Generic Idempotency-Key support for every POST (pure ASGI middleware).

Scope of a key = API key (or anonymous) + request path. Same key + same body replays the stored 2xx response
(header ``Idempotent-Replay: true``); same key + different body -> 409 IDEMPOTENCY_CONFLICT; same key while the first
request is still running -> 409 IDEMPOTENCY_CONFLICT (retryable). Non-2xx responses are not stored, so a failed
request can be retried with the same key. Records live as small JSON files under <jobs_dir>/_idempotency/.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from starlette.responses import JSONResponse

from app.core.errors import envelope

log = logging.getLogger(__name__)
KEY_MAX = 128
TTL_SECONDS = 24 * 3600
_lock = threading.Lock()
_inflight: set = set()


def _root() -> Path:
    from app.api.v1.endpoints.jobs import _services      # lazy: the jobs router imports app.core
    root = _services()[1].root / "_idempotency"
    root.mkdir(parents=True, exist_ok=True)
    return root


class IdempotencyMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
        key = headers.get("idempotency-key")
        if key is None:
            return await self.app(scope, receive, send)
        if not (1 <= len(key) <= KEY_MAX):
            return await self._reject(scope, receive, send, 422, "INVALID_REQUEST", "Idempotency-Key must be 1-128 characters.")

        chunks = []
        while True:
            msg = await receive()
            if msg["type"] == "http.disconnect":
                return
            chunks.append(msg.get("body", b""))
            if not msg.get("more_body"):
                break
        body = b"".join(chunks)
        body_hash = hashlib.sha256(headers.get("content-type", "").encode() + b"\0" + body).hexdigest()
        scope_id = hashlib.sha256(f"{headers.get('x-api-key', '-')}|{scope['path']}|{key}".encode()).hexdigest()
        rec_path = _root() / f"{scope_id}.json"

        with _lock:
            rec = _read(rec_path)
            if rec is None and scope_id in _inflight:
                rec = {"hash": body_hash, "pending": True}
            if rec is None:
                _inflight.add(scope_id)
        if rec is not None:
            if rec["hash"] != body_hash:
                return await self._reject(scope, receive, send, 409, "IDEMPOTENCY_CONFLICT",
                                          "This Idempotency-Key was already used with a different request.")
            if rec.get("pending"):
                return await self._reject(scope, receive, send, 409, "IDEMPOTENCY_CONFLICT",
                                          "A request with this Idempotency-Key is still being processed.", retryable=True)
            replay_body = {**rec["body"], "idempotent_replay": True} if isinstance(rec["body"], dict) else rec["body"]
            resp = JSONResponse(replay_body, status_code=rec["status"],
                                headers={**rec.get("headers", {}), "Idempotent-Replay": "true"})
            return await resp(scope, receive, send)

        sent = False

        async def replay_receive():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        captured: Dict[str, Any] = {"status": 0, "headers": [], "body": b""}

        async def capture_send(message):
            if message["type"] == "http.response.start":
                captured["status"] = message["status"]
                captured["headers"] = message.get("headers", [])
            elif message["type"] == "http.response.body":
                captured["body"] += message.get("body", b"")
            await send(message)

        try:
            await self.app(scope, replay_receive, capture_send)
            if 200 <= captured["status"] < 300:
                try:
                    keep = {k.decode(): v.decode() for k, v in captured["headers"]
                            if k.decode().lower() in ("location", "content-type")}
                    keep.pop("content-type", None)
                    rec_path.write_text(json.dumps({"hash": body_hash, "status": captured["status"], "ts": time.time(),
                                                    "headers": keep, "body": json.loads(captured["body"] or b"null")}),
                                        encoding="utf-8")
                except (ValueError, OSError) as exc:
                    log.warning("idempotency record not stored: %s", type(exc).__name__)
        finally:
            with _lock:
                _inflight.discard(scope_id)

    async def _reject(self, scope, receive, send, status: int, code: str, message: str, retryable: Optional[bool] = None):
        resp = JSONResponse(envelope(code, message, retryable=retryable), status_code=status)
        return await resp(scope, receive, send)


def _read(path: Path) -> Optional[Dict[str, Any]]:
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if time.time() - rec.get("ts", 0) > TTL_SECONDS:
        try:
            path.unlink()
        except OSError:
            pass
        return None
    return rec
