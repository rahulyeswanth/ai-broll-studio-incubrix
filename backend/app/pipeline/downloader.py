"""Safe footage downloader: HTTPS + host allow-list, no redirects, size cap, ffprobe validation."""
from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from .config import PipelineSettings
from .errors import DownloadError
from .media import decode_check, video_info


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):  # noqa: D401
        return None


def download_clip(url: str, dest: Path, settings: PipelineSettings) -> Path:
    u = urlparse(url)
    if u.username or u.password:
        raise DownloadError("Download URL must not contain credentials.")
    if settings.require_https:
        if u.scheme != "https":
            raise DownloadError("Only HTTPS footage downloads are allowed.")
        if settings.allowed_download_hosts and (u.hostname or "").lower() not in settings.allowed_download_hosts:
            raise DownloadError(f"Host '{u.hostname}' is not on the download allow-list.")
    elif u.scheme not in ("http", "https"):
        raise DownloadError("Unsupported URL scheme.")
    limit = settings.max_download_mb * 1024 * 1024
    opener = urllib.request.build_opener(_NoRedirect)
    req = urllib.request.Request(url, headers={"User-Agent": "ai-broll-studio/1.0"})
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    written = 0
    try:
        with opener.open(req, timeout=settings.download_timeout_seconds) as resp:
            declared = resp.headers.get("Content-Length")
            if declared and declared.isdigit() and int(declared) > limit:
                raise DownloadError(f"Clip is {int(declared)/1048576:.0f} MB; limit is {settings.max_download_mb} MB.")
            with tmp.open("wb") as fh:
                while True:
                    chunk = resp.read(64 * 1024)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > limit:
                        raise DownloadError(f"Download exceeded the {settings.max_download_mb} MB limit.")
                    fh.write(chunk)
    except DownloadError:
        tmp.unlink(missing_ok=True)
        raise
    except urllib.error.HTTPError as exc:
        tmp.unlink(missing_ok=True)
        raise DownloadError(f"Footage server returned HTTP {exc.code}.") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        tmp.unlink(missing_ok=True)
        raise DownloadError(f"Could not download footage: {exc}") from exc
    if written == 0:
        tmp.unlink(missing_ok=True)
        raise DownloadError("Footage server returned an empty file.")
    tmp.replace(dest)
    try:
        info = video_info(dest)
    except ValueError as exc:
        dest.unlink(missing_ok=True)
        raise DownloadError(f"Downloaded file is not a valid video: {exc}") from exc
    if info["width"] <= 0 or info["height"] <= 0 or info["duration"] <= 0:
        dest.unlink(missing_ok=True)
        raise DownloadError("Downloaded video has invalid dimensions or zero duration.")
    err = decode_check(dest)
    if err:
        dest.unlink(missing_ok=True)
        raise DownloadError(f"Downloaded video is corrupted: {err}")
    return dest
