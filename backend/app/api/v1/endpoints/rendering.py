
import logging
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

import httpx
from fastapi import APIRouter, HTTPException

from app.providers.footage import get_footage_provider
from app.schemas.rendering import (
    BrollRenderRequest,
    RenderRequest,
    RenderResponse,
)
from app.services.matcher import BrollMatchingService
from app.services.rendering import (
    VideoRenderingError,
    VideoRenderingService,
)
from app.services.segmentation import ScriptSegmentationService

router = APIRouter()
logger = logging.getLogger(__name__)

MAX_VIDEO_DOWNLOAD_BYTES = 100 * 1024 * 1024
ALLOWED_VIDEO_HOSTS = {"cdn.pixabay.com"}


@router.post(
    "/render",
    response_model=RenderResponse,
    summary="Render local video clips into an MP4",
)
def render_video(payload: RenderRequest) -> RenderResponse:
    """Render video clips that are already stored locally."""
    try:
        return VideoRenderingService().render_video(payload)
    except VideoRenderingError as exc:
        message = str(exc)
        status_code = (
            404
            if "File not found" in message
            else 503
            if "must be installed" in message
            else 422
        )
        raise HTTPException(
            status_code=status_code,
            detail=message,
        ) from exc


@router.post(
    "/render-from-broll",
    response_model=RenderResponse,
    summary="Download matched Pixabay clips and render a video",
)
async def render_from_broll(
    payload: BrollRenderRequest,
) -> RenderResponse:
    """
    Match script scenes, download selected Pixabay clips, and render
    them in scene order with optional narration and subtitles.
    """

    provider_name = (payload.provider or "").strip().lower() or None

    try:
        provider = get_footage_provider(provider_name)
    except (ValueError, NotImplementedError) as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    # Create scenes from the supplied script or use supplied scenes.
    if payload.scenes:
        scenes = payload.scenes
    elif payload.script:
        try:
            scenes = ScriptSegmentationService().segment_script(
                script_text=payload.script.text,
                scene_hints=payload.script.scene_hints,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail=str(exc),
            ) from exc
    else:
        raise HTTPException(
            status_code=422,
            detail="Provide either script or scenes.",
        )

    # Match each scene to a footage asset.
    results, warnings = await BrollMatchingService(
        footage_provider=provider
    ).match_scenes(
        scenes=scenes,
        target_aspect_ratio=payload.target_aspect_ratio,
    )

    failed_scenes = [
        result.scene.scene_id
        for result in results
        if (
            result.match_status != "matched"
            or result.scene.selected_asset is None
        )
    ]

    if failed_scenes:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Some scenes could not be matched.",
                "failed_scene_ids": failed_scenes,
                "warnings": warnings,
            },
        )

    # Use the same local storage directory as VideoRenderingService.
    source_dir = VideoRenderingService().source_dir
    source_dir.mkdir(parents=True, exist_ok=True)

    clip_filenames = []
    clip_durations = []

    try:
        async with httpx.AsyncClient(
            timeout=60.0,
            follow_redirects=False,
        ) as client:

            # results preserves the scene order returned by the matcher.
            for result in results:
                asset = result.scene.selected_asset

                if asset is None:
                    raise HTTPException(
                        status_code=422,
                        detail="A matched scene has no selected asset.",
                    )

                # This endpoint intentionally downloads Pixabay CDN URLs only.
                parsed_url = urlparse(asset.download_url)

                if (
                    asset.source != "pixabay"
                    or parsed_url.scheme != "https"
                    or parsed_url.hostname not in ALLOWED_VIDEO_HOSTS
                    or parsed_url.username is not None
                    or parsed_url.password is not None
                ):
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            "Only HTTPS video URLs from cdn.pixabay.com "
                            "are allowed by this download endpoint."
                        ),
                    )

                # Stream the response and enforce a size limit while downloading.
                filename = (
                    f"pixabay_{asset.id}_{uuid4().hex[:8]}.mp4"
                )
                destination = source_dir / filename
                bytes_written = 0

                try:
                    async with client.stream(
                        "GET",
                        asset.download_url,
                    ) as response:
                        response.raise_for_status()

                        content_length = response.headers.get(
                            "content-length"
                        )
                        if content_length:
                            try:
                                declared_size = int(content_length)
                            except ValueError:
                                declared_size = 0

                            if declared_size > MAX_VIDEO_DOWNLOAD_BYTES:
                                raise HTTPException(
                                    status_code=413,
                                    detail=(
                                        "Selected video exceeds the "
                                        "100 MB download limit."
                                    ),
                                )

                        with destination.open("wb") as video_file:
                            async for chunk in response.aiter_bytes(
                                chunk_size=64 * 1024
                            ):
                                bytes_written += len(chunk)

                                if bytes_written > MAX_VIDEO_DOWNLOAD_BYTES:
                                    raise HTTPException(
                                        status_code=413,
                                        detail=(
                                            "Downloaded video exceeds "
                                            "the 100 MB download limit."
                                        ),
                                    )

                                video_file.write(chunk)

                    if bytes_written == 0:
                        raise HTTPException(
                            status_code=502,
                            detail="Pixabay returned an empty video file.",
                        )

                except Exception:
                    # Remove partial downloads if this clip fails.
                    destination.unlink(missing_ok=True)
                    raise

                clip_filenames.append(filename)
                clip_durations.append(
                    result.scene.duration_seconds
                )

    except HTTPException:
        raise
    except httpx.HTTPStatusError as exc:
        logger.warning(
            "Pixabay returned HTTP %s during video download.",
            exc.response.status_code,
        )
        raise HTTPException(
            status_code=502,
            detail=(
                "Pixabay video download failed with HTTP "
                f"{exc.response.status_code}."
            ),
        ) from exc
    except (httpx.HTTPError, OSError) as exc:
        logger.exception("Could not download Pixabay footage.")
        raise HTTPException(
            status_code=502,
            detail=f"Could not download selected footage: {exc}",
        ) from exc

    # Reuse the existing FFmpeg rendering service.
    try:
        render_payload = RenderRequest(
            clips=clip_filenames,
            clip_durations=clip_durations,
            target_aspect_ratio=payload.target_aspect_ratio,
            output_filename=payload.output_filename,
            audio_filename=payload.audio_filename,
            include_audio_from_clips=False,
            subtitle_text=payload.subtitle_text,
        )

        return VideoRenderingService().render_video(render_payload)

    except VideoRenderingError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc
