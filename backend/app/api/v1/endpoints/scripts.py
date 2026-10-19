import logging
from fastapi import APIRouter, HTTPException, status
from app.schemas.domain import ScriptInput, ScriptIngestionResponse
from app.services.segmentation import ScriptSegmentationService

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/ingest",
    response_model=ScriptIngestionResponse,
    status_code=status.HTTP_200_OK,
    summary="Ingest and segment a narration script",
    description="Validates script input, divides narration into ordered scenes, calculates duration, and generates B-roll search queries.",
)
async def ingest_script(payload: ScriptInput) -> ScriptIngestionResponse:
    try:
        service = ScriptSegmentationService()
        scenes = service.segment_script(
            script_text=payload.text,
            scene_hints=payload.scene_hints,
        )

        total_duration = scenes[-1].end_time if scenes else 0.0

        return ScriptIngestionResponse(
            script_text=payload.text,
            target_aspect_ratio=payload.target_aspect_ratio,
            estimated_total_duration=round(total_duration, 2),
            total_scenes=len(scenes),
            scenes=scenes,
        )
    except ValueError as exc:
        logger.warning(f"Validation error during script ingestion: {str(exc)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(f"Unexpected error during script ingestion: {str(exc)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process and segment script.",
        )
