import logging
from fastapi import APIRouter, HTTPException, status
from app.schemas.audio import NarrationRequest, NarrationResponse
from app.services.narration import NarrationService

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/narrate",
    response_model=NarrationResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate narration plan and synchronize scene timings",
    description="Segments a script, synthesizes narration through the selected provider, and creates a continuous scene timeline.",
)
async def narrate_script(payload: NarrationRequest) -> NarrationResponse:
    try:
        return await NarrationService().create_narration(payload)
    except Exception as exc:
        from app.pipeline.errors import PipelineError
        if isinstance(exc, PipelineError):
            raise HTTPException(status_code=exc.http_status, detail=exc.to_dict()) from exc
        if isinstance(exc, ValueError):
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
        logger.exception("Narration generation failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create narration.",
        ) from exc
