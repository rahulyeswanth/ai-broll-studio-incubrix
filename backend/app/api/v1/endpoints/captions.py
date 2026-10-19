from fastapi import APIRouter, HTTPException
from app.schemas.captions import CaptionRequest, CaptionResponse
from app.services.captions import CaptionService

router = APIRouter()

@router.post("/generate", response_model=CaptionResponse, summary="Generate timed captions and SRT/VTT subtitle text")
def generate_captions(payload: CaptionRequest) -> CaptionResponse:
    try:
        return CaptionService().generate(payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
