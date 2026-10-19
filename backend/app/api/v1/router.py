from fastapi import APIRouter
from app.api.v1.endpoints import health, scripts, broll, audio, captions, rendering, jobs

api_v1_router = APIRouter()
api_v1_router.include_router(health.router, tags=["System Health"])
api_v1_router.include_router(scripts.router, prefix="/scripts", tags=["Script Ingestion & Segmentation"])
api_v1_router.include_router(broll.router, prefix="/broll", tags=["B-roll Asset Matching"])

api_v1_router.include_router(audio.router, prefix="/audio", tags=["Narration & Audio Synchronization"])

api_v1_router.include_router(captions.router, prefix="/captions", tags=["Caption Generation & Subtitles"])

api_v1_router.include_router(rendering.router, prefix="/video", tags=["Video Rendering (FFmpeg)"])

api_v1_router.include_router(jobs.router, prefix="/jobs", tags=["One-call Video Generation (script or audio -> MP4)"])
