"""Builds the production dependency set (real providers chosen from settings)."""
from __future__ import annotations

from typing import Any, Optional, Sequence

from .config import PipelineSettings
from .errors import ProviderUnavailableError
from .narration import get_pipeline_tts
from .orchestrator import Deps, JobRequest
from .transcription import get_pipeline_transcriber
from .visual import get_visual_verifier


def build_deps(settings: PipelineSettings, req: JobRequest) -> Deps:
    name = (req.footage_provider or settings.footage_provider or "").lower()
    if name == "mock":
        raise ProviderUnavailableError(
            "The 'mock' footage provider returns placeholder URLs that cannot be downloaded or rendered. "
            "Use FOOTAGE_PROVIDER=pixabay (or pexels) with its API key.")
    try:
        from app.providers.footage import get_footage_provider
        provider = get_footage_provider(name, strict_orientation=False)
    except (ValueError, NotImplementedError) as exc:
        raise ProviderUnavailableError(str(exc)) from exc

    async def search(query: str, aspect: str, n: int) -> Sequence[Any]:
        return await provider.search_footage(query=query, aspect_ratio=aspect, max_results=n)

    try:
        verifier = get_visual_verifier(settings.visual_verifier, settings)
    except ImportError as exc:
        raise ProviderUnavailableError(str(exc)) from exc
    return Deps(settings=settings, search=search, footage_provider_name=name,
                get_tts=lambda: get_pipeline_tts(settings.tts_provider, settings),
                get_transcriber=lambda: get_pipeline_transcriber(settings.transcription_provider, settings),
                verifier=verifier)
