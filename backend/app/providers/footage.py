
from app.core.config import get_settings
from app.providers.base import FootageProvider
from app.providers.mocks import MockFootageProvider
from app.providers.pexels import PexelsFootageProvider
from app.providers.pixabay import PixabayFootageProvider


def get_footage_provider(
    provider_name: str | None = None,
    strict_orientation: bool = True,
) -> FootageProvider:
    """Create the configured footage provider."""

    settings = get_settings()
    name = (provider_name or settings.FOOTAGE_PROVIDER).lower()

    if name == "mock":
        return MockFootageProvider()

    if name == "pexels":
        if not settings.PEXELS_API_KEY:
            raise ValueError("Pexels API key is not configured. Set PEXELS_API_KEY in backend/.env.")
        return PexelsFootageProvider(
            api_key=settings.PEXELS_API_KEY
        )

    if name == "pixabay":
        if not settings.PIXABAY_API_KEY:
            raise ValueError("Pixabay API key is not configured. Set PIXABAY_API_KEY in backend/.env.")
        return PixabayFootageProvider(
            api_key=settings.PIXABAY_API_KEY,
            strict_orientation=strict_orientation,
        )

    raise ValueError(
        f"Unsupported footage provider: {name}. "
        "Choose mock, pexels, or pixabay."
    )
