"""Convert provider assets (the project's ``BrollAsset`` or any object/dict with similar attributes) into
the pipeline's neutral ``Candidate``. Uses attribute access only, so it works without Pydantic."""
from __future__ import annotations

import datetime as _dt
import re
from typing import Any, Dict, List, Optional

from .licensing import policy_for
from .models import Candidate


def _get(asset: Any, name: str, default: Any = None) -> Any:
    if isinstance(asset, dict):
        return asset.get(name, default)
    return getattr(asset, name, default)


def candidate_from_asset(asset: Any, *, now: Optional[str] = None) -> Candidate:
    provider = str(_get(asset, "source", "unknown") or "unknown").lower()
    title = str(_get(asset, "title", "") or "")
    tags = _get(asset, "tags", None)
    if not tags:  # Pixabay puts its comma-separated tag string in the title
        tags = [t.strip() for t in re.split(r"[,;]", title) if t.strip()] if "," in title else []
    pol = policy_for(provider)
    raw_flag = _get(asset, "commercial_use_allowed", None)
    # Commercial status comes from a documented provider policy (or an explicit per-clip denial), not from
    # the legacy default flag that every BrollAsset carries.
    if raw_flag is False:
        commercial: Optional[bool] = False
    elif pol:
        commercial = pol["commercial_use"]
    else:
        commercial = None
    return Candidate(
        provider=provider, clip_id=str(_get(asset, "id", "")), download_url=str(_get(asset, "download_url", "") or ""),
        title=title, tags=list(tags), page_url=_get(asset, "page_url"), preview_url=_get(asset, "preview_url"),
        duration=float(_get(asset, "duration", 0) or 0), width=_get(asset, "width"), height=_get(asset, "height"),
        creator=_get(asset, "author_name"), creator_url=_get(asset, "author_url"),
        license_name=(pol or {}).get("license_name") or _get(asset, "license_type") if pol else None,
        license_url=(pol or {}).get("license_url"),
        license_basis="provider_policy_documented" if pol else "unknown",
        commercial_use=commercial,
        attribution_required=(pol or {}).get("attribution_required") if pol else None,
        ai_generated=_get(asset, "ai_generated", None),
        raw_commercial_flag=raw_flag,
        retrieved_at=now or _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    )
