"""Provider licence policies. Only claims that a provider's *published policy* supports are recorded;
everything else stays ``None``/``unknown``. Per-clip rights are never claimed as verified."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

POLICY_NOTE = ("Summarised from the provider's published licence page when this code was written; "
               "not re-verified at runtime and not a legal opinion. Check the current terms before publishing.")

PROVIDER_POLICIES: Dict[str, Dict[str, Any]] = {
    "pixabay": {
        "license_name": "Pixabay Content License",
        "license_url": "https://pixabay.com/service/license-summary/",
        "commercial_use": True,
        "attribution_required": False,
        "restrictions": [
            "Content must not be sold or distributed unaltered as stand-alone content.",
            "Identifiable people, brands or trademarks in a clip may carry separate rights.",
            "No implication of endorsement by depicted people or brands.",
        ],
    },
    "coverr": {
        "license_name": "Coverr License",
        "license_url": "https://coverr.co/license",
        "commercial_use": True,
        "attribution_required": False,
        "restrictions": [
            "Clips must not be resold or bundled into a service that relies on providing videos.",
            "Clips must not be used to train AI models.",
            "Logos, trademarks and identifiable people in a clip may carry separate rights.",
            "Coverr's API terms ask apps to show a Coverr link/logo; some Coverr videos come from Shutterstock and may carry different terms.",
        ],
    },
    "pexels": {
        "license_name": "Pexels License",
        "license_url": "https://www.pexels.com/license/",
        "commercial_use": True,
        "attribution_required": False,
        "restrictions": [
            "Content must not be sold unaltered.",
            "Identifiable people, brands or trademarks in a clip may carry separate rights.",
        ],
    },
}


def policy_for(provider: str) -> Optional[Dict[str, Any]]:
    return PROVIDER_POLICIES.get((provider or "").lower())


def licence_record(provider: str, *, per_clip_license: Optional[str] = None) -> Dict[str, Any]:
    """Licence block for a selected clip."""
    pol = policy_for(provider)
    if not pol:
        return {"license_name": per_clip_license, "license_url": None, "commercial_use": None,
                "commercial_use_basis": "unknown", "attribution_required": None, "restrictions": [],
                "verified_per_clip": False,
                "note": f"No documented licence policy is configured for provider '{provider}'; treat as unknown."}
    return {"license_name": pol["license_name"], "license_url": pol["license_url"],
            "commercial_use": pol["commercial_use"], "commercial_use_basis": "provider_policy_documented",
            "attribution_required": pol["attribution_required"], "restrictions": list(pol["restrictions"]),
            "verified_per_clip": False, "note": POLICY_NOTE}


def licence_warnings(selected: List[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    unknown = [s for s in selected if s.get("license", {}).get("commercial_use") is None]
    if unknown:
        out.append(f"{len(unknown)} clip(s) have UNKNOWN licence/commercial-use status; do not publish commercially until verified.")
    if any(s.get("license", {}).get("commercial_use") for s in selected):
        out.append("Commercial-use status is based on the provider's published licence policy only; per-clip rights "
                   "(identifiable people, trademarks) are not verified.")
    return out
