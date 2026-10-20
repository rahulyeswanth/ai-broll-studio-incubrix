"""Multi-source footage search with a written, measurable provider-selection rule.

Rule (also stored verbatim in every timeline scene as ``provider_selection.rule``):
  1. Providers are tried in priority order (primary first, then the fallbacks that have an API key).
  2. The next provider is queried for the same query when the previous one returned fewer than ``min_results``
     assets (default 3) or raised an error.
  3. All returned assets are merged; the scene's clip is then the highest relevance score regardless of provider,
     ties going to the earlier provider in the priority order.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

log = logging.getLogger(__name__)

RULE_TEXT = ("Query providers in priority order; query the next provider for the same search when the previous one "
             "returned fewer than {n} assets or failed; merge all assets and keep the highest relevance score "
             "(ties go to the earlier provider).")


class MultiProviderSearch:
    def __init__(self, providers: Sequence[Tuple[str, Any]], min_results: int = 3):
        self.providers = list(providers)
        self.min_results = max(1, int(min_results))
        self.log: Dict[str, Dict[str, Any]] = {}      # normalised query -> what each provider returned

    @property
    def names(self) -> List[str]:
        return [n for n, _ in self.providers]

    @property
    def rule(self) -> str:
        return RULE_TEXT.format(n=self.min_results)

    async def __call__(self, query: str, aspect: str, n: int) -> List[Any]:
        key = " ".join(query.lower().split())
        entry = self.log.setdefault(key, {"query": query, "counts": {}, "errors": {}, "fallback_used": False})
        merged: List[Any] = []
        last_error: Optional[Exception] = None
        for i, (name, prov) in enumerate(self.providers):
            if i > 0 and len(merged) >= self.min_results:
                break
            if i > 0:
                entry["fallback_used"] = True
            try:
                got = list(await prov.search_footage(query=query, aspect_ratio=aspect, max_results=n))
            except Exception as exc:  # noqa: BLE001 - one provider failing must not stop the others
                entry["errors"][name] = f"{type(exc).__name__}"
                last_error = exc
                log.warning("Provider %s failed for %r: %s", name, query, type(exc).__name__)
                continue
            entry["counts"][name] = len(got)
            merged.extend(got)
        if not merged and last_error is not None and len(entry["errors"]) == len(self.providers):
            raise last_error                      # every provider failed: surface it so the matcher records the error
        return merged

    def scene_summary(self, queries: Sequence[str], chosen_provider: Optional[str]) -> Dict[str, Any]:
        per_query = []
        for q in queries:
            e = self.log.get(" ".join(q.lower().split()))
            if e:
                per_query.append({"query": e["query"], "returned_by_provider": e["counts"], "provider_errors": e["errors"],
                                  "fallback_provider_queried": e["fallback_used"]})
        why = None
        if chosen_provider:
            why = (f"'{chosen_provider}' supplied the highest-scoring acceptable clip among the merged results of "
                   f"{', '.join(self.names)}.")
        return {"rule": self.rule, "providers_in_priority_order": self.names, "queries": per_query,
                "chosen_provider": chosen_provider, "chosen_because": why}
