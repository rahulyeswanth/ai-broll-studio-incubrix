from typing import Dict, Any


class BackgroundJobService:
    """Placeholder service for Phase 7 background queue dispatch."""
    async def enqueue_job(self, task_name: str, payload: Dict[str, Any]) -> str:
        raise NotImplementedError("Background job dispatch will be implemented in Phase 7.")
