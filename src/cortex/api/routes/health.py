"""Health route for the local CORTEX runtime."""

import time
from typing import Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Structured runtime health response."""

    status: Literal["healthy", "degraded", "unhealthy"]
    ollama: bool
    chromadb: bool
    uptime_seconds: float


@router.get("/health")
async def health(request: Request) -> HealthResponse:
    """Return health for Ollama, Chroma, and process uptime."""
    provider = request.app.state.ollama_provider
    ollama = await provider.health_check()
    models: list[str] = []
    if ollama:
        try:
            models = await provider.list_models()
        except Exception:
            models = []
    chromadb = getattr(request.app.state.semantic_memory, "_collection", None) is not None
    if not ollama:
        status: Literal["healthy", "degraded", "unhealthy"] = "unhealthy"
    elif not models:
        status = "degraded"
    else:
        status = "healthy"
    return HealthResponse(
        status=status,
        ollama=ollama,
        chromadb=chromadb,
        uptime_seconds=time.monotonic() - request.app.state.started_at,
    )
