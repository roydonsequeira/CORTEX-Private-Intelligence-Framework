"""Health route for the local CORTEX runtime."""

import time
from typing import Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Structured runtime health response."""

    status: Literal["healthy", "degraded", "unhealthy"]
    ollama: bool
    chromadb: bool
    uptime_seconds: float
    model: str | None = None
    missing_models: list[str] = Field(default_factory=list)


@router.get("/health")
async def health(request: Request) -> HealthResponse:
    """Return health for Ollama, Chroma, the configured models, and process uptime.

    ``degraded`` means Ollama is up but a configured model is not pulled (the
    response names it) or no models exist at all.
    """
    from cortex.api.server import missing_models

    provider = request.app.state.ollama_provider
    settings = getattr(request.app.state, "settings", None)
    ollama = await provider.health_check()
    models: list[str] = []
    if ollama:
        try:
            models = await provider.list_models()
        except Exception:
            models = []
    missing = missing_models(settings, set(models)) if ollama and settings else []
    chromadb = getattr(request.app.state.semantic_memory, "_collection", None) is not None
    if not ollama:
        status: Literal["healthy", "degraded", "unhealthy"] = "unhealthy"
    elif not models or missing:
        status = "degraded"
    else:
        status = "healthy"
    return HealthResponse(
        status=status,
        ollama=ollama,
        chromadb=chromadb,
        uptime_seconds=time.monotonic() - request.app.state.started_at,
        model=settings.ollama_model if settings else None,
        missing_models=missing,
    )
