"""Health route for the local Quietpost runtime."""

import time
from typing import Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Structured runtime health response."""

    status: Literal["healthy", "degraded", "unhealthy"]
    model_server: bool
    # Deprecated: the same as model_server, for clients written for 1.x.
    ollama: bool
    chromadb: bool
    uptime_seconds: float
    model: str | None = None
    missing_models: list[str] = Field(default_factory=list)


@router.get("/health")
async def health(request: Request) -> HealthResponse:
    """Return health for the model server, Chroma, the configured models, and uptime.

    ``degraded`` means the model server is up but a configured model is missing
    (the response names it) or it has no models at all.
    """
    from quietpost.api.server import missing_models

    provider = request.app.state.model_provider
    settings = getattr(request.app.state, "settings", None)
    reachable = await provider.health_check()
    models: list[str] = []
    if reachable:
        try:
            models = await provider.list_models()
        except Exception:
            models = []
    missing = missing_models(settings, set(models)) if reachable and settings else []
    chromadb = getattr(request.app.state.semantic_memory, "_collection", None) is not None
    if not reachable:
        status: Literal["healthy", "degraded", "unhealthy"] = "unhealthy"
    elif not models or missing:
        status = "degraded"
    else:
        status = "healthy"
    return HealthResponse(
        status=status,
        model_server=reachable,
        ollama=reachable,
        chromadb=chromadb,
        uptime_seconds=time.monotonic() - request.app.state.started_at,
        model=settings.chat_model if settings else None,
        missing_models=missing,
    )
