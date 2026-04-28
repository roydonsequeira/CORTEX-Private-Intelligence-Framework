"""FastAPI application factory stub — full implementation in Phase 4."""

from typing import Any

from fastapi import FastAPI


def create_app() -> FastAPI:
    """Return the CORTEX FastAPI application. Full implementation in Phase 4."""
    app = FastAPI(
        title="CORTEX",
        description="Private Intelligence Framework — fully local AI agent.",
        version="0.1.0",
    )

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "phase": "0-1-stub"}

    return app
