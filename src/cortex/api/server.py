"""FastAPI application factory for CORTEX."""

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from cortex.agent.kernel import AgentKernel
from cortex.api.middleware.telemetry import telemetry_middleware
from cortex.api.routes import chat, health, memory, tasks, tools
from cortex.config import get_settings
from cortex.memory.episodic import EpisodicMemory
from cortex.memory.manager import MemoryManager
from cortex.memory.procedural import ProceduralMemory
from cortex.memory.semantic import SemanticMemory
from cortex.memory.working import WorkingMemory
from cortex.models.provider import OllamaProvider
from cortex.models.router import ModelRouter
from cortex.observability.logging import configure_logging
from cortex.observability.metrics import setup_metrics
from cortex.observability.tracing import setup_tracing
from cortex.tools.builtin import register_builtin_tools
from cortex.tools.registry import ToolRegistry

logger = structlog.get_logger(__name__)


def create_app() -> FastAPI:
    """Return the CORTEX FastAPI application."""
    app = FastAPI(
        title="CORTEX",
        description="Private Intelligence Framework — fully local AI agent.",
        version="0.1.0",
        lifespan=_lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.middleware("http")(telemetry_middleware)
    app.include_router(chat.router)
    app.include_router(tasks.router)
    app.include_router(memory.router)
    app.include_router(tools.router)
    app.include_router(health.router)
    FastAPIInstrumentor.instrument_app(app)
    return app


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize singleton runtime services for the process."""
    settings = get_settings()
    configure_logging(settings.log_level)
    setup_tracing("cortex-api", settings.otel_endpoint)
    setup_metrics(settings.otel_endpoint)

    provider = OllamaProvider(settings.ollama_base_url)
    if not await provider.health_check():
        logger.critical("ollama_unhealthy_startup", base_url=settings.ollama_base_url)

    episodic = EpisodicMemory(settings.db_path)
    semantic = SemanticMemory(
        settings.chroma_path,
        settings.embed_model,
        provider,
        episodic_memory=episodic,
        consolidation_model=settings.ollama_model,
    )
    procedural = ProceduralMemory(settings.chroma_path, settings.embed_model, provider)
    memory_manager = MemoryManager(WorkingMemory(), episodic, semantic, procedural)
    await memory_manager.initialize()

    router = ModelRouter(provider, settings)
    tool_registry = ToolRegistry(settings)
    register_builtin_tools(tool_registry, settings, semantic)
    tool_registry.auto_discover(settings.plugins_dir)
    kernel = AgentKernel(router, tool_registry, memory_manager, settings)

    app.state.settings = settings
    app.state.ollama_provider = provider
    app.state.model_router = router
    app.state.memory_manager = memory_manager
    app.state.episodic_memory = episodic
    app.state.semantic_memory = semantic
    app.state.procedural_memory = procedural
    app.state.tool_registry = tool_registry
    app.state.agent_kernel = kernel
    app.state.started_at = time.monotonic()
    app.state.task_queue = asyncio.Queue()
    app.state.task_results = {}
    app.state.task_worker = asyncio.create_task(_task_worker(app))

    try:
        yield
    finally:
        app.state.task_worker.cancel()
        with suppress(asyncio.CancelledError):
            await app.state.task_worker
        await provider.aclose()


async def _task_worker(app: FastAPI) -> None:
    """Background worker for fire-and-forget agent tasks."""
    while True:
        item = await app.state.task_queue.get()
        task_id = item["task_id"]
        app.state.task_results[task_id] = {"task_id": task_id, "status": "running"}
        try:
            state = await app.state.agent_kernel.run(item["task"])
            app.state.task_results[task_id] = {
                "task_id": task_id,
                "status": "done",
                "result": state.model_dump(mode="json"),
            }
        except Exception as exc:
            app.state.task_results[task_id] = {
                "task_id": task_id,
                "status": "failed",
                "error": str(exc),
            }
        finally:
            app.state.task_queue.task_done()
