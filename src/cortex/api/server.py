"""FastAPI application factory for CORTEX."""

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from typing import Any, cast

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from cortex.agent.kernel import AgentKernel
from cortex.agent.supervisor import SupervisorAgent
from cortex.api.middleware.auth import APIKeyMiddleware
from cortex.api.middleware.rate_limit import RateLimitMiddleware
from cortex.api.middleware.telemetry import telemetry_middleware
from cortex.api.routes import chat, health, memory, tasks, tools
from cortex.api.task_store import create_task_store
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
from cortex.observability.tracing import setup_tracing, shutdown_tracing
from cortex.tools.builtin import register_builtin_tools
from cortex.tools.registry import ToolRegistry

logger = structlog.get_logger(__name__)


def create_app() -> FastAPI:
    """Return the CORTEX FastAPI application."""
    app = FastAPI(
        title="CORTEX",
        description="Private Intelligence Framework — fully local AI agent.",
        version="1.0.0",
        lifespan=_lifespan,
    )
    settings = get_settings()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(
        RateLimitMiddleware,
        enabled=settings.rate_limit.enabled,
        requests_per_minute=settings.rate_limit.requests_per_minute,
        chat_requests_per_minute=settings.rate_limit.chat_requests_per_minute,
    )
    app.add_middleware(APIKeyMiddleware, api_key=settings.api_key)
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
    setup_tracing("cortex-api", settings.otel_endpoint, enabled=settings.telemetry_enabled)
    setup_metrics(settings.otel_endpoint if settings.telemetry_enabled else None)

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
    supervisor = SupervisorAgent(
        lambda: AgentKernel(router, tool_registry, memory_manager, settings),
        router,
        max_workers=settings.supervisor.max_workers,
    )

    app.state.settings = settings
    app.state.ollama_provider = provider
    app.state.model_router = router
    app.state.memory_manager = memory_manager
    app.state.episodic_memory = episodic
    app.state.semantic_memory = semantic
    app.state.procedural_memory = procedural
    app.state.tool_registry = tool_registry
    app.state.agent_kernel = kernel
    app.state.supervisor_agent = supervisor
    app.state.started_at = time.monotonic()
    app.state.task_queue = asyncio.Queue()
    task_store = create_task_store(settings)
    await task_store.initialize()
    app.state.task_store = task_store
    app.state.shutting_down = False
    app.state.in_flight_runs = 0
    app.state.task_worker = asyncio.create_task(_task_worker(app))

    try:
        yield
    finally:
        logger.info("CORTEX shutting down gracefully")
        app.state.shutting_down = True
        await _wait_for_in_flight(app)
        app.state.task_worker.cancel()
        with suppress(asyncio.CancelledError):
            await app.state.task_worker
        await provider.aclose()
        shutdown_tracing()


async def _task_worker(app: FastAPI) -> None:
    """Background worker for fire-and-forget agent tasks."""
    while True:
        item = await app.state.task_queue.get()
        task_id = item["task_id"]
        await app.state.task_store.set(task_id, {"task_id": task_id, "status": "running"})
        try:
            result = await _run_orchestrated_task(app, item)
            await app.state.task_store.set(
                task_id, {"task_id": task_id, "status": "done", "result": result}
            )
        except Exception as exc:
            await app.state.task_store.set(
                task_id,
                {"task_id": task_id, "status": "failed", "error": str(exc)},
            )
        finally:
            app.state.task_queue.task_done()


async def _run_orchestrated_task(app: FastAPI, item: dict[str, str]) -> dict[str, Any]:
    """Dispatch a queued task to single-agent, LATS, or supervisor orchestration."""
    orchestration = item.get("orchestration", "single")
    if orchestration == "supervisor":
        result = await app.state.supervisor_agent.run(item["task"], task_session_id(item["task_id"]))
        return cast(dict[str, Any], result.model_dump(mode="json"))
    state = await app.state.agent_kernel.run(
        item["task"],
        session_id=task_session_id(item["task_id"]),
        use_lats=orchestration == "lats",
    )
    return cast(dict[str, Any], state.model_dump(mode="json"))


def task_session_id(task_id: str) -> str:
    """Return a deterministic session id for queued task work."""
    return f"task-{task_id}"


async def _wait_for_in_flight(app: FastAPI, timeout_seconds: float = 30.0) -> None:
    """Wait for in-flight agent runs to drain up to a timeout."""
    deadline = time.monotonic() + timeout_seconds
    while getattr(app.state, "in_flight_runs", 0) > 0 and time.monotonic() < deadline:
        await asyncio.sleep(0.1)
