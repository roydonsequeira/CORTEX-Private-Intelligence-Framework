"""FastAPI application factory for CORTEX."""

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from typing import Any, cast

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from cortex.agent.kernel import AgentKernel
from cortex.agent.supervisor import SupervisorAgent
from cortex.api.middleware.auth import APIKeyMiddleware
from cortex.api.middleware.rate_limit import RateLimitMiddleware
from cortex.api.middleware.telemetry import telemetry_middleware
from cortex.api.routes import chat, health, memory, tasks, tools
from cortex.api.task_store import create_task_store
from cortex.config import get_settings
from cortex.config.settings import Settings
from cortex.exceptions import CortexModelError
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
    # add_middleware wraps outward, so the last one added runs first. CORS is
    # added last so that 401/429 responses from the inner layers still carry
    # CORS headers — otherwise the browser reports them as opaque network errors.
    app.middleware("http")(telemetry_middleware)
    app.add_middleware(APIKeyMiddleware, api_key=settings.api_key)
    app.add_middleware(
        RateLimitMiddleware,
        enabled=settings.rate_limit.enabled,
        requests_per_minute=settings.rate_limit.requests_per_minute,
        chat_requests_per_minute=settings.rate_limit.chat_requests_per_minute,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_origin_regex=settings.cors_origin_regex,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # Outermost: reject requests addressed to any other host name. A web page on
    # an attacker's domain that re-resolves to 127.0.0.1 (DNS rebinding) is
    # same-origin to itself, so CORS alone would not stop it from driving the agent.
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts)
    app.add_exception_handler(CortexModelError, _model_error_handler)
    app.add_exception_handler(Exception, _unhandled_error_handler)
    app.include_router(chat.router)
    app.include_router(tasks.router)
    app.include_router(memory.router)
    app.include_router(tools.router)
    app.include_router(health.router)
    FastAPIInstrumentor.instrument_app(app)
    return app


async def _model_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Ollama problems are a dependency outage (503), with the actionable message."""
    return JSONResponse(status_code=503, content={"detail": str(exc)})


async def _unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return a JSON error body (not a bare 'Internal Server Error') for any bug."""
    logger.exception("unhandled_error", path=request.url.path, error=str(exc))
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal error: {type(exc).__name__}: {exc}"},
    )


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize singleton runtime services for the process."""
    settings = get_settings()
    configure_logging(settings.log_level)
    setup_tracing("cortex-api", settings.otel_endpoint, enabled=settings.telemetry_enabled)
    setup_metrics(settings.otel_endpoint if settings.telemetry_enabled else None)

    provider = OllamaProvider(
        settings.ollama_base_url,
        timeout=settings.ollama_timeout_seconds,
        num_ctx=settings.ollama_num_ctx,
        keep_alive=settings.ollama_keep_alive,
    )
    await _check_models(provider, settings)

    episodic = EpisodicMemory(settings.db_path)
    semantic = SemanticMemory(
        settings.chroma_path,
        settings.embed_model,
        provider,
        episodic_memory=episodic,
        consolidation_model=settings.ollama_model,
    )
    procedural = ProceduralMemory(settings.chroma_path, settings.embed_model, provider)
    memory_manager = MemoryManager(
        WorkingMemory(),
        episodic,
        semantic,
        procedural,
        consolidation_enabled=settings.memory_consolidation,
        min_relevance=settings.semantic_min_relevance,
    )
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
    app.state.task_queue = asyncio.PriorityQueue()
    task_store = create_task_store(settings)
    await task_store.initialize()
    app.state.task_store = task_store
    app.state.shutting_down = False
    app.state.in_flight_runs = 0
    app.state.task_worker = asyncio.create_task(_task_worker(app))
    warmup = (
        asyncio.create_task(_warm_up(provider, settings))
        if settings.warmup_on_startup
        else None
    )
    logger.info(
        "cortex_ready",
        model=settings.ollama_model,
        tools=[schema.name for schema in tool_registry.list_tools()],
    )

    try:
        yield
    finally:
        logger.info("CORTEX shutting down gracefully")
        app.state.shutting_down = True
        if warmup is not None and not warmup.done():
            warmup.cancel()
        await _wait_for_in_flight(app)
        app.state.task_worker.cancel()
        with suppress(asyncio.CancelledError):
            await app.state.task_worker
        await memory_manager.drain()
        await provider.aclose()
        shutdown_tracing()


async def _check_models(provider: OllamaProvider, settings: Settings) -> None:
    """Log, at startup, exactly which configured models are missing and how to fix it."""
    if not await provider.health_check():
        logger.critical(
            "ollama_unreachable",
            base_url=settings.ollama_base_url,
            fix="Start Ollama (`ollama serve` or the Ollama app), then restart CORTEX.",
        )
        return
    try:
        available = set(await provider.list_models())
    except CortexModelError as exc:
        logger.warning("ollama_list_models_failed", error=str(exc))
        return
    for model in missing_models(settings, available):
        logger.critical("model_not_pulled", model=model, fix=f"ollama pull {model}")


def missing_models(settings: Settings, available: set[str]) -> list[str]:
    """Return configured models that Ollama does not have (":latest" is implied)."""
    names = available | {name.removesuffix(":latest") for name in available}
    configured = dict.fromkeys(
        [
            settings.ollama_model,
            settings.reasoning_model,
            settings.code_model,
            settings.embed_model,
        ]
    )
    return [model for model in configured if model not in names]


async def _warm_up(provider: OllamaProvider, settings: Settings) -> None:
    """Load the chat and embedding models in the background so the first request is fast."""
    started = time.monotonic()
    try:
        await provider.warmup(settings.ollama_model)
        await provider.embed(settings.embed_model, "warm-up")
        logger.info(
            "models_warm",
            model=settings.ollama_model,
            seconds=round(time.monotonic() - started, 1),
        )
    except Exception as exc:  # warm-up is an optimisation, never fatal
        logger.warning("model_warmup_failed", error=str(exc))


async def _task_worker(app: FastAPI) -> None:
    """Background worker for fire-and-forget agent tasks."""
    while True:
        _rank, _seq, item = await app.state.task_queue.get()
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
