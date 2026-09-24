"""Integration tests for the FastAPI API layer."""

import asyncio
import json
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from cortex.agent.kernel import AgentState
from cortex.api.server import create_app
from cortex.api.task_store import InMemoryTaskStore
from cortex.memory.base import MemoryEntry, MemoryQuery
from cortex.models.provider import Message
from cortex.tools.base import ToolSchema


class _FakeKernel:
    """Deterministic kernel used by API tests."""

    async def run(
        self,
        user_input: str,
        session_id: str | None = None,
        event_queue: asyncio.Queue[dict[str, Any]] | None = None,
        **_: Any,
    ) -> AgentState:
        sid = session_id or "test-session"
        if event_queue is not None:
            await event_queue.put({"type": "session_id", "value": sid})
            await event_queue.put({"type": "plan", "steps": ["answer"]})
            await event_queue.put({"type": "step_start", "step": 1, "description": "answer"})
            await event_queue.put({"type": "token", "value": "ok"})
            await event_queue.put({"type": "done", "steps_taken": 1})
        return AgentState(
            session_id=sid,
            user_input=user_input,
            plan=["answer"],
            steps_taken=1,
            messages=[Message(role="assistant", content="ok")],
            final_answer="ok",
            status="complete",
        )


class _FakeProvider:
    """Fake Ollama provider for health checks."""

    async def health_check(self) -> bool:
        return True

    async def list_models(self) -> list[str]:
        return ["llama3.1:8b"]


class _FakeToolRegistry:
    """Fake tool registry for API tests."""

    def list_tools(self) -> list[ToolSchema]:
        return [
            ToolSchema(
                name="echo",
                description="Echo test tool.",
                parameters={"type": "object", "properties": {}},
            )
        ]


@pytest.fixture()
def app() -> FastAPI:
    """Create an app with mocked runtime state."""
    app = create_app()
    app.state.agent_kernel = _FakeKernel()
    app.state.ollama_provider = _FakeProvider()
    app.state.semantic_memory = type("Semantic", (), {"_collection": object()})()
    app.state.tool_registry = _FakeToolRegistry()
    app.state.started_at = 1.0
    app.state.task_queue = asyncio.PriorityQueue()
    app.state.task_store = InMemoryTaskStore()
    app.state.shutting_down = False
    app.state.in_flight_runs = 0
    return app


@pytest.mark.asyncio
async def test_chat_message_returns_agent_state(app: FastAPI) -> None:
    """POST /chat/message returns a valid AgentState JSON payload."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
        response = await client.post("/chat/message", json={"message": "hello"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "complete"
    assert body["final_answer"] == "ok"


@pytest.mark.asyncio
async def test_health_returns_expected_structure(app: FastAPI) -> None:
    """GET /health returns the Phase 4 health response shape."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert set(response.json()) == {
        "status",
        "ollama",
        "chromadb",
        "uptime_seconds",
        "model",
        "missing_models",
    }


@pytest.mark.asyncio
async def test_tools_returns_registered_schemas(app: FastAPI) -> None:
    """GET /tools returns registered tool schemas."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
        response = await client.get("/tools")

    assert response.status_code == 200
    assert response.json()[0]["name"] == "echo"


@pytest.mark.asyncio
async def test_sse_stream_yields_expected_event_order(app: FastAPI) -> None:
    """POST /chat/stream yields expected SSE event types in order."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
        response = await client.post("/chat/stream", json={"message": "hello"})

    assert response.status_code == 200
    events = []
    for block in response.text.strip().split("\n\n"):
        _, raw = block.split("data: ", 1)
        events.append(json.loads(raw)["type"])
    assert events == ["session_id", "plan", "step_start", "token", "done"]


@pytest.mark.asyncio
async def test_tasks_accept_orchestration_mode(app: FastAPI) -> None:
    """POST /tasks stores requested orchestration mode on queued work."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
        response = await client.post(
            "/tasks",
            json={"task": "research", "priority": "normal", "orchestration": "supervisor"},
        )

    assert response.status_code == 200
    _, _, queued = await app.state.task_queue.get()
    assert queued["orchestration"] == "supervisor"


@pytest.mark.asyncio
async def test_tasks_run_in_priority_order(app: FastAPI) -> None:
    """High-priority tasks are dequeued before normal and low, FIFO within a level."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
        for task, priority in [("a", "low"), ("b", "normal"), ("c", "high"), ("d", "high")]:
            response = await client.post("/tasks", json={"task": task, "priority": priority})
            assert response.status_code == 200

    order = [(await app.state.task_queue.get())[2]["task"] for _ in range(4)]
    assert order == ["c", "d", "b", "a"]


@pytest.mark.asyncio
async def test_chat_rejects_invalid_session_id(app: FastAPI) -> None:
    """Chat requests validate session_id as UUID."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
        response = await client.post(
            "/chat/message",
            json={"message": "hello", "session_id": "not-a-uuid"},
        )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_chat_returns_503_when_shutting_down(app: FastAPI) -> None:
    """New chat work is rejected while graceful shutdown is draining."""
    app.state.shutting_down = True
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
        response = await client.post("/chat/message", json={"message": "hello"})

    assert response.status_code == 503


@pytest.mark.asyncio
async def test_rate_limit_returns_429() -> None:
    """Rate limiter returns 429 with Retry-After when the bucket is empty."""
    from cortex.api.middleware.rate_limit import RateLimitMiddleware

    limited = FastAPI()
    limited.add_middleware(
        RateLimitMiddleware,
        enabled=True,
        requests_per_minute=1,
        chat_requests_per_minute=1,
    )

    @limited.get("/limited")
    async def limited_route() -> dict[str, str]:
        return {"ok": "true"}

    transport = httpx.ASGITransport(app=limited)
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
        first = await client.get("/limited")
        second = await client.get("/limited")

    assert first.status_code == 200
    assert second.status_code == 429
    assert "Retry-After" in second.headers


def _api_key_app() -> FastAPI:
    """A minimal app protected by APIKeyMiddleware with a known key."""
    from cortex.api.middleware.auth import APIKeyMiddleware

    protected = FastAPI()
    protected.add_middleware(APIKeyMiddleware, api_key="s3cret")

    @protected.get("/tools")
    async def tools_route() -> dict[str, str]:
        return {"ok": "true"}

    @protected.get("/health")
    async def health_route() -> dict[str, str]:
        return {"status": "healthy"}

    return protected


@pytest.mark.asyncio
async def test_api_key_rejects_missing_key() -> None:
    """A configured API key returns 401 when the bearer token is absent."""
    transport = httpx.ASGITransport(app=_api_key_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
        response = await client.get("/tools")

    assert response.status_code == 401
    assert response.headers.get("WWW-Authenticate") == "Bearer"


@pytest.mark.asyncio
async def test_api_key_accepts_valid_key() -> None:
    """A correct bearer token is accepted on a protected route."""
    transport = httpx.ASGITransport(app=_api_key_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
        response = await client.get("/tools", headers={"Authorization": "Bearer s3cret"})

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_api_key_health_is_always_open() -> None:
    """/health never requires a key even when one is configured."""
    transport = httpx.ASGITransport(app=_api_key_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
        response = await client.get("/health")

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_api_key_disabled_when_unset(app: FastAPI) -> None:
    """With no key configured (localhost default), routes stay open."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
        response = await client.get("/tools")

    assert response.status_code == 200


def test_rate_limiter_evicts_idle_buckets() -> None:
    """Buckets idle past the TTL are evicted so the dict cannot grow unbounded."""
    from cortex.api.middleware.rate_limit import _BUCKET_TTL_SECONDS, RateLimitMiddleware

    middleware = RateLimitMiddleware(app=None, requests_per_minute=100)
    middleware._consume("1.1.1.1", "default", 100)
    assert ("1.1.1.1", "default") in middleware._buckets

    # Age the first client's bucket well past the eviction TTL.
    middleware._buckets[("1.1.1.1", "default")].updated_at -= _BUCKET_TTL_SECONDS + 1
    middleware._consume("2.2.2.2", "default", 100)

    assert ("1.1.1.1", "default") not in middleware._buckets
    assert ("2.2.2.2", "default") in middleware._buckets


@pytest.mark.asyncio
async def test_rejects_foreign_host_header(app: FastAPI) -> None:
    """Requests addressed to another host name are refused (DNS-rebinding defence)."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://evil.example") as client:
        response = await client.get("/tools")

    assert response.status_code == 400


@pytest.mark.asyncio
async def test_cors_allows_only_the_local_ui_origin(app: FastAPI) -> None:
    """A browser page on another origin cannot call the agent; the local UI can."""
    transport = httpx.ASGITransport(app=app)
    preflight = {"Access-Control-Request-Method": "POST"}
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
        ui = await client.options(
            "/chat/message", headers={"Origin": "http://localhost:3000", **preflight}
        )
        evil = await client.options(
            "/chat/message", headers={"Origin": "https://evil.example", **preflight}
        )

    assert ui.headers.get("access-control-allow-origin") == "http://localhost:3000"
    assert "access-control-allow-origin" not in evil.headers


@pytest.mark.asyncio
async def test_cors_allows_the_ui_on_another_local_port(app: FastAPI) -> None:
    """If :3000 is busy the UI runs on :3001; loopback origins on any port work."""
    transport = httpx.ASGITransport(app=app)
    preflight = {"Access-Control-Request-Method": "POST"}
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
        alt_port = await client.options(
            "/chat/stream", headers={"Origin": "http://localhost:3001", **preflight}
        )
        lookalike = await client.options(
            "/chat/stream", headers={"Origin": "http://localhost.evil.example", **preflight}
        )

    assert alt_port.headers.get("access-control-allow-origin") == "http://localhost:3001"
    assert "access-control-allow-origin" not in lookalike.headers


@pytest.mark.asyncio
async def test_direct_tool_endpoint_is_disabled_by_default(app: FastAPI) -> None:
    """POST /tools/{name}/execute bypasses the agent, so it is off unless enabled."""
    from unittest.mock import AsyncMock

    from cortex.config.settings import Settings
    from cortex.tools.base import ToolResult

    execute = AsyncMock(
        return_value=ToolResult(tool_name="echo", success=True, output="hi", execution_time_ms=0.0)
    )
    app.state.tool_registry.execute = execute
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
        disabled = await client.post("/tools/echo/execute", json={"text": "hi"})
        app.state.settings = Settings(debug_tool_endpoint=True)
        enabled = await client.post("/tools/echo/execute", json={"text": "hi"})

    assert disabled.status_code == 404
    execute.assert_awaited_once_with("echo", text="hi")
    assert enabled.json()["output"] == "hi"


@pytest.mark.asyncio
async def test_request_id_header_is_sanitised(app: FastAPI) -> None:
    """A client X-Request-ID is echoed only if it is a short plain token."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
        good = await client.get("/tools", headers={"X-Request-ID": "abc-123"})
        bad = await client.get("/tools", headers={"X-Request-ID": "x" * 200})

    assert good.headers["X-Request-ID"] == "abc-123"
    assert bad.headers["X-Request-ID"] != "x" * 200
    assert len(bad.headers["X-Request-ID"]) == 32


class _FakeMemory:
    def __init__(self, content: str, memory_type: str) -> None:
        self._entry = MemoryEntry(
            id="x",
            content=content,
            metadata={},
            timestamp=datetime.now(UTC),
            memory_type=memory_type,  # type: ignore[arg-type]
        )

    async def retrieve(self, query: MemoryQuery) -> list[MemoryEntry]:
        return [self._entry]


@pytest.mark.asyncio
async def test_memory_search_can_include_procedural_patterns(app: FastAPI) -> None:
    """types=procedural shows learned tool patterns; the default search does not."""
    app.state.episodic_memory = _FakeMemory("hello", "episodic")
    app.state.semantic_memory = _FakeMemory("The user's name is Sam.", "semantic")
    app.state.procedural_memory = _FakeMemory("fibonacci -> python_exec", "procedural")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
        default = await client.get("/memory/search", params={"q": "x"})
        with_patterns = await client.get(
            "/memory/search", params={"q": "x", "types": "procedural"}
        )
    assert {r["memory_type"] for r in default.json()["results"]} == {"episodic", "semantic"}
    assert [r["memory_type"] for r in with_patterns.json()["results"]] == ["procedural"]
