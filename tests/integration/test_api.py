"""Integration tests for the FastAPI API layer."""

import asyncio
import json
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from cortex.agent.kernel import AgentState
from cortex.api.server import create_app
from cortex.models.provider import Message
from cortex.tools.base import ToolSchema


class _FakeKernel:
    """Deterministic kernel used by API tests."""

    async def run(
        self,
        user_input: str,
        session_id: str | None = None,
        event_queue: asyncio.Queue[dict[str, Any]] | None = None,
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
    app.state.task_queue = asyncio.Queue()
    app.state.task_results = {}
    app.state.shutting_down = False
    app.state.in_flight_runs = 0
    return app


@pytest.mark.asyncio
async def test_chat_message_returns_agent_state(app: FastAPI) -> None:
    """POST /chat/message returns a valid AgentState JSON payload."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/chat/message", json={"message": "hello"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "complete"
    assert body["final_answer"] == "ok"


@pytest.mark.asyncio
async def test_health_returns_expected_structure(app: FastAPI) -> None:
    """GET /health returns the Phase 4 health response shape."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert set(response.json()) == {"status", "ollama", "chromadb", "uptime_seconds"}


@pytest.mark.asyncio
async def test_tools_returns_registered_schemas(app: FastAPI) -> None:
    """GET /tools returns registered tool schemas."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/tools")

    assert response.status_code == 200
    assert response.json()[0]["name"] == "echo"


@pytest.mark.asyncio
async def test_sse_stream_yields_expected_event_order(app: FastAPI) -> None:
    """POST /chat/stream yields expected SSE event types in order."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
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
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/tasks",
            json={"task": "research", "priority": "normal", "orchestration": "supervisor"},
        )

    assert response.status_code == 200
    queued = await app.state.task_queue.get()
    assert queued["orchestration"] == "supervisor"


@pytest.mark.asyncio
async def test_chat_rejects_invalid_session_id(app: FastAPI) -> None:
    """Chat requests validate session_id as UUID."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
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
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
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
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        first = await client.get("/limited")
        second = await client.get("/limited")

    assert first.status_code == 200
    assert second.status_code == 429
    assert "Retry-After" in second.headers


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
