"""Chat API routes, including Server-Sent Events streaming."""

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any, Literal, cast
from uuid import UUID

import structlog
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    """Chat request payload for streaming and non-streaming modes."""

    message: str = Field(min_length=1, max_length=8192)
    session_id: UUID | None = None
    model_capability: Literal["FAST", "REASONING"] | None = None


@router.post("/message")
async def chat_message(payload: ChatRequest, request: Request) -> dict[str, Any]:
    """Run the agent and return the completed AgentState."""
    _reject_if_shutting_down(request)
    await _increment_in_flight(request)
    try:
        state = await request.app.state.agent_kernel.run(
            payload.message, session_id=str(payload.session_id) if payload.session_id else None
        )
        return cast(dict[str, Any], state.model_dump(mode="json"))
    finally:
        await _decrement_in_flight(request)


@router.post("/stream")
async def chat_stream(payload: ChatRequest, request: Request) -> StreamingResponse:
    """Stream agent events as SSE.

    SSE is deliberately one-way HTTP: simpler than WebSockets, reconnectable by
    browsers, and exactly suited to agent-to-client progress events.
    """
    _reject_if_shutting_down(request)
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    async def run_agent() -> None:
        await _increment_in_flight(request)
        try:
            await request.app.state.agent_kernel.run(
                payload.message,
                session_id=str(payload.session_id) if payload.session_id else None,
                event_queue=queue,
            )
        except Exception as exc:
            logger.exception("chat_stream_failed", error=str(exc))
            await queue.put({"type": "error", "message": str(exc)})
        finally:
            await _decrement_in_flight(request)
            await queue.put({"type": "_close"})

    async def event_source() -> AsyncIterator[str]:
        task = asyncio.create_task(run_agent())
        try:
            while True:
                event = await queue.get()
                if event.get("type") == "_close":
                    break
                yield f"data: {json.dumps(event)}\n\n"
        finally:
            if not task.done():
                task.cancel()

    return StreamingResponse(event_source(), media_type="text/event-stream")


@router.get("/sessions/{session_id}/history")
async def session_history(session_id: str, request: Request) -> list[dict[str, Any]]:
    """Return ordered episodic memory history for a session."""
    episodic = getattr(request.app.state, "episodic_memory", None)
    if episodic is None:
        raise HTTPException(status_code=503, detail="Episodic memory is unavailable.")
    history = await episodic.get_session_history(session_id)
    return [cast(dict[str, Any], message.model_dump(mode="json")) for message in history]


def _reject_if_shutting_down(request: Request) -> None:
    """Return 503 for new chat work while shutdown is draining."""
    if getattr(request.app.state, "shutting_down", False):
        raise HTTPException(status_code=503, detail="CORTEX is shutting down.")


async def _increment_in_flight(request: Request) -> None:
    """Track one in-flight agent run."""
    request.app.state.in_flight_runs = getattr(request.app.state, "in_flight_runs", 0) + 1


async def _decrement_in_flight(request: Request) -> None:
    """Release one in-flight agent run."""
    request.app.state.in_flight_runs = max(
        0, getattr(request.app.state, "in_flight_runs", 0) - 1
    )
