"""Unit tests for SupervisorAgent."""

import asyncio
import time
from collections.abc import Callable
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from cortex.agent.kernel import AgentKernel, AgentState
from cortex.agent.supervisor import SupervisorAgent
from cortex.models.provider import ModelResponse
from cortex.models.router import ModelRouter


def _response(content: str) -> ModelResponse:
    return ModelResponse(
        content=content,
        model="test",
        input_tokens=1,
        output_tokens=1,
        latency_ms=1.0,
        raw={},
    )


class _WorkerKernel:
    """Worker that sleeps briefly so concurrency is observable."""

    async def run(
        self,
        user_input: str,
        session_id: str | None = None,
        _allow_lats: bool = False,
    ) -> AgentState:
        await asyncio.sleep(0.05)
        return AgentState(
            session_id=session_id or "worker",
            user_input=user_input,
            final_answer=f"done: {user_input}",
            status="complete",
        )


@pytest.mark.asyncio
async def test_supervisor_runs_subtasks_in_parallel() -> None:
    """SupervisorAgent runs worker kernels concurrently."""
    router = MagicMock(spec=ModelRouter)
    router.complete = AsyncMock(
        side_effect=[
            _response('["a", "b", "c"]'),
            _response("aggregate"),
        ]
    )
    factory = cast(Callable[[], AgentKernel], lambda: _WorkerKernel())
    supervisor = SupervisorAgent(factory, router, max_workers=3)

    start = time.monotonic()
    result = await supervisor.run("task", "s")
    elapsed = time.monotonic() - start

    assert result.status == "complete"
    assert len(result.sub_states) == 3
    assert elapsed < 0.13


@pytest.mark.asyncio
async def test_supervisor_aggregation_receives_all_results() -> None:
    """SupervisorAgent aggregation is called with every worker result."""
    router = MagicMock(spec=ModelRouter)
    router.complete = AsyncMock(return_value=_response('["a", "b"]'))
    factory = cast(Callable[[], AgentKernel], lambda: _WorkerKernel())
    supervisor = SupervisorAgent(factory, router, max_workers=2)
    supervisor._aggregate = AsyncMock(return_value="final")  # type: ignore[method-assign]

    result = await supervisor.run("task", "s")

    supervisor._aggregate.assert_awaited_once()
    assert supervisor._aggregate.await_args is not None
    passed_states = supervisor._aggregate.await_args.args[0]
    assert len(passed_states) == 2
    assert result.final_answer == "final"
