"""Unit tests for Language Agent Tree Search."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from cortex.agent.kernel import AgentState
from cortex.agent.loop import LATSLoop, SearchNode
from cortex.models.provider import ModelResponse
from cortex.models.router import ModelRouter

SearchNode.model_rebuild(_types_namespace={"AgentState": AgentState})


def _response(content: str) -> ModelResponse:
    return ModelResponse(
        content=content,
        model="test",
        input_tokens=1,
        output_tokens=1,
        latency_ms=1.0,
        raw={},
    )


def _node(value: float, visits: int, parent_id: str | None = None) -> SearchNode:
    return SearchNode(
        id=f"node-{value}-{visits}",
        state=AgentState(session_id="s", user_input="task"),
        parent_id=parent_id,
        value=value,
        visit_count=visits,
        depth=1,
    )


def test_ucb1_select_prefers_high_value_unvisited_node() -> None:
    """UCB1 selection prioritizes high-value unexplored leaf nodes."""
    router = MagicMock(spec=ModelRouter)
    loop = LATSLoop(MagicMock(), router)
    parent = _node(0.1, 5, None)
    parent.id = "parent"
    low = _node(0.2, 2, "parent")
    high_unvisited = _node(0.9, 0, "parent")
    nodes = {parent.id: parent, low.id: low, high_unvisited.id: high_unvisited}

    selected = loop._ucb1_select(nodes)

    assert selected is high_unvisited


@pytest.mark.asyncio
async def test_evaluate_state_returns_score_between_zero_and_one() -> None:
    """_evaluate_state parses model scoring JSON and clamps to [0, 1]."""
    router = MagicMock(spec=ModelRouter)
    router.complete = AsyncMock(
        return_value=_response(
            '{"goal_completion": 0.8, "factual_correctness": 0.6, "conciseness": 0.7}'
        )
    )
    loop = LATSLoop(MagicMock(), router)

    score = await loop._evaluate_state(
        AgentState(
            session_id="s",
            user_input="task",
            final_answer="answer",
            status="complete",
        )
    )

    assert 0.0 <= score <= 1.0


@pytest.mark.asyncio
async def test_evaluate_state_caches_by_state() -> None:
    """Equivalent states are scored once; the cached value is reused."""
    router = MagicMock(spec=ModelRouter)
    router.complete = AsyncMock(
        return_value=_response(
            '{"goal_completion": 0.8, "factual_correctness": 0.6, "conciseness": 0.7}'
        )
    )
    loop = LATSLoop(MagicMock(), router)
    state = AgentState(
        session_id="s", user_input="task", final_answer="answer", status="complete"
    )

    first = await loop._evaluate_state(state)
    second = await loop._evaluate_state(state)

    assert first == second
    router.complete.assert_awaited_once()  # second call served from cache


@pytest.mark.asyncio
async def test_heuristic_evaluator_skips_the_model() -> None:
    """The heuristic evaluator scores without any LLM call."""
    router = MagicMock(spec=ModelRouter)
    router.complete = AsyncMock()
    loop = LATSLoop(MagicMock(), router, evaluator="heuristic")

    complete = await loop._evaluate_state(
        AgentState(session_id="s", user_input="t", final_answer="done", status="complete")
    )
    failed = await loop._evaluate_state(
        AgentState(session_id="s", user_input="t", status="failed")
    )

    assert complete == 1.0
    assert failed == 0.0
    router.complete.assert_not_awaited()


@pytest.mark.asyncio
async def test_lats_run_terminates_when_budget_exhausted() -> None:
    """run() terminates when the simulation budget is exhausted."""
    router = MagicMock(spec=ModelRouter)
    router.complete = AsyncMock(
        side_effect=[
            _response('["branch one"]'),
            _response('{"goal_completion": 0.1, "factual_correctness": 0.1, "conciseness": 0.1}'),
        ]
    )
    kernel = MagicMock()
    kernel.run = AsyncMock(
        return_value=AgentState(
            session_id="child",
            user_input="task",
            final_answer="partial",
            status="failed",
        )
    )
    loop = LATSLoop(kernel, router, simulation_budget=1)

    state = await loop.run("task", "s")

    assert state.final_answer == "partial"
    assert kernel.run.await_count == 1
