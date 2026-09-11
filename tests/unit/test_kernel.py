"""Unit tests for AgentKernel — router, tool_registry, and memory are mocked."""

import asyncio
from collections.abc import AsyncIterator
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from cortex.agent.kernel import AgentKernel, AgentState
from cortex.agent.loop import LATSLoop
from cortex.config.settings import Settings
from cortex.memory.manager import MemoryManager
from cortex.memory.procedural import ToolPattern
from cortex.models.provider import ModelResponse, StreamChunk
from cortex.models.router import ModelRouter
from cortex.tools.base import ToolResult
from cortex.tools.registry import ToolRegistry


def _mock_model_response(content: str) -> ModelResponse:
    raw: dict[str, Any] = {
        "model": "llama3.1:8b",
        "message": {"role": "assistant", "content": content},
        "prompt_eval_count": 5,
        "eval_count": 10,
    }
    return ModelResponse(
        content=content,
        model="llama3.1:8b",
        input_tokens=5,
        output_tokens=10,
        latency_ms=50.0,
        raw=raw,
    )


def _mock_tool_call_response(tool_name: str = "dummy") -> ModelResponse:
    """Simulate an executor step that calls a tool (keeps loop alive)."""
    raw: dict[str, Any] = {
        "model": "llama3.1:8b",
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": tool_name, "arguments": {}}}],
        },
        "prompt_eval_count": 5,
        "eval_count": 10,
    }
    return ModelResponse(
        content="",
        model="llama3.1:8b",
        input_tokens=5,
        output_tokens=10,
        latency_ms=50.0,
        raw=raw,
    )


def _make_kernel(settings: Settings | None = None) -> tuple[AgentKernel, MagicMock, MagicMock]:
    router = MagicMock(spec=ModelRouter)
    router.route.return_value = "llama3.1:8b"
    router.complete = AsyncMock()

    tool_registry = MagicMock(spec=ToolRegistry)
    tool_registry.list_tools.return_value = []
    tool_registry.to_ollama_tools.return_value = []
    tool_registry.execute = AsyncMock(
        return_value=ToolResult(
            tool_name="dummy",
            success=True,
            output="tool ok",
            execution_time_ms=1.0,
        )
    )

    memory_manager = MagicMock(spec=MemoryManager)
    memory_manager.retrieve_context = AsyncMock(return_value="")
    memory_manager.store_turn = AsyncMock()
    memory_manager.end_session = AsyncMock()
    memory_manager.retrieve_tool_patterns = AsyncMock(return_value=[])
    memory_manager.store_tool_pattern = AsyncMock()

    cfg = settings or Settings(
        ollama_base_url="http://fake:11434",
        max_agent_steps=10,
    )

    kernel = AgentKernel(
        router=router,
        tool_registry=tool_registry,
        memory_manager=memory_manager,
        settings=cfg,
    )
    return kernel, router, tool_registry


@pytest.mark.asyncio
async def test_run_trivial_task_returns_complete() -> None:
    """A task where the model returns a direct answer completes with status=complete."""
    kernel, router, _ = _make_kernel()

    router.complete.side_effect = [
        _mock_model_response('["Answer the user directly"]'),
        _mock_model_response("The answer is 42."),
        _mock_model_response('{"progress": true}'),
    ]

    state = await kernel.run("What is 6 times 7?")

    assert state.status == "complete"
    assert state.final_answer is not None
    assert state.steps_taken >= 1


@pytest.mark.asyncio
async def test_run_exhausted_steps_returns_failed() -> None:
    """When max_steps is reached without completion, status is set to failed."""
    kernel, router, _ = _make_kernel(Settings(ollama_base_url="http://fake:11434", max_agent_steps=2))

    router.complete.side_effect = [
        _mock_model_response('["step one", "step two"]'),
        _mock_tool_call_response("dummy"),
        _mock_model_response('{"progress": true}'),
        _mock_tool_call_response("dummy"),
        _mock_model_response('{"progress": true}'),
    ]

    state = await kernel.run("An impossible task that takes many steps")

    assert state.status == "failed"


@pytest.mark.asyncio
async def test_run_creates_unique_session_id_when_none_given() -> None:
    """run() generates a session_id when none is provided."""
    kernel, router, _ = _make_kernel()
    router.complete.side_effect = [
        _mock_model_response('["single step"]'),
        _mock_model_response("Done."),
        _mock_model_response('{"progress": true}'),
    ]

    state = await kernel.run("test input")
    assert state.session_id
    assert len(state.session_id) == 32


@pytest.mark.asyncio
async def test_run_uses_provided_session_id() -> None:
    """run() uses the caller-supplied session_id."""
    kernel, router, _ = _make_kernel()
    router.complete.side_effect = [
        _mock_model_response('["step"]'),
        _mock_model_response("Result."),
        _mock_model_response('{"progress": true}'),
    ]

    state = await kernel.run("task", session_id="my-session-abc")
    assert state.session_id == "my-session-abc"


@pytest.mark.asyncio
async def test_run_reflector_halts_on_no_progress_when_lats_disabled() -> None:
    """Reflector stops the loop after two non-progress steps when LATS is disabled."""
    kernel, router, _ = _make_kernel(
        Settings(ollama_base_url="http://fake:11434", max_agent_steps=20)
    )

    router.complete.side_effect = [
        _mock_model_response('["step one", "step two"]'),
        _mock_tool_call_response("dummy"),
        _mock_model_response('{"progress": false}'),
        _mock_tool_call_response("dummy"),
        _mock_model_response('{"progress": false}'),
    ]

    state = await kernel.run("An unresolvable task", _allow_lats=False)
    assert state.status == "failed"


@pytest.mark.asyncio
async def test_run_reflector_falls_back_to_lats(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reflector failure falls back to LATS before returning failed."""
    kernel, router, _ = _make_kernel(
        Settings(ollama_base_url="http://fake:11434", max_agent_steps=20)
    )
    router.complete.side_effect = [
        _mock_model_response('["step one", "step two"]'),
        _mock_tool_call_response("dummy"),
        _mock_model_response('{"progress": false}'),
        _mock_tool_call_response("dummy"),
        _mock_model_response('{"progress": false}'),
    ]
    lats_state = AgentState(
        session_id="s1",
        user_input="An unresolvable task",
        final_answer="lats recovered",
        status="complete",
    )
    run_mock = AsyncMock(return_value=lats_state)
    monkeypatch.setattr(LATSLoop, "run", run_mock)

    state = await kernel.run("An unresolvable task")

    assert state is lats_state
    run_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_records_tool_pattern_on_success() -> None:
    """A successful run that used a tool stores a procedural pattern of that sequence."""
    kernel, router, _ = _make_kernel(
        Settings(ollama_base_url="http://fake:11434", max_agent_steps=10, stream_tokens=False)
    )
    router.complete.side_effect = [
        _mock_model_response('["use the tool", "answer"]'),
        _mock_tool_call_response("dummy"),
        _mock_model_response('{"progress": true}'),
        _mock_model_response("Final answer."),
    ]

    state = await kernel.run("a task that uses a tool")

    memory_manager = cast(MagicMock, kernel._memory_manager)
    assert state.status == "complete"
    memory_manager.store_tool_pattern.assert_awaited_once()
    kwargs = memory_manager.store_tool_pattern.await_args.kwargs
    assert kwargs["tool_sequence"] == ["dummy"]
    assert kwargs["success"] is True
    assert kwargs["task_description"] == "a task that uses a tool"


@pytest.mark.asyncio
async def test_run_threads_pattern_hint_into_planner() -> None:
    """A retrieved pattern is threaded into the planner prompt as a hint."""
    kernel, router, _ = _make_kernel(
        Settings(ollama_base_url="http://fake:11434", max_agent_steps=5, stream_tokens=False)
    )
    memory_manager = cast(MagicMock, kernel._memory_manager)
    memory_manager.retrieve_tool_patterns = AsyncMock(
        return_value=[
            ToolPattern(
                task_description="past task",
                tool_sequence=["calculator", "python_exec"],
                success=True,
                avg_steps=2,
            )
        ]
    )
    router.complete.side_effect = [
        _mock_model_response('["single step"]'),
        _mock_model_response("Done."),
    ]

    await kernel.run("a similar task")

    planner_messages = router.complete.await_args_list[0].args[1]
    planner_user_content = planner_messages[-1].content
    assert "calculator, python_exec" in planner_user_content
    assert "similar task used these tools" in planner_user_content


def _drain(queue: asyncio.Queue[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    while not queue.empty():
        events.append(queue.get_nowait())
    return events


@pytest.mark.asyncio
async def test_run_streams_real_tokens_before_done() -> None:
    """With streaming on and a queue, token events arrive as the model generates."""
    kernel, router, _ = _make_kernel(
        Settings(ollama_base_url="http://fake:11434", max_agent_steps=5, stream_tokens=True)
    )
    router.complete.side_effect = [_mock_model_response('["Answer the user directly"]')]

    async def fake_stream(*_args: Any, **_kwargs: Any) -> AsyncIterator[StreamChunk]:
        for piece in ["Hel", "lo, ", "world", "!"]:
            yield StreamChunk(content=piece)
        yield StreamChunk(content="", done=True, output_tokens=4)

    router.stream_complete = fake_stream

    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    state = await kernel.run("say hello", event_queue=queue)

    events = _drain(queue)
    token_events = [e for e in events if e["type"] == "token"]
    types = [e["type"] for e in events]

    assert state.final_answer == "Hello, world!"
    assert state.streamed_final is True
    assert len(token_events) > 1
    # tokens are emitted before the done event, not chunked after it
    assert types.index("token") < types.index("done")
    assert "".join(e["value"] for e in token_events) == "Hello, world!"


@pytest.mark.asyncio
async def test_run_with_use_lats_routes_to_lats(monkeypatch: pytest.MonkeyPatch) -> None:
    """use_lats=True routes execution through LATSLoop."""
    kernel, _, _ = _make_kernel()
    lats_state = AgentState(
        session_id="s1",
        user_input="hard task",
        final_answer="lats answer",
        status="complete",
    )
    run_mock = AsyncMock(return_value=lats_state)
    monkeypatch.setattr(LATSLoop, "run", run_mock)

    state = await kernel.run("hard task", session_id="s1", use_lats=True)

    assert state is lats_state
    run_mock.assert_awaited_once_with("hard task", "s1")
