"""Shared fakes for the security regression tests (no network, no real model)."""

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from cortex.agent.kernel import AgentKernel
from cortex.config.settings import Settings
from cortex.memory.manager import MemoryManager
from cortex.models.provider import ModelResponse
from cortex.models.router import ModelRouter
from cortex.tools.base import ToolResult
from cortex.tools.registry import ToolRegistry


def model_text(content: str) -> ModelResponse:
    """A model reply with plain text and no tool call."""
    return ModelResponse(
        content=content,
        model="fake",
        input_tokens=1,
        output_tokens=1,
        latency_ms=1.0,
        raw={"message": {"role": "assistant", "content": content}},
    )


def model_tool_call(name: str, arguments: dict[str, Any]) -> ModelResponse:
    """A model reply that calls one tool."""
    return ModelResponse(
        content="",
        model="fake",
        input_tokens=1,
        output_tokens=1,
        latency_ms=1.0,
        raw={
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": name, "arguments": arguments}}],
            }
        },
    )


def ok(tool_name: str, output: str = "ok") -> ToolResult:
    return ToolResult(tool_name=tool_name, success=True, output=output, execution_time_ms=1.0)


def failed(tool_name: str, error: str) -> ToolResult:
    return ToolResult(
        tool_name=tool_name, success=False, output="", error=error, execution_time_ms=1.0
    )


def scripted_kernel(
    plan: list[str],
    executor_replies: list[ModelResponse],
    tool_results: dict[str, ToolResult] | None = None,
) -> tuple[AgentKernel, MagicMock, MagicMock]:
    """A kernel whose planner, reflector and executor are scripted fakes.

    The planner always returns ``plan``, the reflector always reports progress,
    and executor steps consume ``executor_replies`` in order (then "Done.").
    ``tool_results`` maps a tool name to what the (mock) registry returns.
    """
    replies = list(executor_replies)
    results = tool_results or {}

    async def complete(_capability: Any, messages: Any, *_args: Any, **_kwargs: Any) -> Any:
        system = messages[0].content if messages else ""
        if system.startswith("You are CORTEX"):
            return replies.pop(0) if replies else model_text("Done.")
        if "progress evaluator" in system:
            return model_text('{"progress": true}')
        return model_text(json.dumps(plan))

    router = MagicMock(spec=ModelRouter)
    router.route.return_value = "fake"
    router.complete = AsyncMock(side_effect=complete)

    async def execute(name: str, /, **_kwargs: Any) -> ToolResult:
        return results.get(name, ok(name))

    registry = MagicMock(spec=ToolRegistry)
    registry.list_tools.return_value = []
    registry.to_ollama_tools.return_value = []
    registry.execute = AsyncMock(side_effect=execute)

    memory = MagicMock(spec=MemoryManager)
    memory.retrieve_context = AsyncMock(return_value="")
    memory.recent_history = AsyncMock(return_value=[])
    memory.store_turn = AsyncMock()
    memory.end_session = AsyncMock()
    memory.retrieve_tool_patterns = AsyncMock(return_value=[])
    memory.store_tool_pattern = AsyncMock()

    settings = Settings(
        ollama_base_url="http://fake:11434", max_agent_steps=8, stream_tokens=False
    )
    kernel = AgentKernel(
        router=router, tool_registry=registry, memory_manager=memory, settings=settings
    )
    return kernel, registry, memory


def executed_tools(registry: MagicMock) -> list[str]:
    """Names of the tools the kernel actually executed through the registry."""
    return [call.args[0] for call in registry.execute.await_args_list]
