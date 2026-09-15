"""Executor — a single step of the ReAct loop with exponential backoff retry."""

import asyncio
import json
from typing import TYPE_CHECKING, Any

import structlog

from cortex.exceptions import CortexToolError
from cortex.models.provider import GenerationConfig, Message
from cortex.models.router import ModelCapability, ModelRouter
from cortex.observability.metrics import increment_agent_steps
from cortex.observability.tracing import get_tracer
from cortex.tools.base import ToolResult
from cortex.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from cortex.agent.kernel import AgentState

logger = structlog.get_logger(__name__)
_tracer = get_tracer(__name__)

_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0

_SYSTEM_PROMPT = """\
You are CORTEX, a precise AI agent. Use the available tools to complete the \
current step. When you have a final answer, reply directly without calling any tool.
"""


class Executor:
    """Executes a single ReAct step: think → act → observe."""

    def __init__(self, stream: bool = False) -> None:
        """Create an executor. When ``stream`` is set, final-answer tokens are
        streamed to the event queue as the model generates them.
        """
        self._stream = stream

    async def step(
        self,
        state: "AgentState",
        tool_registry: ToolRegistry,
        router: ModelRouter,
        event_queue: asyncio.Queue[dict[str, Any]] | None = None,
    ) -> "AgentState":
        """Advance the agent state by one ReAct step.

        Returns the mutated state with updated messages, tool_results, and status.
        """

        with _tracer.start_as_current_span("executor.step") as span:
            span.set_attribute("session_id", state.session_id)
            span.set_attribute("step", state.steps_taken)
            span.set_attribute("step_count", state.steps_taken)
            span.set_attribute("model_name", router.route(ModelCapability.FAST))

            context_messages = [Message(role="system", content=_SYSTEM_PROMPT)]
            if state.plan:
                plan_text = "\n".join(
                    f"{i + 1}. {s}" for i, s in enumerate(state.plan)
                )
                context_messages.append(
                    Message(
                        role="user",
                        content=f"Plan:\n{plan_text}\n\nNow execute step {state.steps_taken + 1}.",
                    )
                )
            context_messages.extend(state.messages)

            ollama_tools = tool_registry.to_ollama_tools()
            tools_arg = ollama_tools if ollama_tools else None

            if self._stream and event_queue is not None:
                content, raw_tool_calls, streamed = await self._stream_step(
                    context_messages, tools_arg, router, event_queue
                )
            else:
                response = await router.complete(
                    ModelCapability.FAST,
                    context_messages,
                    GenerationConfig(temperature=0.3),
                    tools=tools_arg,
                )
                content = response.content
                raw_tool_calls = response.raw.get("message", {}).get("tool_calls")
                streamed = False

            state.steps_taken += 1
            increment_agent_steps(model=router.route(ModelCapability.FAST))

            if raw_tool_calls:
                state = await self._handle_tool_calls(
                    raw_tool_calls, state, tool_registry, router, event_queue
                )
            else:
                state.messages.append(Message(role="assistant", content=content))
                state.final_answer = content
                state.status = "complete"
                state.streamed_final = streamed

        return state

    async def _stream_step(
        self,
        context_messages: list[Message],
        tools_arg: list[dict[str, Any]] | None,
        router: ModelRouter,
        event_queue: asyncio.Queue[dict[str, Any]],
    ) -> tuple[str, list[dict[str, Any]] | None, bool]:
        """Stream a step, emitting token events, and return (content, tool_calls, streamed).

        ``streamed`` is True only when the step produced a final answer that was
        emitted token-by-token (i.e. no tool call), so the kernel can avoid
        re-emitting it.
        """
        parts: list[str] = []
        tool_calls: list[dict[str, Any]] | None = None
        emitted = False
        async for chunk in router.stream_complete(
            ModelCapability.FAST,
            context_messages,
            GenerationConfig(temperature=0.3),
            tools=tools_arg,
        ):
            if chunk.content:
                parts.append(chunk.content)
                await self._emit(event_queue, {"type": "token", "value": chunk.content})
                emitted = True
            if chunk.tool_calls:
                tool_calls = chunk.tool_calls
        return "".join(parts), tool_calls, emitted and tool_calls is None

    async def _handle_tool_calls(
        self,
        raw_tool_calls: list[dict[str, Any]],
        state: "AgentState",
        tool_registry: ToolRegistry,
        router: ModelRouter,
        event_queue: asyncio.Queue[dict[str, Any]] | None = None,
    ) -> "AgentState":
        """Execute each tool call with retry, appending results to state."""

        for tc in raw_tool_calls:
            fn = tc.get("function", tc)
            tool_name: str = fn.get("name", "")
            raw_args = fn.get("arguments", {})
            kwargs: dict[str, object] = (
                raw_args if isinstance(raw_args, dict) else json.loads(raw_args)
            )

            with _tracer.start_as_current_span("executor.tool_call") as span:
                span.set_attribute("session_id", state.session_id)
                span.set_attribute("tool_name", tool_name)
                span.set_attribute("step_count", state.steps_taken)
            await self._emit(
                event_queue,
                {"type": "tool_call", "tool": tool_name, "args": kwargs},
            )
            result = await self._execute_with_retry(tool_name, tool_registry, **kwargs)
            state.tool_results.append(result)
            await self._emit(
                event_queue,
                {
                    "type": "tool_result",
                    "tool": tool_name,
                    "success": result.success,
                    "output": result.output[:512],
                    "error": result.error,
                },
            )
            state.messages.append(
                Message(
                    role="tool",
                    content=result.output if result.success else (result.error or "Tool failed."),
                )
            )

        state.status = "executing"
        return state

    async def _execute_with_retry(
        self,
        tool_name: str,
        tool_registry: ToolRegistry,
        **kwargs: object,
    ) -> ToolResult:
        """Execute a tool with exponential backoff on failure (max 3 retries)."""
        last_result = ToolResult(
            tool_name=tool_name,
            success=False,
            output="",
            error="Tool execution did not run.",
            execution_time_ms=0.0,
        )
        for attempt in range(_MAX_RETRIES):
            try:
                result = await tool_registry.execute(tool_name, **kwargs)
                if result.success:
                    return result
                last_result = result
            except CortexToolError as exc:
                logger.warning(
                    "tool_retry",
                    tool=tool_name,
                    attempt=attempt + 1,
                    error=str(exc),
                )
                last_result = ToolResult(
                    tool_name=tool_name,
                    success=False,
                    output="",
                    error=str(exc),
                    execution_time_ms=0.0,
                )

            if attempt < _MAX_RETRIES - 1:
                await asyncio.sleep(_RETRY_BASE_DELAY * (2**attempt))

        return last_result

    @staticmethod
    async def _emit(
        queue: asyncio.Queue[dict[str, Any]] | None, event: dict[str, Any]
    ) -> None:
        """Push executor events onto the queue if one is provided."""
        if queue is not None:
            await queue.put(event)
