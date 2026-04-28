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

    async def step(
        self,
        state: "AgentState",
        tool_registry: ToolRegistry,
        router: ModelRouter,
    ) -> "AgentState":
        """Advance the agent state by one ReAct step.

        Returns the mutated state with updated messages, tool_results, and status.
        """

        with _tracer.start_as_current_span("executor.step") as span:
            span.set_attribute("session_id", state.session_id)
            span.set_attribute("step", state.steps_taken)

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

            response = await router.complete(
                ModelCapability.FAST,
                context_messages,
                GenerationConfig(temperature=0.3),
                tools=ollama_tools if ollama_tools else None,
            )

            state.steps_taken += 1
            increment_agent_steps(model=router.route(ModelCapability.FAST))

            raw_tool_calls = response.raw.get("message", {}).get("tool_calls")
            if raw_tool_calls:
                state = await self._handle_tool_calls(
                    raw_tool_calls, state, tool_registry, router
                )
            else:
                state.messages.append(
                    Message(role="assistant", content=response.content)
                )
                state.final_answer = response.content
                state.status = "complete"

        return state

    async def _handle_tool_calls(
        self,
        raw_tool_calls: list[dict[str, Any]],
        state: "AgentState",
        tool_registry: ToolRegistry,
        router: ModelRouter,
    ) -> "AgentState":
        """Execute each tool call with retry, appending results to state."""

        for tc in raw_tool_calls:
            fn = tc.get("function", tc)
            tool_name: str = fn.get("name", "")
            raw_args = fn.get("arguments", {})
            kwargs: dict[str, object] = (
                raw_args if isinstance(raw_args, dict) else json.loads(raw_args)
            )

            result = await self._execute_with_retry(tool_name, tool_registry, **kwargs)
            state.tool_results.append(result)
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
