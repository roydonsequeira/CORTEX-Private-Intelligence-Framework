"""AgentKernel — main orchestrator. Stateless per-call; no shared mutable instance state."""

import asyncio
import uuid
from typing import Literal

import structlog
from pydantic import BaseModel, Field

from cortex.agent.executor import Executor
from cortex.agent.planner import Planner
from cortex.agent.reflector import Reflector
from cortex.config.settings import Settings
from cortex.memory.manager import MemoryManager
from cortex.models.provider import Message, ToolResult
from cortex.models.router import ModelRouter
from cortex.observability.tracing import get_tracer
from cortex.tools.registry import ToolRegistry

logger = structlog.get_logger(__name__)
_tracer = get_tracer(__name__)


class AgentState(BaseModel):
    """Immutable-by-convention snapshot of the agent's progress on a single task."""

    session_id: str
    user_input: str
    plan: list[str] = Field(default_factory=list)
    steps_taken: int = 0
    messages: list[Message] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    final_answer: str | None = None
    status: Literal["planning", "executing", "reflecting", "complete", "failed"] = "planning"


class AgentKernel:
    """Orchestrates planner → executor loop → reflector for a single agent run.

    Each call to run() is fully independent. No shared mutable state on the instance.
    """

    def __init__(
        self,
        router: ModelRouter,
        tool_registry: ToolRegistry,
        memory_manager: MemoryManager,
        settings: Settings,
    ) -> None:
        self._router = router
        self._tool_registry = tool_registry
        self._memory_manager = memory_manager
        self._settings = settings
        self._planner = Planner(router)
        self._executor = Executor()

    async def run(
        self,
        user_input: str,
        session_id: str | None = None,
        event_queue: asyncio.Queue | None = None,
    ) -> AgentState:
        """Execute the agent loop for user_input and return the final AgentState."""
        sid = session_id or uuid.uuid4().hex
        state = AgentState(session_id=sid, user_input=user_input)

        with _tracer.start_as_current_span("kernel.run") as span:
            span.set_attribute("session_id", sid)

            context = await self._memory_manager.retrieve_context(user_input, sid)
            if context:
                state.messages.append(Message(role="system", content=context))

            state.messages.append(Message(role="user", content=user_input))

            await self._emit(event_queue, {"type": "session_id", "value": sid})

            state.status = "planning"
            try:
                tool_names = [s.name for s in self._tool_registry.list_tools()]
                state.plan = await self._planner.decompose(user_input, tool_names)
            except Exception as exc:
                logger.warning("planner_failed", error=str(exc), session_id=sid)
                state.plan = [user_input]

            await self._emit(event_queue, {"type": "plan", "steps": state.plan})

            reflector = Reflector(self._router)
            state.status = "executing"
            max_steps = self._settings.max_agent_steps

            while state.status not in ("complete", "failed") and state.steps_taken < max_steps:
                await self._emit(
                    event_queue,
                    {
                        "type": "step_start",
                        "step": state.steps_taken + 1,
                        "description": (
                            state.plan[state.steps_taken]
                            if state.steps_taken < len(state.plan)
                            else "continue"
                        ),
                    },
                )

                state = await self._executor.step(state, self._tool_registry, self._router)

                if state.tool_results:
                    last = state.tool_results[-1]
                    await self._emit(
                        event_queue,
                        {
                            "type": "tool_result",
                            "success": last.success,
                            "output": last.output[:512],
                        },
                    )

                await self._memory_manager.store_turn(sid, "assistant", state.messages[-1].content)

                if state.status not in ("complete", "failed"):
                    state.status = "reflecting"
                    state = await reflector.evaluate(state)
                    if state.status == "reflecting":
                        state.status = "executing"

            if state.status not in ("complete", "failed"):
                state.status = "failed"
                state.final_answer = state.final_answer or "Maximum steps reached without answer."

            if state.final_answer:
                await self._emit(
                    event_queue,
                    {"type": "token", "value": state.final_answer},
                )

            await self._emit(
                event_queue, {"type": "done", "steps_taken": state.steps_taken}
            )
            await self._memory_manager.end_session(sid)

        logger.info(
            "kernel_run_complete",
            session_id=sid,
            steps=state.steps_taken,
            status=state.status,
        )
        return state

    @staticmethod
    async def _emit(queue: asyncio.Queue | None, event: dict) -> None:
        """Push an event onto the queue if one is provided."""
        if queue is not None:
            await queue.put(event)
