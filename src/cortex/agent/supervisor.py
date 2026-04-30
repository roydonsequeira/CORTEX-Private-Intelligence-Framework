"""Supervisor-worker orchestration for parallel multi-agent tasks."""

import asyncio
import json
from collections.abc import Callable
from typing import Literal
from uuid import uuid4

import structlog
from pydantic import BaseModel, Field

from cortex.agent.kernel import AgentKernel, AgentState
from cortex.models.provider import GenerationConfig, Message
from cortex.models.router import ModelCapability, ModelRouter

logger = structlog.get_logger(__name__)


class SubTask(BaseModel):
    """A self-contained task assigned to a worker agent."""

    id: str
    description: str
    assigned_to: str
    result: AgentState | None = None
    status: Literal["pending", "running", "done", "failed"] = "pending"


class SupervisorResult(BaseModel):
    """Result from supervisor-worker orchestration."""

    session_id: str
    task: str
    subtasks: list[SubTask] = Field(default_factory=list)
    sub_states: list[AgentState] = Field(default_factory=list)
    final_answer: str
    status: Literal["complete", "failed"]


class SupervisorAgent:
    """Orchestrates multiple AgentKernel workers in parallel."""

    def __init__(
        self,
        kernel_factory: Callable[[], AgentKernel],
        router: ModelRouter,
        max_workers: int = 3,
    ) -> None:
        self._kernel_factory = kernel_factory
        self._router = router
        self._max_workers = max_workers

    async def run(self, task: str, session_id: str) -> SupervisorResult:
        """Decompose a task, run workers in parallel, and aggregate outputs."""
        subtasks = await self._decompose(task)
        if not subtasks:
            return SupervisorResult(
                session_id=session_id,
                task=task,
                subtasks=[],
                sub_states=[],
                final_answer="Supervisor could not decompose the task.",
                status="failed",
            )

        runnable = subtasks[: self._max_workers]
        sub_states = await asyncio.gather(
            *(self._run_subtask(subtask, session_id) for subtask in runnable)
        )
        final_answer = await self._aggregate(sub_states, task)
        return SupervisorResult(
            session_id=session_id,
            task=task,
            subtasks=runnable,
            sub_states=sub_states,
            final_answer=final_answer,
            status="complete" if final_answer else "failed",
        )

    async def _decompose(self, task: str) -> list[SubTask]:
        """Return independent sub-tasks that can run in parallel."""
        response = await self._router.complete(
            ModelCapability.REASONING,
            [
                Message(
                    role="system",
                    content=(
                        "Decompose this task into independent sub-tasks that can be "
                        "worked on in parallel. Each sub-task must be self-contained. "
                        "Output JSON array of sub-task descriptions. Maximum 4 sub-tasks."
                    ),
                ),
                Message(role="user", content=task),
            ],
            GenerationConfig(temperature=0.2, max_tokens=1024),
        )
        descriptions = _parse_json_list(response.content)
        return [
            SubTask(
                id=uuid4().hex,
                description=description,
                assigned_to=f"worker-{idx + 1}",
            )
            for idx, description in enumerate(descriptions[:4])
        ]

    async def _aggregate(self, sub_results: list[AgentState], original_task: str) -> str:
        """Synthesize worker outputs into a coherent final answer."""
        outputs = "\n\n".join(
            f"Worker {idx + 1} ({state.status}): {state.final_answer or ''}"
            for idx, state in enumerate(sub_results)
        )
        response = await self._router.complete(
            ModelCapability.REASONING,
            [
                Message(
                    role="system",
                    content=(
                        "You are synthesizing the results of parallel research tasks. "
                        "Combine the following results into a coherent, well-structured "
                        "answer to the original task."
                    ),
                ),
                Message(
                    role="user",
                    content=f"Original task: {original_task}\n\nSub-results:\n{outputs}",
                ),
            ],
            GenerationConfig(temperature=0.2, max_tokens=2048),
        )
        return response.content.strip()

    async def _run_subtask(self, subtask: SubTask, session_id: str) -> AgentState:
        """Run one worker kernel and attach the result to the subtask."""
        subtask.status = "running"
        kernel = self._kernel_factory()
        try:
            state = await kernel.run(
                subtask.description,
                session_id=f"{session_id}-{subtask.assigned_to}",
                _allow_lats=False,
            )
            subtask.result = state
            subtask.status = "done" if state.status == "complete" else "failed"
            return state
        except Exception as exc:
            logger.warning("supervisor_worker_failed", subtask=subtask.id, error=str(exc))
            state = AgentState(
                session_id=f"{session_id}-{subtask.assigned_to}",
                user_input=subtask.description,
                final_answer=str(exc),
                status="failed",
            )
            subtask.result = state
            subtask.status = "failed"
            return state


def _parse_json_list(raw: str) -> list[str]:
    """Parse an LLM JSON array response, falling back to numbered lines."""
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except json.JSONDecodeError:
        pass
    return [
        line.lstrip("0123456789. -").strip()
        for line in raw.splitlines()
        if line.strip()
    ]
