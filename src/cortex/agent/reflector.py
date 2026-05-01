"""Reflector — evaluates each step and halts the loop if no progress is made."""

import json
from typing import TYPE_CHECKING

import structlog

from cortex.models.provider import GenerationConfig, Message
from cortex.models.router import ModelCapability, ModelRouter
from cortex.observability.tracing import get_tracer

if TYPE_CHECKING:
    from cortex.agent.kernel import AgentState

logger = structlog.get_logger(__name__)
_tracer = get_tracer(__name__)

_SYSTEM_PROMPT = """\
You are a strict progress evaluator. Respond ONLY with valid JSON: \
{"progress": true} or {"progress": false}. \
"progress" is true if the last step meaningfully advanced the task goal.
"""


class Reflector:
    """Self-critique pass after each executor step."""

    def __init__(self, router: ModelRouter) -> None:
        self._router = router
        self._consecutive_no_progress: int = 0

    async def evaluate(self, state: "AgentState") -> "AgentState":
        """Ask the model whether the last step made progress; halt loop if not.

        Returns the (possibly mutated) state.
        """
        if state.status in ("complete", "failed"):
            return state

        last_message = state.messages[-1].content if state.messages else ""
        messages = [
            Message(role="system", content=_SYSTEM_PROMPT),
            Message(
                role="user",
                content=(
                    f"Original task: {state.user_input}\n"
                    f"Steps taken so far: {state.steps_taken}\n"
                    f"Last step output: {last_message[:500]}"
                ),
            ),
        ]

        with _tracer.start_as_current_span("reflector.evaluate") as span:
            span.set_attribute("session_id", state.session_id)
            span.set_attribute("step_count", state.steps_taken)
            span.set_attribute("model_name", self._router.route(ModelCapability.FAST))
            response = await self._router.complete(
                ModelCapability.FAST,
                messages,
                GenerationConfig(temperature=0.0, max_tokens=32),
            )

        made_progress = self._parse_progress(response.content)
        if made_progress:
            self._consecutive_no_progress = 0
        else:
            self._consecutive_no_progress += 1
            logger.info(
                "reflector_no_progress",
                consecutive=self._consecutive_no_progress,
                session_id=state.session_id,
            )
            if self._consecutive_no_progress >= 2:
                state.status = "failed"
                state.final_answer = "Agent failed to make progress. Stopping."

        return state

    def _parse_progress(self, raw: str) -> bool:
        """Parse the model's JSON progress signal, defaulting to True on parse error."""
        try:
            data = json.loads(raw.strip())
            return bool(data.get("progress", True))
        except (json.JSONDecodeError, AttributeError):
            return True
