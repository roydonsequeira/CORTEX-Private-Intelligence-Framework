"""Task planner — decomposes a user request into concrete executable steps."""

import json

import structlog

from cortex.exceptions import CortexPlannerError
from cortex.models.provider import GenerationConfig, Message
from cortex.models.router import ModelCapability, ModelRouter
from cortex.observability.tracing import get_tracer

logger = structlog.get_logger(__name__)
_tracer = get_tracer(__name__)

_SYSTEM_PROMPT = """\
You are a precise task planner. Given a user request and available tools, \
produce a numbered list of concrete steps. Each step must be independently \
executable. Do not include steps that require human input. Maximum 8 steps.

Respond with ONLY a JSON array of strings, e.g.:
["Step one description", "Step two description"]
"""


class Planner:
    """Breaks a user task into an ordered list of executable steps via the LLM."""

    def __init__(self, router: ModelRouter) -> None:
        self._router = router

    async def decompose(self, user_input: str, available_tools: list[str]) -> list[str]:
        """Return a list of step descriptions for the given task.

        Raises CortexPlannerError if the model returns unparseable output.
        """
        tools_summary = ", ".join(available_tools) if available_tools else "none"
        messages = [
            Message(role="system", content=_SYSTEM_PROMPT),
            Message(
                role="user",
                content=f"Task: {user_input}\nAvailable tools: {tools_summary}",
            ),
        ]
        with _tracer.start_as_current_span("planner.decompose"):
            response = await self._router.complete(
                ModelCapability.REASONING,
                messages,
                GenerationConfig(temperature=0.2, max_tokens=1024),
            )

        raw = response.content.strip()
        try:
            steps: list[str] = json.loads(raw)
            if not isinstance(steps, list):
                raise ValueError("Expected a JSON array.")
            return [str(s) for s in steps if s]
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("planner_parse_failed", raw=raw[:200], error=str(exc))
            lines = [
                line.lstrip("0123456789. -").strip()
                for line in raw.splitlines()
                if line.strip()
            ]
            if lines:
                return lines
            raise CortexPlannerError(f"Planner returned unparseable output: {raw[:200]}") from exc
