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
You are a precise task planner. Given a user request and the available tools, \
produce the SHORTEST plan that fully solves it.

Rules:
- Use the FEWEST steps possible. Most tasks need only ONE step; simple ones \
(a calculation, a single tool call, a direct answer) should be a one-step plan.
- Never repeat a step or re-derive a result you already have. Each step must do \
something genuinely new.
- Prefer a single tool that solves the whole task over chaining several tools.
- Do not invent steps that reference tools not listed, and do not include steps \
that require human input.
- Maximum 5 steps.

Respond with ONLY a JSON array of concise step strings, e.g.:
["Compute the answer with the python_exec tool and report it"]
"""


class Planner:
    """Breaks a user task into an ordered list of executable steps via the LLM."""

    def __init__(self, router: ModelRouter) -> None:
        self._router = router

    async def decompose(
        self,
        user_input: str,
        available_tools: list[str],
        hints: list[str] | None = None,
    ) -> list[str]:
        """Return a list of step descriptions for the given task.

        ``hints`` are optional lines derived from procedural memory (tool-use
        patterns from similar past tasks) that bias the planner toward proven
        approaches. Raises CortexPlannerError if the model returns unparseable output.
        """
        tools_summary = ", ".join(available_tools) if available_tools else "none"
        user_content = f"Task: {user_input}\nAvailable tools: {tools_summary}"
        if hints:
            hint_block = "\n".join(f"- {hint}" for hint in hints)
            user_content += (
                f"\n\nHints from similar past tasks (reuse what fits):\n{hint_block}"
            )
        messages = [
            Message(role="system", content=_SYSTEM_PROMPT),
            Message(role="user", content=user_content),
        ]
        with _tracer.start_as_current_span("planner.decompose") as span:
            span.set_attribute("model_name", self._router.route(ModelCapability.REASONING))
            span.set_attribute("tool_count", len(available_tools))
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
