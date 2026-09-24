"""AgentKernel — main orchestrator. Stateless per-call; no shared mutable instance state."""

import asyncio
import re
import time
import uuid
from typing import Any, Literal

import structlog
from opentelemetry.trace import Span
from pydantic import BaseModel, Field

from cortex.agent.executor import Executor, instruction_text
from cortex.agent.loop import LATSLoop
from cortex.agent.planner import Planner
from cortex.agent.reflector import Reflector
from cortex.agent.runnability import cannot_run_reply
from cortex.config.settings import Settings
from cortex.exceptions import CortexModelError
from cortex.memory.manager import MemoryManager
from cortex.models.provider import Message
from cortex.models.router import ModelCapability, ModelRouter
from cortex.observability.tracing import get_tracer
from cortex.tools.base import ToolResult
from cortex.tools.registry import ToolRegistry

logger = structlog.get_logger(__name__)
_tracer = get_tracer(__name__)

# Recent dialogue shown to the planner so follow-up questions plan in context.
_PLANNER_HISTORY_MESSAGES = 4
_PLANNER_HISTORY_CHARS = 300


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
    streamed_final: bool = False
    # Long-term memory block injected into the system prompt for this run.
    context: str = Field(default="", exclude=True)
    # tool-call signature -> rendered result, used to catch repeated identical calls.
    tool_history: dict[str, str] = Field(default_factory=dict, exclude=True)
    repeated_calls: int = Field(default=0, exclude=True)
    stalled: bool = Field(default=False, exclude=True)
    # The planner saw procedural-memory hints for this run.
    hinted: bool = Field(default=False, exclude=True)


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
        self._executor = Executor(stream=settings.stream_tokens)

    async def run(
        self,
        user_input: str,
        session_id: str | None = None,
        event_queue: asyncio.Queue[dict[str, Any]] | None = None,
        use_lats: bool | None = None,
        capability: ModelCapability = ModelCapability.FAST,
        _allow_lats: bool = True,
        _persist: bool = True,
    ) -> AgentState:
        """Execute the agent loop for user_input and return the final AgentState.

        Model/Ollama failures do not raise: the run ends with ``status="failed"``,
        an actionable message in ``final_answer``, and an ``error`` event, so API
        and UI callers always receive a well-formed result. ``_persist=False``
        (used by LATS simulations) keeps throwaway branches out of memory.
        """
        # str(uuid4()) (dashed) round-trips through the API's UUID-typed session_id
        # unchanged; a bare hex id would come back dashed and split the session.
        sid = session_id or str(uuid.uuid4())
        should_use_lats = _allow_lats and (use_lats is True or self._settings.use_lats)
        if should_use_lats:
            return await self._run_lats(user_input, sid, event_queue)

        state = AgentState(session_id=sid, user_input=user_input)

        with _tracer.start_as_current_span("kernel.run") as span:
            span.set_attribute("session_id", sid)
            span.set_attribute("orchestration", "react")
            await self._emit(event_queue, {"type": "session_id", "value": sid})

            model_failed = False
            try:
                escalated = await self._react(
                    state, event_queue, capability, _allow_lats, _persist, span
                )
            except CortexModelError as exc:
                logger.error("kernel_model_error", session_id=sid, error=str(exc))
                model_failed = True
                state.status = "failed"
                state.final_answer = str(exc)
                state.streamed_final = True  # reported via the error event, not tokens
                await self._emit(event_queue, {"type": "error", "message": str(exc)})
                escalated = None

            if escalated is not None:
                state = escalated
            elif state.final_answer and not state.streamed_final:
                for chunk in _chunk_text(state.final_answer):
                    await self._emit(event_queue, {"type": "token", "value": chunk})

            await self._emit(event_queue, {"type": "done", "steps_taken": state.steps_taken})
            if _persist:
                await self._finish_session(sid, state, model_failed)

        logger.info(
            "kernel_run_complete",
            session_id=sid,
            steps=state.steps_taken,
            status=state.status,
        )
        return state

    async def _react(
        self,
        state: AgentState,
        event_queue: asyncio.Queue[dict[str, Any]] | None,
        capability: ModelCapability,
        allow_lats: bool,
        persist: bool,
        span: Span,
    ) -> AgentState | None:
        """Plan and run the ReAct loop. Returns a LATS state if the run escalated."""
        sid = state.session_id
        history = await self._load_history(sid) if persist else []
        state.context = await self._load_context(state.user_input, sid)
        state.messages.extend(history)
        state.messages.append(Message(role="user", content=state.user_input))
        if persist:
            await self._store(sid, "user", state.user_input)

        canned = cannot_run_reply(state.user_input, history)
        if canned is not None:
            # "Run it" for a game, GUI or input() program: explain directly
            # instead of letting the model paste the whole program again.
            state.plan = ["Explain why this code can't run in the sandbox"]
            await self._emit(event_queue, {"type": "plan", "steps": state.plan})
            state.messages.append(Message(role="assistant", content=canned))
            state.final_answer = canned
            state.status = "complete"
            return None

        started = time.monotonic()
        deadline = started + self._settings.max_run_seconds
        state.status = "planning"
        hints = await self._retrieve_pattern_hints(state.user_input, sid)
        state.hinted = bool(hints)
        try:
            # Name plus a short description, so the planner knows what each tool
            # can and cannot do (e.g. that filesystem has no delete action).
            tools = [
                f"{schema.name} ({schema.description.split('.')[0].strip()})"
                for schema in self._tool_registry.list_tools()
            ]
            plan = _refuse_destructive_plan(
                await self._planner.decompose(
                    state.user_input,
                    tools,
                    hints=hints,
                    conversation=_format_history(history),
                )
            )
            # Guards read only the user's own words, never quoted/pasted text.
            instruction = instruction_text(state.user_input)
            plan = _refuse_unsupported_file_ops(plan)
            plan = _drop_unrequested_web_steps(plan, instruction)
            plan = _keep_code_requests_unrun(plan, instruction)
            plan = _ensure_file_save_step(plan, instruction)
            state.plan = _honour_explicit_python(plan, instruction)
        except CortexModelError:
            raise  # Ollama itself is unavailable; the executor would fail the same way
        except Exception as exc:
            logger.warning("planner_failed", error=str(exc), session_id=sid)
            state.plan = [state.user_input]

        await self._emit(event_queue, {"type": "plan", "steps": state.plan})

        reflector = Reflector(self._router)
        state.status = "executing"
        max_steps = self._settings.max_agent_steps
        allow_tools = not _is_tool_free_plan(state.plan)
        planned_tools = (
            _planned_tools(
                state.plan,
                [t.name for t in self._tool_registry.list_tools()],
                instruction_text(state.user_input),
            )
            if allow_tools
            else []
        )
        # Each recovery (skipped planned tool, promised-but-missing retry) runs once.
        recoveries_left = {"run_code", "nudge", "retry"}

        while (
            state.status not in ("complete", "failed")
            and state.steps_taken < max_steps
            and time.monotonic() < deadline
        ):
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
            span.set_attribute("step_count", state.steps_taken)

            results_before = len(state.tool_results)
            repeats_before = state.repeated_calls
            state = await self._executor.step(
                state,
                self._tool_registry,
                self._router,
                event_queue,
                capability,
                allow_tools=allow_tools,
            )
            if persist:
                await self._store_step_tools(sid, state, results_before)

            # Tools the model attempted this run (a failed attempt still counts).
            used = {call.name for m in state.messages for call in (m.tool_calls or [])}
            skipped = [tool for tool in planned_tools if tool not in used]
            recovery = (
                _next_recovery(skipped[0], state.final_answer or "", recoveries_left)
                if state.status == "complete" and skipped
                else None
            )
            if (
                state.status == "complete"
                and state.tool_results
                and not state.tool_results[-1].success
                and "retry" in recoveries_left
                and _PROMISED_RETRY.search(state.final_answer or "")
            ):
                # "Let's try running the code again" — and the turn ended there.
                recoveries_left.discard("retry")
                await self._retry_failed_tool(state, state.tool_results[-1], event_queue)
                continue
            if skipped and recovery:
                # The plan needs a tool the user asked for, but the model answered
                # without it (seen: "I have saved x to notes.txt" with no write; a
                # fetched page "counted" with no Python). Discard that answer and
                # recover.
                recoveries_left.discard(recovery)
                results_before = len(state.tool_results)
                await self._recover_planned_tool(state, skipped[0], recovery, event_queue)
                if persist:
                    await self._store_step_tools(sid, state, results_before)
                continue
            if state.status in ("complete", "failed"):
                break
            new_results = state.tool_results[results_before:]
            clean_step = (
                bool(new_results)
                and all(result.success for result in new_results)
                and state.repeated_calls == repeats_before
            )
            if clean_step:
                reflector.record_progress()
                continue
            state.status = "reflecting"
            state = await reflector.evaluate(state)
            if state.status == "reflecting":
                state.status = "executing"

        remaining = deadline - time.monotonic()
        if state.status == "failed" and state.stalled and allow_lats and remaining > 15:
            lats_state = await self._escalate_to_lats(state, event_queue, remaining)
            if lats_state is not None:
                return lats_state

        if state.status != "complete":
            await self._finalize_incomplete(state, event_queue, capability)

        if state.status == "complete":
            await self._record_pattern(state.user_input, state, sid)
        return None

    async def _recover_planned_tool(
        self,
        state: AgentState,
        tool: str,
        recovery: str,
        event_queue: asyncio.Queue[dict[str, Any]] | None,
    ) -> None:
        """Replace a tool-less answer with a real use of ``tool``.

        ``run_code``: the model wrote the Python instead of running it ("use
        Python to simulate..." answered with a code block), so that code is run in
        the sandbox on its behalf. ``nudge``: it is told, as the latest message, to
        call the tool.
        """
        answer = state.final_answer or ""
        if state.messages and state.messages[-1].role == "assistant":
            state.messages.pop()
        if state.streamed_final:
            await self._emit(event_queue, {"type": "token_reset"})
        state.final_answer = None
        state.streamed_final = False
        state.status = "executing"

        if recovery == "run_code":
            logger.info("planned_tool_ran_model_code", session_id=state.session_id)
            await self._executor.run_tool_for_model(
                state, self._tool_registry, "python_exec", {"code": _python_code(answer)},
                event_queue,
            )
            return
        logger.info("planned_tool_nudge", session_id=state.session_id, tool=tool)
        state.messages.append(
            Message(
                role="user",
                content=(
                    f"Use the {tool} tool now to do what I asked. Do not answer until "
                    "you have its result."
                ),
            )
        )

    async def _retry_failed_tool(
        self,
        state: AgentState,
        failed: ToolResult,
        event_queue: asyncio.Queue[dict[str, Any]] | None,
    ) -> None:
        """Replace "let's try again" (with no retry) by an instruction to retry now."""
        logger.info("promised_retry_nudge", session_id=state.session_id, tool=failed.tool_name)
        if state.messages and state.messages[-1].role == "assistant":
            state.messages.pop()
        if state.streamed_final:
            await self._emit(event_queue, {"type": "token_reset"})
        state.final_answer = None
        state.streamed_final = False
        state.status = "executing"
        state.messages.append(
            Message(
                role="user",
                content=(
                    f"The {failed.tool_name} call failed: {(failed.error or '')[:300]} "
                    f"Fix the problem and call {failed.tool_name} again now, instead of "
                    "describing what you will do."
                ),
            )
        )

    async def _finalize_incomplete(
        self,
        state: AgentState,
        event_queue: asyncio.Queue[dict[str, Any]] | None,
        capability: ModelCapability,
    ) -> None:
        """Produce a best-effort answer when the loop stalled or hit its step limit.

        The run stays ``failed`` (the task did not complete within budget), but the
        user still gets an answer built from everything gathered so far rather
        than a bare "maximum steps reached".
        """
        reason = "stalled" if state.stalled else "step_or_time_limit"
        logger.info("kernel_finalizing", session_id=state.session_id, reason=reason)
        state.status = "failed"
        try:
            answer, streamed = await self._executor.synthesize(
                state, self._router, event_queue, capability
            )
        except CortexModelError:
            raise
        except Exception as exc:
            logger.warning("synthesis_failed", error=str(exc), session_id=state.session_id)
            answer, streamed = "", False
        state.final_answer = answer or (
            "I could not complete this request within my step budget. "
            "Try rephrasing it or breaking it into smaller questions."
        )
        state.streamed_final = streamed and bool(answer)

    async def _escalate_to_lats(
        self,
        state: AgentState,
        event_queue: asyncio.Queue[dict[str, Any]] | None,
        remaining_seconds: float | None = None,
    ) -> AgentState | None:
        """Hand a stalled task to LATS, bounded by a wall-clock budget.

        Returns the LATS state when it completes the task; otherwise None, and
        the caller synthesises a best-effort answer from the ReAct transcript.
        """
        lats = self._settings.lats
        if not lats.escalate_on_stall:
            return None
        await self._emit(
            event_queue,
            {"type": "plan", "steps": ["Stalled — escalating to Language Agent Tree Search"]},
        )
        try:
            lats_state = await asyncio.wait_for(
                self._lats_search(state.user_input, state.session_id),
                timeout=min(
                    lats.escalation_timeout_seconds,
                    remaining_seconds or lats.escalation_timeout_seconds,
                ),
            )
        except TimeoutError:
            logger.warning("lats_escalation_timeout", session_id=state.session_id)
            return None
        except CortexModelError:
            raise
        except Exception as exc:
            logger.warning("lats_escalation_failed", error=str(exc), session_id=state.session_id)
            return None
        if lats_state.status != "complete" or not lats_state.final_answer:
            return None
        lats_state.steps_taken += state.steps_taken
        for chunk in _chunk_text(lats_state.final_answer):
            await self._emit(event_queue, {"type": "token", "value": chunk})
        lats_state.streamed_final = True
        return lats_state

    async def _load_history(self, sid: str) -> list[Message]:
        """Replay this session's recent dialogue; memory problems never fail a run."""
        try:
            return await self._memory_manager.recent_history(
                sid, max_turns=self._settings.history_turns
            )
        except Exception as exc:
            logger.warning("history_load_failed", error=str(exc), session_id=sid)
            return []

    async def _load_context(self, user_input: str, sid: str) -> str:
        """Fetch long-term memory for the request; failures degrade to no context."""
        try:
            return await self._memory_manager.retrieve_context(user_input, sid)
        except Exception as exc:
            logger.warning("memory_context_failed", error=str(exc), session_id=sid)
            return ""

    async def _store(self, sid: str, role: str, content: str) -> None:
        """Persist one turn; storage problems are logged, never raised."""
        if not content.strip():
            return
        try:
            await self._memory_manager.store_turn(sid, role, content)
        except Exception as exc:
            logger.warning("memory_store_failed", error=str(exc), session_id=sid)

    async def _store_step_tools(self, sid: str, state: AgentState, results_before: int) -> None:
        """Persist the tool results produced by the step just taken."""
        for result in state.tool_results[results_before:]:
            content = result.output if result.success else f"ERROR: {result.error or 'failed'}"
            await self._store(sid, "tool", f"[{result.tool_name}] {content}"[:4000])

    async def _finish_session(self, sid: str, state: AgentState, model_failed: bool) -> None:
        """Persist the answer and schedule background memory consolidation.

        An infrastructure error message ("cannot reach Ollama") is not stored as
        the assistant's reply, so it is never replayed as conversation history.
        """
        if state.final_answer and not model_failed:
            await self._store(sid, "assistant", state.final_answer)
        try:
            await self._memory_manager.end_session(sid)
        except Exception as exc:
            logger.warning("end_session_failed", error=str(exc), session_id=sid)

    async def _retrieve_pattern_hints(self, user_input: str, sid: str) -> list[str]:
        """Return planner hints from procedural memory for similar past tasks."""
        if not self._settings.procedural_memory_enabled:
            return []
        try:
            patterns = await self._memory_manager.retrieve_tool_patterns(
                user_input, min_relevance=self._settings.procedural_min_relevance
            )
        except Exception as exc:  # procedural memory is advisory, never fatal
            logger.warning("pattern_retrieval_failed", error=str(exc), session_id=sid)
            return []
        return [
            f"A similar task used these tools in order: "
            f"{', '.join(p.tool_sequence)} (took {p.avg_steps} step(s))."
            for p in patterns
            if p.success and p.tool_sequence
        ]

    async def _record_pattern(self, user_input: str, state: AgentState, sid: str) -> None:
        """Persist the tool sequence of a successful run for future planner hints.

        Runs whose plan was shaped by hints are not recorded: otherwise one
        unnecessary tool choice is hinted, repeated, and re-recorded forever.
        """
        if not self._settings.procedural_memory_enabled or state.hinted:
            return
        tool_sequence = [result.tool_name for result in state.tool_results if result.success]
        if not tool_sequence:
            return  # nothing procedural to learn from a pure-reasoning answer
        try:
            await self._memory_manager.store_tool_pattern(
                task_description=user_input,
                tool_sequence=tool_sequence,
                success=True,
                avg_steps=state.steps_taken,
            )
        except Exception as exc:  # learning is best-effort, never fatal to the run
            logger.warning("pattern_store_failed", error=str(exc), session_id=sid)

    async def _run_lats(
        self,
        user_input: str,
        session_id: str,
        event_queue: asyncio.Queue[dict[str, Any]] | None = None,
    ) -> AgentState:
        """Run the LATS loop and emit compatible SSE events."""
        await self._emit(event_queue, {"type": "session_id", "value": session_id})
        await self._emit(
            event_queue,
            {"type": "plan", "steps": ["Run Language Agent Tree Search"]},
        )
        try:
            state = await self._lats_search(user_input, session_id)
        except CortexModelError as exc:
            state = AgentState(
                session_id=session_id,
                user_input=user_input,
                status="failed",
                final_answer=str(exc),
            )
            await self._emit(event_queue, {"type": "error", "message": str(exc)})
        else:
            if state.final_answer:
                for chunk in _chunk_text(state.final_answer):
                    await self._emit(event_queue, {"type": "token", "value": chunk})
        await self._emit(event_queue, {"type": "done", "steps_taken": state.steps_taken})
        return state

    async def _lats_search(self, user_input: str, session_id: str) -> AgentState:
        """Run LATS inside its tracing span and return the best state found."""
        with _tracer.start_as_current_span("kernel.lats") as span:
            span.set_attribute("session_id", session_id)
            span.set_attribute("orchestration", "lats")
            span.set_attribute("lats.max_depth", self._settings.lats.max_depth)
            span.set_attribute("lats.n_branches", self._settings.lats.n_branches)
            span.set_attribute("lats.budget", self._settings.lats.budget)
            loop = LATSLoop(
                self,
                self._router,
                max_depth=self._settings.lats.max_depth,
                n_branches=self._settings.lats.n_branches,
                simulation_budget=self._settings.lats.budget,
                evaluator=self._settings.lats.evaluator,
            )
            state = await loop.run(user_input, session_id)
            span.set_attribute("step_count", state.steps_taken)
            return state

    @staticmethod
    async def _emit(
        queue: asyncio.Queue[dict[str, Any]] | None, event: dict[str, Any]
    ) -> None:
        """Push an event onto the queue if one is provided."""
        if queue is not None:
            await queue.put(event)


# A plan step that would destroy files or folders. No tool can do this, so such a
# plan is replaced with a refusal rather than letting the model improvise.
_DESTRUCTIVE_STEP = re.compile(
    r"\b(delete|wipe|erase|destroy|rm\s+-rf)\b.*\b(all|every|everything|workspace|"
    r"folder|folders|directory|directories|files|disk|drive)\b",
    re.IGNORECASE,
)
_REFUSAL_STEP = "Politely refuse: deleting files is not permitted"


# Steps that talk *about* deletion ("Explain how to delete files in Python") are
# knowledge answers, not actions.
_KNOWLEDGE_STEP = re.compile(
    r"^\s*(answer|explain|describe|tell|summari[sz]e|compare|list the ways|how to)\b",
    re.IGNORECASE,
)


def _refuse_destructive_plan(plan: list[str]) -> list[str]:
    """Replace a plan that tries to destroy data with an explicit refusal step."""
    if any(
        _DESTRUCTIVE_STEP.search(step) and not _KNOWLEDGE_STEP.match(step) for step in plan
    ):
        logger.warning("destructive_plan_refused", plan=plan)
        return [_REFUSAL_STEP]
    return plan


def _is_tool_free_plan(plan: list[str]) -> bool:
    """True when the planner decided no tool is needed: a direct answer or a refusal.

    The executor then runs without tool schemas, so a model that ignores the
    plan cannot start calling tools anyway (seen: a refusal plan that still
    produced 165 filesystem calls when tools were offered).
    """
    if len(plan) != 1:
        return False
    step = plan[0].strip().lower()
    return step.startswith("answer directly") or "refuse" in step


# Words in the user's own request that show they want a particular tool used.
_TOOL_INTENT = {
    "filesystem": re.compile(
        r"\b(file|files|folder|directory|save|write|read|create|readme|according to)\b"
        r"|\.(txt|md|json|csv|py|yaml|yml|log)\b",
        re.IGNORECASE,
    ),
    "python_exec": re.compile(r"\b(python|run|execute|code|script)\b", re.IGNORECASE),
    "calculator": re.compile(r"\b(calculate|calculator|compute)\b", re.IGNORECASE),
    # "According to the README, ..." answered "based on the README" without reading it.
    "doc_search": re.compile(
        r"\b(index|indexed|documents?|search|readme|docs|according to)\b", re.IGNORECASE
    ),
    "web_fetch": re.compile(
        r"https?://|www\.|\b(fetch|url|website|web ?page|online|internet|browse)\b",
        re.IGNORECASE,
    ),
}


def _planned_tools(plan: list[str], tool_names: list[str], user_input: str) -> list[str]:
    """Tools the plan names that the user's request also calls for, in plan order.

    Requiring both keeps the recovery for real misses ("save it to notes.txt"
    answered without saving) and away from planner over-reach (a calculator step
    for "how tall is it in feet?").
    """
    found: list[str] = []
    for step in plan:
        lowered = step.lower()
        for name in tool_names:
            intent = _TOOL_INTENT.get(name)
            if (
                name not in found
                and intent is not None
                and intent.search(user_input)
                and re.search(rf"\b{re.escape(name.lower())}\b", lowered)
            ):
                found.append(name)
    return found


_WEB_STEP = re.compile(r"\bweb_fetch\b|\bfetch\b.*\bhttps?://", re.IGNORECASE)


def _drop_unrequested_web_steps(plan: list[str], user_input: str) -> list[str]:
    """Remove web_fetch steps the user never asked for.

    The planner sometimes plans a fetch of a guessed URL (a Wikipedia page) for a
    general-knowledge question; that is slow, needs the network, and adds nothing.
    """
    if _TOOL_INTENT["web_fetch"].search(user_input):
        return plan
    kept = [step for step in plan if not _WEB_STEP.search(step)]
    if len(kept) == len(plan):
        return plan
    logger.info("unrequested_web_steps_dropped", plan=plan)
    return kept or ["Answer directly from knowledge"]


_PYTHON_BLOCK = re.compile(r"```(?:python|py)[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def _python_code(answer: str) -> str:
    """The Python code blocks of an answer, joined; empty if there are none."""
    return "\n\n".join(block.strip() for block in _PYTHON_BLOCK.findall(answer)).strip()


# An answer that announces a retry instead of doing it ("Let's try running the
# code again.", "Let me fix that and run it").
_PROMISED_RETRY = re.compile(
    r"\b(let[’']?s|let me|i[’']?ll|i will|i am going to|i[’']?m going to)\s+"
    r"(\w+\s+){0,3}(try|run|re-?run|fix|correct|execute|attempt)\b",
    re.IGNORECASE,
)


def _next_recovery(tool: str, answer: str, available: set[str]) -> str | None:
    """How to recover an answer that skipped the planned tool, if any way is left."""
    if tool == "python_exec" and "run_code" in available and _python_code(answer):
        return "run_code"
    if "nudge" in available:
        return "nudge"
    return None


_EXPLICIT_PYTHON = re.compile(r"\b(use|using)\s+python\b", re.IGNORECASE)
# "Write a calculator using Python" asks for code to read, not to run.
_WRITE_CODE = re.compile(
    r"\b(write|generate|create|build|make|fix|debug|refactor|correct)\b.*"
    r"\b(code|program|script|function|app|application|game|class|bug)\b",
    re.IGNORECASE | re.DOTALL,
)
_RUN_WORDS = re.compile(r"\b(run|execute)\b", re.IGNORECASE)
_CODE_REQUEST_START = re.compile(
    r"^\s*(please\s+)?(can you\s+)?(write|generate|create|build|make|develop|implement|code)\b",
    re.IGNORECASE,
)
_PYTHON_STEP = "Run the code with python_exec and report the output"


def _honour_explicit_python(plan: list[str], user_input: str) -> list[str]:
    """Add a python_exec step when the user said "use Python" but the plan has none.

    Seen: "Explain bubble sort vs merge sort, then use Python to time both" was
    planned as a direct answer, so no tools were offered and nothing was timed.
    Refusals are left alone.
    """
    if not _EXPLICIT_PYTHON.search(user_input) or not _is_tool_free_plan(plan):
        return plan
    if "refuse" in plan[0].lower():
        return plan
    # "Write a calculator using Python" asks for code to read; "Spell my name
    # backwards using Python" asks for a result.
    if _CODE_REQUEST_START.match(user_input) and not _RUN_WORDS.search(user_input):
        return plan
    logger.info("explicit_python_step_added", plan=plan)
    return [*plan, _PYTHON_STEP]


_RENAME_STEP = re.compile(r"\b(rename|move)\b", re.IGNORECASE)
_FILE_TARGET = re.compile(r"\bfilesystem\b|\bfiles?\b|\bfolders?\b|\S+\.\w{1,5}\b", re.IGNORECASE)
_UNSUPPORTED_REFUSAL = "Politely refuse: the filesystem tool cannot rename, move or delete files"


def _refuse_unsupported_file_ops(plan: list[str]) -> list[str]:
    """Refuse up front when the plan would rename or move files (no tool can).

    Seen: "Rename README.md to OLD.md" planned as a filesystem rename, two invalid
    tool calls, a stall and a LATS escalation: 67 s to say it could not be done.
    """
    for step in plan:
        if (
            _RENAME_STEP.search(step)
            and _FILE_TARGET.search(step)
            and not _KNOWLEDGE_STEP.match(step)
        ):
            logger.info("unsupported_file_op_refused", plan=plan)
            return [_UNSUPPORTED_REFUSAL]
    return plan


# "Save a Python script called hello.py", "Create a file called todo.md".
_SAVE_FILE = re.compile(
    r"\b(save|store)\b.*\.(py|txt|md|json|csv)\b"
    r"|\b(write|create|make)\s+(a\s+|the\s+)?(new\s+)?file\b",
    re.IGNORECASE | re.DOTALL,
)
_SAVE_STEP = "Write the requested file with the filesystem tool"


def _ensure_file_save_step(plan: list[str], user_input: str) -> list[str]:
    """Make an explicit "save it to hello.py" plan a filesystem write.

    Seen: "Save a Python script called hello.py that prints hello" planned as
    python_exec, which ran print('hello') and then claimed the file was saved.
    """
    if not _SAVE_FILE.search(user_input) or any("filesystem" in s.lower() for s in plan):
        return plan
    if _is_tool_free_plan(plan) and "refuse" in plan[0].lower():
        return plan
    kept = [
        step
        for step in plan
        if "python_exec" not in step.lower() and not step.lower().startswith("answer directly")
    ]
    logger.info("file_save_step_added", plan=plan)
    return [*kept, _SAVE_STEP]


_CODE_ANSWER_STEP = "Answer directly: write the complete code"


def _keep_code_requests_unrun(plan: list[str], user_input: str) -> list[str]:
    """Plan a written answer for "write a script that..." requests that never ask to run it.

    Seen: "Write a Python script that reads a CSV file and prints the average"
    planned as write-and-run, so the sandbox tried (and failed) to open a file.
    """
    if (
        _WRITE_CODE.search(user_input)
        and not _RUN_WORDS.search(user_input)
        and not _EXPLICIT_PYTHON.search(user_input)
        and any("python_exec" in step.lower() for step in plan)
    ):
        logger.info("code_request_kept_unrun", plan=plan)
        return [_CODE_ANSWER_STEP]
    return plan


def _format_history(history: list[Message]) -> str:
    """Compact the last few dialogue messages for the planner prompt."""
    recent = history[-_PLANNER_HISTORY_MESSAGES:]
    return "\n".join(
        f"{m.role.capitalize()}: {m.content[:_PLANNER_HISTORY_CHARS]}" for m in recent
    )


def _chunk_text(text: str, chunk_size: int = 80) -> list[str]:
    """Chunk a finished answer into token-like SSE payloads.

    Used only for answers that were not produced by real token streaming: the
    ``stream_tokens=False`` fallback, and LATS results (which are selected from a
    search tree post-hoc rather than generated in a single streamed pass).
    """
    if not text:
        return []
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
