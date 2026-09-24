"""MemoryManager — unified facade over CORTEX's four memory tiers."""

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

import structlog

from cortex.memory.base import MemoryEntry, MemoryQuery
from cortex.memory.episodic import EpisodicMemory
from cortex.memory.procedural import ProceduralMemory, ToolPattern
from cortex.memory.semantic import SemanticMemory
from cortex.memory.working import WorkingMemory
from cortex.models.provider import Message

logger = structlog.get_logger(__name__)

# Characters of each replayed history message; keeps long answers or pasted
# documents from crowding the current request out of the context window.
_MAX_HISTORY_CHARS = 1500
_SEMANTIC_CONTEXT_TOP_K = 3


class MemoryManager:
    """Unified memory interface for the AgentKernel."""

    def __init__(
        self,
        working: WorkingMemory,
        episodic: EpisodicMemory,
        semantic: SemanticMemory,
        procedural: ProceduralMemory,
        consolidation_enabled: bool = True,
        min_relevance: float = 0.0,
    ) -> None:
        self._working = working
        self._episodic = episodic
        self._semantic = semantic
        self._procedural = procedural
        self._consolidation_enabled = consolidation_enabled
        self._min_relevance = min_relevance
        self._background: set[asyncio.Task[None]] = set()

    async def initialize(self) -> None:
        """Initialize all persistent memory stores."""
        await self._episodic.initialize()
        await self._semantic.initialize()
        await self._procedural.initialize()

    async def store_turn(self, session_id: str, role: str, content: str) -> None:
        """Persist a conversation turn to working and episodic memory."""
        entry_id = uuid4().hex
        now = datetime.now(UTC)
        metadata = {"session_id": session_id, "role": role}
        working_entry = MemoryEntry(
            id=entry_id,
            content=content,
            metadata=metadata,
            timestamp=now,
            memory_type="working",
        )
        episodic_entry = MemoryEntry(
            id=entry_id,
            content=content,
            metadata=metadata,
            timestamp=now,
            memory_type="episodic",
        )
        await self._working.store(working_entry)
        await self._episodic.store(episodic_entry)

    async def recent_history(self, session_id: str, max_turns: int = 6) -> list[Message]:
        """Return the last ``max_turns`` user/assistant exchanges of a session.

        Tool messages are omitted: they only make sense next to the tool call that
        produced them, and the assistant's final answer already summarises them.
        """
        if max_turns <= 0:
            return []
        history = await self._episodic.get_session_history(session_id)
        dialogue = [m for m in history if m.role in ("user", "assistant") and m.content.strip()]
        recent = dialogue[-max_turns * 2 :]
        return [
            Message(role=m.role, content=_clip(m.content, _MAX_HISTORY_CHARS)) for m in recent
        ]

    async def retrieve_context(self, query: str, session_id: str) -> str:
        """Return long-term (semantic) memories relevant to the query as a context block.

        The current session's own dialogue is replayed separately by
        ``recent_history``; this block carries facts learned in other sessions.
        """
        entries = await self._semantic.retrieve(
            MemoryQuery(text=query, top_k=_SEMANTIC_CONTEXT_TOP_K, memory_types=["semantic"])
        )
        relevant = [
            entry
            for entry in entries
            if float(entry.metadata.get("relevance_score", 1.0)) >= self._min_relevance
        ]
        entries = _dedupe(relevant)
        if not entries:
            return ""
        lines = [f"- {entry.content}" for entry in entries]
        return (
            "Long-term memory (facts about the user from earlier sessions — use only "
            "if relevant; when asked what you know about the user, include all of them):\n"
            + "\n".join(lines)
        )

    async def end_session(self, session_id: str) -> None:
        """Clear this session's working memory and consolidate it in the background.

        Consolidation calls the model, so it never runs on the request path: the
        answer is returned immediately and facts are extracted afterwards.
        """
        cleared = self._working.clear_session(session_id)
        logger.info("working_memory_cleared", session_id=session_id, entries=cleared)
        if not self._consolidation_enabled:
            return
        task = asyncio.create_task(self._consolidate(session_id))
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def _consolidate(self, session_id: str) -> None:
        """Run semantic consolidation, logging (never raising) on failure."""
        try:
            await self._semantic.consolidate(session_id)
        except Exception as exc:  # consolidation is best-effort
            logger.warning("memory_consolidation_failed", session_id=session_id, error=str(exc))

    async def drain(self, timeout_seconds: float = 10.0) -> None:
        """Wait briefly for background consolidation to finish (used at shutdown)."""
        if not self._background:
            return
        await asyncio.wait(set(self._background), timeout=timeout_seconds)

    async def store_tool_pattern(
        self,
        task_description: str,
        tool_sequence: list[str],
        success: bool,
        avg_steps: int,
    ) -> None:
        """Record a successful tool-use pattern for future retrieval."""
        await self._procedural.store_pattern(
            ToolPattern(
                task_description=task_description,
                tool_sequence=tool_sequence,
                success=success,
                avg_steps=avg_steps,
            )
        )

    async def retrieve_tool_patterns(
        self, task: str, top_k: int = 3, min_relevance: float = 0.0
    ) -> list[ToolPattern]:
        """Return learned tool-use patterns for similar tasks."""
        return await self._procedural.retrieve_patterns(
            task, top_k=top_k, min_relevance=min_relevance
        )


def _clip(text: str, limit: int) -> str:
    """Truncate text to ``limit`` characters with a marker."""
    return text if len(text) <= limit else text[:limit] + " …[truncated]"


def _dedupe(entries: list[MemoryEntry]) -> list[MemoryEntry]:
    """Deduplicate memory entries by normalized content, preserving order."""
    seen: set[str] = set()
    deduped: list[MemoryEntry] = []
    for entry in entries:
        key = " ".join(entry.content.casefold().split())
        if key and key not in seen:
            seen.add(key)
            deduped.append(entry)
    return deduped
