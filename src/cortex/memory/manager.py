"""MemoryManager — unified facade over CORTEX's four memory tiers."""

from datetime import UTC, datetime
from uuid import uuid4

import structlog

from cortex.memory.base import MemoryEntry, MemoryQuery
from cortex.memory.episodic import EpisodicMemory
from cortex.memory.procedural import ProceduralMemory, ToolPattern
from cortex.memory.semantic import SemanticMemory
from cortex.memory.working import WorkingMemory

logger = structlog.get_logger(__name__)


class MemoryManager:
    """Unified memory interface for the AgentKernel."""

    def __init__(
        self,
        working: WorkingMemory,
        episodic: EpisodicMemory,
        semantic: SemanticMemory,
        procedural: ProceduralMemory,
    ) -> None:
        self._working = working
        self._episodic = episodic
        self._semantic = semantic
        self._procedural = procedural

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

    async def retrieve_context(self, query: str, session_id: str) -> str:
        """Query episodic and semantic memory and return a formatted context block."""
        memory_query = MemoryQuery(text=query, top_k=5, session_id=session_id)
        episodic = await self._episodic.retrieve(memory_query)
        semantic = await self._semantic.retrieve(
            MemoryQuery(text=query, top_k=5, memory_types=["semantic"])
        )
        entries = _dedupe(episodic + semantic)
        if not entries:
            return ""
        lines = [f"- [{entry.memory_type}] {entry.content}" for entry in entries]
        return "--- Relevant Memory ---\n" + "\n".join(lines) + "\n---"

    async def end_session(self, session_id: str) -> None:
        """Consolidate session memory and clear working store."""
        await self._semantic.consolidate(session_id)
        self._working.clear()

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
        self, task: str, top_k: int = 3
    ) -> list[ToolPattern]:
        """Return learned tool-use patterns for similar tasks."""
        return await self._procedural.retrieve_patterns(task, top_k=top_k)


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
