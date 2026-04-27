"""MemoryManager stub — full implementation in Phase 2.

The kernel uses this interface; the stub keeps Phase 1 self-contained.
"""

import structlog

logger = structlog.get_logger(__name__)


class MemoryManager:
    """Unified memory interface for the AgentKernel.

    Stub: all operations are no-ops. Phase 2 replaces this with the full
    four-tier implementation (working, episodic, semantic, procedural).
    """

    async def store_turn(self, session_id: str, role: str, content: str) -> None:
        """Persist a conversation turn to working and episodic memory."""

    async def retrieve_context(self, query: str, session_id: str) -> str:
        """Query episodic and semantic memory and return a formatted context block."""
        return ""

    async def end_session(self, session_id: str) -> None:
        """Consolidate session memory and clear working store."""

    async def store_tool_pattern(
        self,
        task_description: str,
        tool_sequence: list[str],
        success: bool,
        avg_steps: int,
    ) -> None:
        """Record a successful tool-use pattern for future retrieval."""
