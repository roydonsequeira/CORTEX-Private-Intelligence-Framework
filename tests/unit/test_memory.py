"""Unit tests for CORTEX memory tiers."""

from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock
from uuid import uuid4

import chromadb
import pytest

from cortex.memory.base import MemoryEntry, MemoryQuery
from cortex.memory.episodic import EpisodicMemory
from cortex.memory.manager import MemoryManager
from cortex.memory.procedural import ProceduralMemory
from cortex.memory.semantic import SemanticMemory
from cortex.memory.working import WorkingMemory
from cortex.models.provider import ModelResponse, OllamaProvider


def _entry(content: str, memory_type: str = "working", **metadata: object) -> MemoryEntry:
    return MemoryEntry(
        id=uuid4().hex,
        content=content,
        metadata=metadata,
        timestamp=datetime.now(UTC),
        memory_type=memory_type,  # type: ignore[arg-type]
    )


def _mock_provider() -> OllamaProvider:
    provider = AsyncMock(spec=OllamaProvider)
    provider.embed = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    provider.complete = AsyncMock(
        return_value=ModelResponse(
            content='["User prefers local-only AI.", "Project is named CORTEX."]',
            model="llama3.1:8b",
            input_tokens=1,
            output_tokens=1,
            latency_ms=1.0,
            raw={},
        )
    )
    return provider


@pytest.mark.asyncio
async def test_working_memory_drops_oldest_on_overflow() -> None:
    """WorkingMemory evicts oldest entries beyond capacity."""
    memory = WorkingMemory(max_entries=2)
    await memory.store(_entry("first"))
    await memory.store(_entry("second"))
    await memory.store(_entry("third"))

    results = await memory.retrieve(MemoryQuery(text="", top_k=10))
    assert [entry.content for entry in results] == ["second", "third"]


@pytest.mark.asyncio
async def test_episodic_memory_stores_and_retrieves_session(tmp_path: Path) -> None:
    """EpisodicMemory stores and retrieves entries scoped to a session."""
    memory = EpisodicMemory(tmp_path / "cortex.db")
    await memory.initialize()
    await memory.store(
        _entry(
            "hello from session a",
            "episodic",
            session_id="session-a",
            role="user",
        )
    )
    await memory.store(
        _entry(
            "hello from session b",
            "episodic",
            session_id="session-b",
            role="assistant",
        )
    )

    results = await memory.retrieve(MemoryQuery(text="hello", session_id="session-a"))
    assert len(results) == 1
    assert results[0].content == "hello from session a"
    assert results[0].metadata["role"] == "user"


@pytest.mark.asyncio
async def test_episodic_memory_returns_ordered_history(tmp_path: Path) -> None:
    """Session history is returned in chronological order."""
    memory = EpisodicMemory(tmp_path / "cortex.db")
    await memory.initialize()
    await memory.store(_entry("first", "episodic", session_id="s1", role="user"))
    await memory.store(_entry("second", "episodic", session_id="s1", role="assistant"))

    history = await memory.get_session_history("s1")
    assert [message.content for message in history] == ["first", "second"]


@pytest.mark.asyncio
async def test_semantic_memory_consolidate_extracts_facts(tmp_path: Path) -> None:
    """SemanticMemory.consolidate extracts facts via the model and stores them."""
    provider = _mock_provider()
    episodic = EpisodicMemory(tmp_path / "cortex.db")
    await episodic.initialize()
    await episodic.store(_entry("I prefer local AI.", "episodic", session_id="s1", role="user"))

    semantic = SemanticMemory(
        tmp_path / "chroma",
        "nomic-embed-text",
        provider,
        episodic_memory=episodic,
        client=chromadb.EphemeralClient(),
    )

    await semantic.consolidate("s1")
    results = await semantic.retrieve(MemoryQuery(text="local AI", top_k=5))

    assert len(results) >= 1
    assert any("local-only AI" in entry.content for entry in results)


@pytest.mark.asyncio
async def test_memory_manager_retrieve_context_dedupes(tmp_path: Path) -> None:
    """MemoryManager.retrieve_context deduplicates overlapping memory entries."""
    provider = _mock_provider()
    episodic = EpisodicMemory(tmp_path / "cortex.db")
    semantic = SemanticMemory(
        tmp_path / "semantic",
        "nomic-embed-text",
        provider,
        episodic_memory=episodic,
        client=chromadb.EphemeralClient(),
    )
    procedural = ProceduralMemory(
        tmp_path / "procedural",
        "nomic-embed-text",
        provider,
        client=chromadb.EphemeralClient(),
    )
    manager = MemoryManager(WorkingMemory(), episodic, semantic, procedural)
    await manager.initialize()

    await manager.store_turn("s1", "user", "The project is called CORTEX.")
    await semantic.store(
        _entry(
            "The project is called CORTEX.",
            "semantic",
            source_session="s1",
        )
    )

    context = await manager.retrieve_context("project", "s1")
    assert context.count("The project is called CORTEX.") == 1


@pytest.mark.asyncio
async def test_semantic_and_procedural_share_one_chroma_client(tmp_path: Path) -> None:
    """Two PersistentClients on one path corrupt each other's HNSW view (chromadb 1.5
    raises "Nothing found on disk"); both tiers must reuse a single client."""
    provider = _mock_provider()
    semantic = SemanticMemory(tmp_path / "chroma", "nomic-embed-text", provider)
    procedural = ProceduralMemory(tmp_path / "chroma", "nomic-embed-text", provider)
    await semantic.initialize()
    await procedural.initialize()
    assert semantic._client is procedural._client

    for i in range(10):
        await procedural.store(_entry(f"task {i}", "procedural", tool_sequence=["calc"]))
        await semantic.store(_entry(f"fact {i}", "semantic"))
        assert await procedural.retrieve(MemoryQuery(text="task", top_k=3))
        assert await semantic.retrieve(MemoryQuery(text="fact", top_k=3))


def _manager(tmp_path: Path, provider: OllamaProvider, semantic: SemanticMemory) -> MemoryManager:
    episodic = EpisodicMemory(tmp_path / "cortex.db")
    procedural = ProceduralMemory(
        tmp_path / "procedural", "nomic-embed-text", provider, client=chromadb.EphemeralClient()
    )
    return MemoryManager(WorkingMemory(), episodic, semantic, procedural)


@pytest.mark.asyncio
async def test_recent_history_returns_dialogue_without_tool_messages(tmp_path: Path) -> None:
    """recent_history replays user/assistant turns in order and drops tool output."""
    provider = _mock_provider()
    semantic = SemanticMemory(
        tmp_path / "semantic", "nomic-embed-text", provider, client=chromadb.EphemeralClient()
    )
    manager = _manager(tmp_path, provider, semantic)
    await manager.initialize()
    await manager.store_turn("s1", "user", "My name is Roydon.")
    await manager.store_turn("s1", "tool", "[python_exec] 42")
    await manager.store_turn("s1", "assistant", "Nice to meet you, Roydon.")
    await manager.store_turn("s2", "user", "other session")

    history = await manager.recent_history("s1", max_turns=5)

    assert [(m.role, m.content) for m in history] == [
        ("user", "My name is Roydon."),
        ("assistant", "Nice to meet you, Roydon."),
    ]


@pytest.mark.asyncio
async def test_consolidation_is_deduplicated_and_skips_trivial_turns(tmp_path: Path) -> None:
    """Re-learning a fact upserts it; greetings are not consolidated at all."""
    provider = _mock_provider()
    episodic = EpisodicMemory(tmp_path / "cortex.db")
    await episodic.initialize()
    semantic = SemanticMemory(
        tmp_path / "chroma",
        "nomic-embed-text",
        provider,
        episodic_memory=episodic,
        client=chromadb.EphemeralClient(),
        collection_name="dedupe_test",
    )
    await episodic.store(_entry("hi", "episodic", session_id="greet", role="user"))
    await semantic.consolidate("greet")
    cast(AsyncMock, provider.complete).assert_not_awaited()

    await episodic.store(
        _entry("I prefer local AI for privacy.", "episodic", session_id="s1", role="user")
    )
    await semantic.consolidate("s1")
    await semantic.consolidate("s1")
    assert semantic._collection is not None
    assert semantic._collection.count() == 2  # two distinct facts, each stored once


@pytest.mark.asyncio
async def test_end_session_consolidates_in_background(tmp_path: Path) -> None:
    """end_session returns immediately; consolidation errors are swallowed."""
    provider = _mock_provider()
    semantic = SemanticMemory(
        tmp_path / "semantic", "nomic-embed-text", provider, client=chromadb.EphemeralClient()
    )
    consolidate = AsyncMock(side_effect=RuntimeError("model offline"))
    semantic.consolidate = consolidate  # type: ignore[method-assign]
    manager = _manager(tmp_path, provider, semantic)
    await manager.initialize()

    await manager.end_session("s1")
    await manager.drain()

    consolidate.assert_awaited_once_with("s1")
