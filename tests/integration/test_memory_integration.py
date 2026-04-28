"""Integration tests for Chroma-backed memory."""

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import chromadb
import pytest

from cortex.memory.base import MemoryEntry, MemoryQuery
from cortex.memory.semantic import SemanticMemory


class _EmbeddingProvider:
    """Deterministic embedding provider for Chroma integration tests."""

    async def embed(self, model: str, text: str | list[str]) -> list[list[float]]:
        texts = [text] if isinstance(text, str) else text
        return [[1.0, 0.0, 0.0] if "alpha" in t else [0.0, 1.0, 0.0] for t in texts]


@pytest.mark.asyncio
async def test_semantic_memory_retrieves_top_three_by_similarity(tmp_path: Path) -> None:
    """SemanticMemory stores entries in Chroma and retrieves the most relevant three."""
    provider = _EmbeddingProvider()
    memory = SemanticMemory(
        tmp_path / "chroma",
        "nomic-embed-text",
        provider,  # type: ignore[arg-type]
        client=chromadb.EphemeralClient(),
    )
    await memory.initialize()

    for i in range(10):
        content = f"alpha document {i}" if i < 5 else f"beta document {i}"
        await memory.store(
            MemoryEntry(
                id=uuid4().hex,
                content=content,
                metadata={"source_session": "integration"},
                timestamp=datetime.now(UTC),
                memory_type="semantic",
            )
        )

    results = await memory.retrieve(MemoryQuery(text="alpha query", top_k=3))

    assert len(results) == 3
    assert all("alpha" in entry.content for entry in results)
