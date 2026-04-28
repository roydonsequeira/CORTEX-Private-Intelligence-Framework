"""Procedural memory — reusable tool-use patterns backed by ChromaDB."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import chromadb
from pydantic import BaseModel

from cortex.memory.base import BaseMemory, MemoryEntry, MemoryQuery
from cortex.models.provider import OllamaProvider
from cortex.observability.tracing import get_tracer

_tracer = get_tracer(__name__)


class ToolPattern(BaseModel):
    """A learned sequence of tools for a class of tasks."""

    task_description: str
    tool_sequence: list[str]
    success: bool
    avg_steps: int


class ProceduralMemory(BaseMemory):
    """Stores and retrieves tool-use patterns for similar tasks."""

    def __init__(
        self,
        chroma_path: str | Path,
        embed_model: str,
        provider: OllamaProvider,
        collection_name: str = "cortex_procedural",
        client: Any | None = None,
    ) -> None:
        self._chroma_path = Path(chroma_path)
        self._embed_model = embed_model
        self._provider = provider
        self._collection_name = collection_name
        self._client = client
        self._collection: Any | None = None

    async def initialize(self) -> None:
        """Create or connect to the cortex_procedural Chroma collection."""
        if self._client is None:
            self._chroma_path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self._chroma_path))
        self._collection = self._client.get_or_create_collection(self._collection_name)

    async def store_pattern(self, pattern: ToolPattern) -> None:
        """Store a successful or failed tool-use pattern."""
        await self.store(
            MemoryEntry(
                id=uuid4().hex,
                content=pattern.task_description,
                metadata={
                    "tool_sequence": pattern.tool_sequence,
                    "success": pattern.success,
                    "avg_steps": pattern.avg_steps,
                },
                timestamp=datetime.now(UTC),
                memory_type="procedural",
            )
        )

    async def retrieve_patterns(self, task: str, top_k: int = 3) -> list[ToolPattern]:
        """Retrieve tool patterns for similar past tasks."""
        entries = await self.retrieve(
            MemoryQuery(text=task, top_k=top_k, memory_types=["procedural"])
        )
        patterns: list[ToolPattern] = []
        for entry in entries:
            raw_sequence = entry.metadata.get("tool_sequence", "[]")
            if isinstance(raw_sequence, str):
                tool_sequence = json.loads(raw_sequence)
            else:
                tool_sequence = raw_sequence
            patterns.append(
                ToolPattern(
                    task_description=entry.content,
                    tool_sequence=list(tool_sequence),
                    success=bool(entry.metadata.get("success", False)),
                    avg_steps=int(entry.metadata.get("avg_steps", 0)),
                )
            )
        return patterns

    async def store(self, entry: MemoryEntry) -> str:
        """Embed and upsert a procedural memory entry into ChromaDB."""
        await self.initialize()
        assert self._collection is not None
        with _tracer.start_as_current_span("memory.procedural.store"):
            embedding = (await self._provider.embed(self._embed_model, entry.content))[0]
            self._collection.upsert(
                ids=[entry.id],
                documents=[entry.content],
                embeddings=[embedding],
                metadatas=[_metadata_for_chroma(entry)],
            )
        return entry.id

    async def retrieve(self, query: MemoryQuery) -> list[MemoryEntry]:
        """Retrieve similar procedural memory entries."""
        await self.initialize()
        assert self._collection is not None
        with _tracer.start_as_current_span("memory.procedural.retrieve"):
            embedding = (await self._provider.embed(self._embed_model, query.text))[0]
            result = self._collection.query(
                query_embeddings=[embedding],
                n_results=query.top_k,
            )
        return _entries_from_query_result(result)

    async def delete(self, entry_id: str) -> None:
        """Delete a procedural memory by id."""
        await self.initialize()
        assert self._collection is not None
        self._collection.delete(ids=[entry_id])


def _metadata_for_chroma(entry: MemoryEntry) -> dict[str, str | int | float | bool]:
    """Return Chroma-compatible scalar metadata for a procedural entry."""
    metadata: dict[str, str | int | float | bool] = {}
    for key, value in entry.metadata.items():
        if isinstance(value, str | int | float | bool):
            metadata[key] = value
        elif value is not None:
            metadata[key] = json.dumps(value)
    metadata["timestamp"] = entry.timestamp.isoformat()
    metadata["memory_type"] = entry.memory_type
    return metadata


def _entries_from_query_result(result: dict[str, Any]) -> list[MemoryEntry]:
    """Convert Chroma procedural query output to MemoryEntry objects."""
    ids = result.get("ids", [[]])[0]
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    entries: list[MemoryEntry] = []
    for idx, entry_id in enumerate(ids):
        metadata = dict(metadatas[idx] or {})
        timestamp = datetime.fromisoformat(
            str(metadata.pop("timestamp", datetime.now(UTC).isoformat()))
        )
        entries.append(
            MemoryEntry(
                id=entry_id,
                content=documents[idx],
                metadata=metadata,
                timestamp=timestamp,
                memory_type="procedural",
            )
        )
    return entries
