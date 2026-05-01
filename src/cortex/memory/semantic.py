"""Semantic memory — long-term vector store backed by ChromaDB."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import chromadb
import structlog

from cortex.memory.base import BaseMemory, MemoryEntry, MemoryQuery
from cortex.memory.episodic import EpisodicMemory
from cortex.models.provider import GenerationConfig, Message, OllamaProvider
from cortex.observability.tracing import get_tracer

logger = structlog.get_logger(__name__)
_tracer = get_tracer(__name__)


class SemanticMemory(BaseMemory):
    """Cross-session semantic memory stored in ChromaDB."""

    def __init__(
        self,
        chroma_path: str | Path,
        embed_model: str,
        provider: OllamaProvider,
        episodic_memory: EpisodicMemory | None = None,
        collection_name: str = "cortex_semantic",
        client: Any | None = None,
        consolidation_model: str = "llama3.1:8b",
    ) -> None:
        self._chroma_path = Path(chroma_path)
        self._embed_model = embed_model
        self._provider = provider
        self._episodic_memory = episodic_memory
        self._collection_name = collection_name
        self._client = client
        self._collection: Any | None = None
        self._consolidation_model = consolidation_model

    async def initialize(self) -> None:
        """Create or connect to the cortex_semantic Chroma collection."""
        if self._client is None:
            self._chroma_path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self._chroma_path))
        self._collection = self._client.get_or_create_collection(self._collection_name)

    async def store(self, entry: MemoryEntry) -> str:
        """Embed content and upsert it into ChromaDB."""
        await self.initialize()
        assert self._collection is not None
        with _tracer.start_as_current_span("memory.semantic.store") as span:
            span.set_attribute("memory.type", entry.memory_type)
            span.set_attribute("memory.collection", self._collection_name)
            if "source_session" in entry.metadata:
                span.set_attribute("session_id", str(entry.metadata["source_session"]))
            embedding = (await self._provider.embed(self._embed_model, entry.content))[0]
            metadata = _metadata_for_chroma(entry)
            self._collection.upsert(
                ids=[entry.id],
                documents=[entry.content],
                embeddings=[embedding],
                metadatas=[metadata],
            )
        return entry.id

    async def retrieve(self, query: MemoryQuery) -> list[MemoryEntry]:
        """Retrieve semantically similar entries from ChromaDB."""
        await self.initialize()
        assert self._collection is not None
        with _tracer.start_as_current_span("memory.semantic.retrieve") as span:
            span.set_attribute("memory.collection", self._collection_name)
            span.set_attribute("memory.top_k", query.top_k)
            if query.session_id:
                span.set_attribute("session_id", query.session_id)
            embedding = (await self._provider.embed(self._embed_model, query.text))[0]
            result = self._collection.query(
                query_embeddings=[embedding],
                n_results=query.top_k,
            )
        return _entries_from_query_result(result)

    async def consolidate(self, session_id: str) -> None:
        """Extract durable facts from episodic history and store them as semantic memory."""
        if self._episodic_memory is None:
            logger.debug("semantic_consolidation_skipped_no_episodic", session_id=session_id)
            return
        history = await self._episodic_memory.get_session_history(session_id)
        if not history:
            return

        transcript = "\n".join(f"{m.role}: {m.content}" for m in history)
        messages = [
            Message(
                role="system",
                content=(
                    "Given this conversation, extract 3-5 atomic facts that would be "
                    "useful to remember in future sessions. Format as JSON array of strings."
                ),
            ),
            Message(role="user", content=transcript),
        ]
        with _tracer.start_as_current_span("memory.semantic.consolidate") as span:
            span.set_attribute("session_id", session_id)
            span.set_attribute("model_name", self._consolidation_model)
            response = await self._provider.complete(
                self._consolidation_model,
                messages,
                GenerationConfig(temperature=0.1, max_tokens=512),
            )
        try:
            facts = json.loads(response.content)
        except json.JSONDecodeError:
            logger.warning("semantic_consolidation_parse_failed", raw=response.content[:200])
            return
        if not isinstance(facts, list):
            return
        for fact in [str(f).strip() for f in facts if str(f).strip()][:5]:
            await self.store(
                MemoryEntry(
                    id=uuid4().hex,
                    content=fact,
                    metadata={"source_session": session_id},
                    timestamp=datetime.now(UTC),
                    memory_type="semantic",
                )
            )

    async def delete(self, entry_id: str) -> None:
        """Delete a semantic memory by id."""
        await self.initialize()
        assert self._collection is not None
        self._collection.delete(ids=[entry_id])


def _metadata_for_chroma(entry: MemoryEntry) -> dict[str, str | int | float | bool]:
    """Return Chroma-compatible scalar metadata for a MemoryEntry."""
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
    """Convert Chroma query output to MemoryEntry objects."""
    ids = result.get("ids", [[]])[0]
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0] if result.get("distances") else []
    entries: list[MemoryEntry] = []
    for idx, entry_id in enumerate(ids):
        metadata = dict(metadatas[idx] or {})
        if idx < len(distances):
            metadata["relevance_score"] = 1.0 / (1.0 + float(distances[idx]))
        timestamp = datetime.fromisoformat(
            str(metadata.pop("timestamp", datetime.now(UTC).isoformat()))
        )
        entries.append(
            MemoryEntry(
                id=entry_id,
                content=documents[idx],
                metadata=metadata,
                timestamp=timestamp,
                memory_type="semantic",
            )
        )
    return entries
