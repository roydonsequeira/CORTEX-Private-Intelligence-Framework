"""Semantic memory — long-term vector store backed by ChromaDB."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

from cortex.memory.base import BaseMemory, MemoryEntry, MemoryQuery
from cortex.memory.chroma import get_chroma_client
from cortex.memory.episodic import EpisodicMemory
from cortex.models.parsing import extract_json
from cortex.models.provider import GenerationConfig, Message, OllamaProvider
from cortex.observability.tracing import get_tracer

logger = structlog.get_logger(__name__)
_tracer = get_tracer(__name__)

_MIN_CONSOLIDATION_CHARS = 12
_CONSOLIDATION_PROMPT = (
    "You maintain long-term memory about the USER for a personal AI assistant. "
    "From this exchange, extract at most 3 durable facts about the user — their "
    "name, role, preferences, projects, goals, or stated decisions. Each fact is a "
    "short standalone sentence in the third person (e.g. \"The user's name is Sam.\"). "
    "Do NOT include general knowledge, calculation results, or anything about the "
    "assistant. If there is nothing worth remembering, return []. "
    "Respond with ONLY a JSON array of strings."
)


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
        if self._collection is not None:
            return
        if self._client is None:
            self._client = get_chroma_client(self._chroma_path)
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
        available = self._collection.count()
        if available == 0:
            return []  # nothing stored yet: skip the embedding round-trip
        with _tracer.start_as_current_span("memory.semantic.retrieve") as span:
            span.set_attribute("memory.collection", self._collection_name)
            span.set_attribute("memory.top_k", query.top_k)
            if query.session_id:
                span.set_attribute("session_id", query.session_id)
            embedding = (await self._provider.embed(self._embed_model, query.text))[0]
            result = self._collection.query(
                query_embeddings=[embedding],
                n_results=min(query.top_k, available),
            )
        return _entries_from_query_result(result)

    async def consolidate(self, session_id: str, last_messages: int = 4) -> None:
        """Extract durable facts from the latest exchange and store them as semantic memory.

        Only the most recent ``last_messages`` user/assistant messages are read, so
        each exchange is consolidated once instead of re-reading the whole session
        every turn. Fact ids are derived from the normalised text, so re-learning
        the same fact updates it instead of duplicating it.
        """
        if self._episodic_memory is None:
            logger.debug("semantic_consolidation_skipped_no_episodic", session_id=session_id)
            return
        history = await self._episodic_memory.get_session_history(session_id)
        dialogue = [m for m in history if m.role in ("user", "assistant")][-last_messages:]
        user_text = " ".join(m.content for m in dialogue if m.role == "user")
        if len(user_text.strip()) < _MIN_CONSOLIDATION_CHARS:
            return  # greetings and one-word turns carry nothing worth remembering

        transcript = "\n".join(f"{m.role}: {m.content[:2000]}" for m in dialogue)
        messages = [
            Message(role="system", content=_CONSOLIDATION_PROMPT),
            Message(role="user", content=transcript),
        ]
        with _tracer.start_as_current_span("memory.semantic.consolidate") as span:
            span.set_attribute("session_id", session_id)
            span.set_attribute("model_name", self._consolidation_model)
            response = await self._provider.complete(
                self._consolidation_model,
                messages,
                GenerationConfig(temperature=0.1, max_tokens=256),
            )
        facts = extract_json(response.content)
        if not isinstance(facts, list):
            logger.warning("semantic_consolidation_parse_failed", raw=response.content[:200])
            return
        for fact in [str(f).strip() for f in facts if str(f).strip()][:3]:
            await self.store(
                MemoryEntry(
                    id=_fact_id(fact),
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


def _fact_id(fact: str) -> str:
    """Stable id for a fact so the same fact learned twice is upserted, not duplicated."""
    normalised = " ".join(fact.casefold().split())
    return "fact-" + hashlib.sha1(normalised.encode("utf-8")).hexdigest()


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
