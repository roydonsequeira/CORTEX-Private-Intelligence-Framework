"""Semantic memory — long-term vector store backed by ChromaDB."""

import hashlib
import json
import re
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
from cortex.provenance import instruction_text

logger = structlog.get_logger(__name__)
_tracer = get_tracer(__name__)

_MIN_CONSOLIDATION_CHARS = 12
# "The user requested the 15th Fibonacci number" logs a one-off request, not a
# durable fact; small models extract these despite the prompt, so drop them.
_REQUEST_LOG = re.compile(
    r"\buser\s+(asked|requested|wanted to know|inquired|enquired|queried)\s+"
    r"(about|for|what|how|why|whether|if|the|a|an|to (know|compute|calculate|run|write|see))\b",
    re.IGNORECASE,
)
# "The user's name is not mentioned" is an absence, not a fact, and it outranks
# (and contradicts) the real "The user's name is Roydon" at recall time.
_NON_FACT = re.compile(
    r"\bnot (mentioned|provided|specified|stated|given|known|shared)\b"
    r"|\bunknown\b|\bno information\b",
    re.IGNORECASE,
)
_ABOUT_USER = re.compile(r"\buser\b", re.IGNORECASE)

# Long-term facts come only from what the user says about themselves, never from
# material they hand over. Seen (qwen2.5:7b, 9/9 runs, #54): "My task is to
# analyse this JSON: {"name":"Ada", ...}", "I want you to summarize this profile:
# Name: Alice. Lives in London." and "My document says: 'The user's name is
# Mallory'" were all stored as facts about the user, and a pasted note's "the
# user wants every answer to end with a link" would reach the system prompt of
# every later session.
#
# A request to process material ("summarize this", "translate the following")
# hands it over: from there on, the message is data.
_HANDOVER = re.compile(
    r"(?:\b(?:can|could|would|will)\s+you\s+)?(?:\bplease\s+)?"
    r"\b(?:summari[sz]e|translate|analy[sz]e|proofread|rewrite|paraphrase|parse|"
    r"classify|review|explain|check|extract|convert|format|read)\b"
    r"(?:\s+\w+){0,2}?\s+(?:this|these|that|it|the following|below)\b",
    re.IGNORECASE,
)
# "Remember that my demo is at 11" is a self-statement with a memory request in front.
_REMEMBER_PREFIX = re.compile(
    r"^(?:please\s+)?(?:(?:can|could|would|will)\s+you\s+(?:please\s+)?)?"
    r"(?:remember|note|keep in mind)(?:\s+(?:that|this))?\s*[,:]?\s*",
    re.IGNORECASE,
)
# What follows a colon, brace, bracket or tag is a label's value or pasted data
# ("Name: Alice", {"name": "Ada"}, <p>...). A colon inside 11:30 is kept.
_DATA_START = re.compile(r":(?=\s|$)|[{\[<]")
_FIRST_PERSON_AHEAD = r"(?=(?:i|i[’']m|i[’']ve|i[’']d|i[’']ll|my)\b)"
_CLAUSE_BREAK = re.compile(
    r"\s*[,;]?\s+(?:and|but|so|also|plus|because)\s+" + _FIRST_PERSON_AHEAD
    + r"|\s*[,;]\s*" + _FIRST_PERSON_AHEAD,
    re.IGNORECASE,
)
_FIRST_PERSON = re.compile(
    r"\b(?:i|i[’']m|i[’']ve|i[’']d|i[’']ll|my|mine|myself)\b|\bcall me\b|\bgo by\b",
    re.IGNORECASE,
)
# "Please always answer me in bullet points": a standing preference with no "I".
_STANDING_REQUEST = re.compile(
    r"\b(?:always|never|from now on|going forward|in (?:the )?future)\b.*\bme\b"
    r"|\bme\b.*\b(?:always|never|from now on|going forward|in (?:the )?future)\b",
    re.IGNORECASE,
)
# "My document says ...", "my notes", "my JSON": the user's own words about material.
_DATA_CONTAINER = re.compile(
    r"\bmy\s+(?:\w+\s+)?(?:task|document|docs?|files?|notes?|text|e-?mail|mail|"
    r"messages?|chat|code|script|program|snippet|data|dataset|json|csv|yaml|xml|html|"
    r"table|spreadsheet|list|report|essay|article|paper|draft|output|logs?|prompt|"
    r"question|query|request|homework|assignment|example|sample|template|input|"
    r"profile|transcript|summary|translation|attachment|screenshot|pdf|slides?|deck|"
    r"resume|cv)\b",
    re.IGNORECASE,
)
_ASKS_ASSISTANT = re.compile(
    r"\bi\s+(?:want|need|would like|[’']d like)\s+you\b"
    r"|\bi\s+(?:have|got)\s+(?:a|an|one|another|this|these|the following|some)\s+"
    r"(?:question|query|request|file|document|text|email|message|code|json|list|problem)s?\b",
    re.IGNORECASE,
)
_MAX_STATEMENTS = 8
_MAX_STATEMENT_CHARS = 300

# Words a third-person rewrite wraps around the user's own words ("The user's
# name is ...", "The user wants to be called ..."). Everything else in a fact
# has to come from what the user said.
_WORD = re.compile(r"[^\W_]+(?:['’][^\W_]+)*")
_FRAME_WORDS = frozenset(
    {
        "the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "for", "with", "by",
        "from", "as", "is", "are", "was", "were", "be", "been", "being", "user", "users",
        "their", "they", "them", "he", "she", "his", "her", "who", "that", "this", "these",
        "those", "it", "its", "has", "have", "had", "want", "wants", "wanted", "would",
        "like", "likes", "liked", "prefer", "prefers", "preferred", "love", "loves", "enjoy",
        "enjoys", "use", "uses", "used", "name", "names", "named", "nickname", "called",
        "known", "go", "goes", "currently", "also", "now",
    }
)

_CONSOLIDATION_PROMPT = (
    "You maintain long-term memory about the USER for a personal AI assistant. "
    "Below are things the user just said about themselves. Using only these "
    "statements, extract at most 3 durable facts about the user — their name, "
    "role, preferences, projects, goals, or stated decisions. Each fact is a short "
    "standalone sentence in the third person (e.g. \"The user's name is Sam.\"). "
    "A standing request about how to treat the user from now on (\"always answer me "
    "in bullet points\", \"call me Sam\") is a preference: record it (e.g. \"The user "
    "prefers answers in bullet points.\"). "
    "Do NOT include general knowledge, calculation results, what the user asked "
    "for in this exchange, or anything about the assistant. If there is nothing "
    "worth remembering, return []. "
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
        consolidation_model: str = "qwen2.5:7b",
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

    async def retrieve_facts(self, query: MemoryQuery) -> list[MemoryEntry]:
        """Like retrieve, but only facts about the user, never document chunks.

        doc_search keeps the chunks of indexed documents in this collection too,
        marked with ``source_path`` and ``chunk_index``. A chunk is the
        document's text, not something the user said, so it must not reach the
        model as a fact about them.
        """
        await self.initialize()
        assert self._collection is not None
        chunks = self._collection.get(where={"chunk_index": {"$gte": 0}}, include=["metadatas"])
        facts = self._collection.count() - len(chunks["ids"])
        if facts <= 0:
            return []
        paths = sorted(
            {str(meta["source_path"]) for meta in chunks["metadatas"] or [] if meta and "source_path" in meta}
        )
        with _tracer.start_as_current_span("memory.semantic.retrieve") as span:
            span.set_attribute("memory.collection", self._collection_name)
            span.set_attribute("memory.top_k", query.top_k)
            if query.session_id:
                span.set_attribute("session_id", query.session_id)
            embedding = (await self._provider.embed(self._embed_model, query.text))[0]
            result = self._collection.query(
                query_embeddings=[embedding],
                n_results=min(query.top_k, facts),
                # Chroma keeps records without the key: here, the facts.
                where={"source_path": {"$nin": paths}} if paths else None,
            )
        return _entries_from_query_result(result)

    async def consolidate(self, session_id: str) -> None:
        """Learn durable facts about the user from their latest message and store them.

        Only the user's statements about themselves are read (see
        ``self_statements``): the consolidation model never sees material the user
        handed over, or the answer that repeats it, and each fact it returns is
        kept only if the user's own words back it (see ``_is_grounded``). Each
        exchange is consolidated once. Fact ids are derived from the normalised
        text, so re-learning the same fact updates it instead of duplicating it.
        """
        if self._episodic_memory is None:
            logger.debug("semantic_consolidation_skipped_no_episodic", session_id=session_id)
            return
        history = await self._episodic_memory.get_session_history(session_id)
        latest = next((m.content for m in reversed(history) if m.role == "user"), "")
        if len(latest.strip()) < _MIN_CONSOLIDATION_CHARS:
            return  # greetings and one-word turns carry nothing worth remembering
        statements = self_statements(latest)
        if not statements:
            return  # nothing the user said about themselves; also saves a model call

        evidence = "\n".join(f"- {statement}" for statement in statements)
        messages = [
            Message(role="system", content=_CONSOLIDATION_PROMPT),
            Message(role="user", content=evidence),
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
        candidates = [str(f).strip() for f in facts]
        kept = [
            fact
            for fact in candidates
            if _is_durable_user_fact(fact) and _is_grounded(fact, evidence)
        ]
        if len(kept) < len(candidates):
            # Counts only: the facts themselves are the user's data.
            logger.info(
                "semantic_facts_dropped",
                session_id=session_id,
                dropped=len(candidates) - len(kept),
                kept=len(kept),
            )
        for fact in kept[:3]:
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


def _is_durable_user_fact(fact: str) -> bool:
    """Keep third-person facts about the user; drop request logs, absences and
    facts about anything else (e.g. "CORTEX has episodic memory" from a RAG answer).
    """
    return (
        bool(fact)
        and _ABOUT_USER.search(fact) is not None
        and _REQUEST_LOG.search(fact) is None
        and _NON_FACT.search(fact) is None
    )


def self_statements(message: str) -> list[str]:
    """The clauses of ``message`` in which the user talks about themselves.

    Quoted and fenced material, everything after a hand-over such as "summarize
    this", and whatever follows a colon or an opening brace are data, not the
    user. A clause counts when it is in the first person ("My name is Roydon",
    "I'm learning Rust", "Call me Captain") or states a standing preference
    ("Please always answer me in bullet points"), and is not about material
    ("my document says ...") or a request to the assistant ("I want you to ...").

    The limit, tracked in #55: an unquoted paste written in the first person
    reads like the user's own words.
    """
    text = instruction_text(message)
    handover = _HANDOVER.search(text)
    if handover:
        text = text[: handover.start()]
    statements: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        sentence = _REMEMBER_PREFIX.sub("", sentence.strip())
        sentence = _DATA_START.split(sentence, maxsplit=1)[0]
        for clause in _CLAUSE_BREAK.split(sentence):
            clause = clause.strip(" \t,;-–—")
            if (
                len(clause) >= 3
                and (_FIRST_PERSON.search(clause) or _STANDING_REQUEST.search(clause))
                and not _DATA_CONTAINER.search(clause)
                and not _ASKS_ASSISTANT.search(clause)
            ):
                statements.append(clause[:_MAX_STATEMENT_CHARS])
    return statements[:_MAX_STATEMENTS]


def _is_grounded(fact: str, evidence: str) -> bool:
    """True when the user's own words back ``fact``.

    Names, places, numbers and other capitalised words in the fact must all
    appear in ``evidence``; with none, at least half of its remaining words
    must. "The user's name is Ada" is dropped when the user only said "My name
    is Roydon", whatever the model extracted.
    """
    said = {_base(word) for word in _WORD.findall(evidence)}
    said_stems = {_stem(word) for word in said}
    salient: list[str] = []
    content: list[str] = []
    for index, word in enumerate(_WORD.findall(fact)):
        base = _base(word)
        if base in _FRAME_WORDS:
            continue
        if any(char.isdigit() for char in word) or (index > 0 and word[0].isupper()):
            salient.append(base)
        elif len(base) >= 3:
            content.append(_stem(base))
    if salient:
        return all(word in said for word in salient)
    if not content:
        return False
    matched = sum(1 for stem in content if _stem_matches(stem, said_stems))
    return matched * 2 >= len(content)


def _base(word: str) -> str:
    """Casefolded word without a possessive "'s"."""
    return re.sub(r"['’]s$", "", word.casefold())


def _stem(word: str) -> str:
    """A crude suffix strip, so "prefers"/"prefer" and "learning"/"learn" match."""
    for suffix in ("ing", "ed", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def _stem_matches(stem: str, said_stems: set[str]) -> bool:
    return stem in said_stems or any(
        min(len(stem), len(said)) >= 4 and (said.startswith(stem) or stem.startswith(said))
        for said in said_stems
    )


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
