"""Document search tool — index local documents into SemanticMemory."""

import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import anyio

from cortex.memory.base import MemoryEntry, MemoryQuery
from cortex.memory.semantic import SemanticMemory
from cortex.tools.base import BaseTool, ToolResult, ToolSchema


class DocumentSearchTool(BaseTool):
    """Search and index local documents using semantic similarity."""

    schema = ToolSchema(
        name="doc_search",
        description="Search through locally indexed documents using semantic similarity.",
        parameters={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["search", "index_document"]},
                "query": {"type": "string"},
                "path": {"type": "string"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
                "chunk_size": {"type": "integer", "minimum": 128, "maximum": 4096},
                "overlap": {"type": "integer", "minimum": 0, "maximum": 1024},
            },
            "additionalProperties": False,
        },
    )

    def __init__(
        self,
        semantic_memory: SemanticMemory,
        allowed_root: str | Path | None = None,
    ) -> None:
        self._semantic_memory = semantic_memory
        # Indexing reads files from disk, so it is confined to the same workspace
        # root as the filesystem tool (no indexing of ~/.ssh or other secrets).
        self._allowed_root = Path(allowed_root).resolve() if allowed_root is not None else None

    async def execute(self, **kwargs: object) -> ToolResult:
        """Execute a document search action."""
        start = time.monotonic()
        # Models frequently omit `action`; a query alone clearly means "search".
        action = str(kwargs.get("action") or ("index_document" if kwargs.get("path") else "search"))
        try:
            if action == "search":
                query = str(kwargs.get("query", "")).strip()
                if not query:
                    return _result(False, "", start, "The 'search' action requires a 'query'.")
                results = await self.search(query, _int_arg(kwargs.get("top_k"), 5))
                if not results:
                    return _result(
                        True,
                        "No indexed documents matched this query (index a document with "
                        "action='index_document' first).",
                        start,
                    )
                output = json.dumps(results, ensure_ascii=False)
            elif action == "index_document":
                if not kwargs.get("path"):
                    return _result(False, "", start, "The 'index_document' action requires a 'path'.")
                chunks = await self.index_document(
                    str(kwargs["path"]),
                    _int_arg(kwargs.get("chunk_size"), 512),
                    _int_arg(kwargs.get("overlap"), 64),
                )
                output = f"Indexed {chunks} chunks."
            else:
                return _result(False, "", start, f"Unsupported action: {action}")
        except (OSError, ValueError, TypeError) as exc:
            return _result(False, "", start, str(exc))
        return _result(True, output, start)

    async def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Search indexed document chunks by semantic similarity.

        Semantic memory also holds facts learned about the user; only chunks of
        indexed documents (which carry a ``source_path``) are returned here.
        """
        entries = await self._semantic_memory.retrieve(
            MemoryQuery(text=query, top_k=max(top_k * 4, 20))
        )
        documents = [entry for entry in entries if entry.metadata.get("source_path")]
        return [
            {
                "content": entry.content,
                "source": entry.metadata.get("source_path"),
                "relevance": round(float(entry.metadata.get("relevance_score", 0.0)), 3),
            }
            for entry in documents[:top_k]
        ]

    def _resolve(self, path: str) -> Path:
        """Resolve a document path, confined to the allowed root when one is set."""
        candidate = Path(path.strip())
        if self._allowed_root is None:
            return candidate
        resolved = (
            candidate.resolve()
            if candidate.is_absolute()
            else (self._allowed_root / candidate).resolve()
        )
        if not resolved.is_relative_to(self._allowed_root):
            raise OSError("Only documents inside the CORTEX workspace can be indexed.")
        return resolved

    async def index_document(
        self, path: str, chunk_size: int = 512, overlap: int = 64
    ) -> int:
        """Split and index a local UTF-8 document into SemanticMemory.

        Chunk ids are derived from the path and chunk index, so re-indexing the
        same document replaces its chunks instead of duplicating them.
        """
        doc_path = self._resolve(path)
        if not doc_path.is_file():
            raise OSError(f"Document not found: {path}")
        content = (await anyio.Path(doc_path).read_bytes()).decode("utf-8", errors="replace")
        chunks = _chunk_document(content, chunk_size=chunk_size, overlap=overlap)
        path_key = hashlib.sha1(str(doc_path).encode("utf-8")).hexdigest()[:16]
        for idx, chunk in enumerate(chunks):
            await self._semantic_memory.store(
                MemoryEntry(
                    id=f"doc-{path_key}-{idx}",
                    content=chunk,
                    metadata={
                        "source_path": str(doc_path),
                        "chunk_index": idx,
                        "total_chunks": len(chunks),
                    },
                    timestamp=datetime.now(UTC),
                    memory_type="semantic",
                )
            )
        return len(chunks)


def _chunk_document(content: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text on paragraph boundaries first, then character limits with overlap."""
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.extend(_sliding_chunks(paragraph, chunk_size, overlap))
        elif len(current) + len(paragraph) + 2 <= chunk_size:
            current = (current + "\n\n" + paragraph).strip()
        else:
            chunks.append(current.strip())
            current = paragraph
    if current:
        chunks.append(current.strip())
    return chunks


def _sliding_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split a long paragraph into overlapping character chunks."""
    chunks: list[str] = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(text):
        chunks.append(text[start : start + chunk_size].strip())
        start += step
    return [chunk for chunk in chunks if chunk]


def _int_arg(value: object, default: int) -> int:
    """Parse an integer tool argument with a default."""
    if value is None:
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value)
    raise TypeError(f"Expected integer-compatible value, got {type(value).__name__}.")


def _result(
    success: bool, output: str, start: float, error: str | None = None
) -> ToolResult:
    """Build a ToolResult for doc_search."""
    return ToolResult(
        tool_name="doc_search",
        success=success,
        output=output,
        error=error,
        execution_time_ms=(time.monotonic() - start) * 1000,
    )
