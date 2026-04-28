"""Document search tool — index local documents into SemanticMemory."""

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

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
            "required": ["action"],
            "additionalProperties": False,
        },
    )

    def __init__(self, semantic_memory: SemanticMemory) -> None:
        self._semantic_memory = semantic_memory

    async def execute(self, **kwargs: object) -> ToolResult:
        """Execute a document search action."""
        start = time.monotonic()
        action = str(kwargs["action"])
        if action == "search":
            results = await self.search(
                str(kwargs.get("query", "")), _int_arg(kwargs.get("top_k"), 5)
            )
            output = json.dumps(results)
        elif action == "index_document":
            chunks = await self.index_document(
                str(kwargs["path"]),
                _int_arg(kwargs.get("chunk_size"), 512),
                _int_arg(kwargs.get("overlap"), 64),
            )
            output = f"Indexed {chunks} chunks."
        else:
            return _result(False, "", start, f"Unsupported action: {action}")
        return _result(True, output, start)

    async def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Search indexed document chunks by semantic similarity."""
        entries = await self._semantic_memory.retrieve(MemoryQuery(text=query, top_k=top_k))
        return [
            {
                "content": entry.content,
                "metadata": entry.metadata,
                "timestamp": entry.timestamp.isoformat(),
            }
            for entry in entries
        ]

    async def index_document(
        self, path: str, chunk_size: int = 512, overlap: int = 64
    ) -> int:
        """Split and index a local document into SemanticMemory."""
        doc_path = Path(path)
        content = await anyio.Path(doc_path).read_text()
        chunks = _chunk_document(content, chunk_size=chunk_size, overlap=overlap)
        for idx, chunk in enumerate(chunks):
            await self._semantic_memory.store(
                MemoryEntry(
                    id=uuid4().hex,
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
