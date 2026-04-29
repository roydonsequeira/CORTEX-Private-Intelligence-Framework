"""Memory inspection and document indexing routes."""

from typing import Any, cast

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, Field

from cortex.memory.base import MemoryQuery

router = APIRouter(prefix="/memory", tags=["memory"])


class IndexDocumentRequest(BaseModel):
    """Document indexing request."""

    path: str = Field(min_length=1)
    chunk_size: int = 512


@router.get("/search")
async def search_memory(
    request: Request,
    q: str = Query(..., min_length=1),
    types: str = "semantic,episodic",
    top_k: int = Query(10, ge=1, le=50),
) -> dict[str, Any]:
    """Search episodic and/or semantic memory."""
    requested = {item.strip() for item in types.split(",") if item.strip()}
    results = []
    if "episodic" in requested:
        entries = await request.app.state.episodic_memory.retrieve(
            MemoryQuery(text=q, top_k=top_k, memory_types=["episodic"])
        )
        results.extend(entries)
    if "semantic" in requested:
        entries = await request.app.state.semantic_memory.retrieve(
            MemoryQuery(text=q, top_k=top_k, memory_types=["semantic"])
        )
        results.extend(entries)
    return {"results": [entry.model_dump(mode="json") for entry in results[:top_k]]}


@router.get("/sessions")
async def list_sessions(request: Request) -> dict[str, list[str]]:
    """List all known episodic memory sessions."""
    return {"sessions": await request.app.state.episodic_memory.list_session_ids()}


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, request: Request) -> dict[str, str]:
    """Delete all episodic memory for a session."""
    await request.app.state.episodic_memory.delete_session(session_id)
    return {"status": "deleted", "session_id": session_id}


@router.post("/index")
async def index_document(payload: IndexDocumentRequest, request: Request) -> dict[str, Any]:
    """Index a local document through the doc_search tool."""
    result = await request.app.state.tool_registry.execute(
        "doc_search",
        action="index_document",
        path=payload.path,
        chunk_size=payload.chunk_size,
    )
    return cast(dict[str, Any], result.model_dump(mode="json"))
