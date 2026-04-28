"""Base abstractions for CORTEX memory stores."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

MemoryType = Literal["working", "episodic", "semantic", "procedural"]


class MemoryEntry(BaseModel):
    """A single memory item from any memory tier."""

    id: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime
    memory_type: MemoryType


class MemoryQuery(BaseModel):
    """Query envelope shared by all memory stores."""

    text: str
    top_k: int = 5
    memory_types: list[str] = Field(default_factory=lambda: ["semantic", "episodic"])
    session_id: str | None = None


class BaseMemory(ABC):
    """Abstract interface implemented by every memory tier."""

    @abstractmethod
    async def store(self, entry: MemoryEntry) -> str:
        """Store a memory entry and return its id."""
        ...

    @abstractmethod
    async def retrieve(self, query: MemoryQuery) -> list[MemoryEntry]:
        """Retrieve memory entries matching query."""
        ...

    @abstractmethod
    async def delete(self, entry_id: str) -> None:
        """Delete a memory entry by id."""
        ...
