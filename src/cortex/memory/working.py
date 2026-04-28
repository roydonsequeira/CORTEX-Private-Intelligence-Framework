"""Working memory — transient in-context scratchpad."""

from collections import OrderedDict

from cortex.memory.base import BaseMemory, MemoryEntry, MemoryQuery


class WorkingMemory(BaseMemory):
    """In-memory scratchpad with bounded capacity."""

    def __init__(self, max_entries: int = 50) -> None:
        self._store: OrderedDict[str, MemoryEntry] = OrderedDict()
        self._max_entries = max_entries

    async def store(self, entry: MemoryEntry) -> str:
        """Store an entry and evict the oldest entry on overflow."""
        self._store[entry.id] = entry
        self._store.move_to_end(entry.id)
        while len(self._store) > self._max_entries:
            self._store.popitem(last=False)
        return entry.id

    async def retrieve(self, query: MemoryQuery) -> list[MemoryEntry]:
        """Retrieve entries by simple case-insensitive substring matching."""
        text = query.text.casefold()
        entries = list(self._store.values())
        if text:
            entries = [entry for entry in entries if text in entry.content.casefold()]
        return entries[-query.top_k :]

    async def delete(self, entry_id: str) -> None:
        """Delete an entry if present."""
        self._store.pop(entry_id, None)

    def clear(self) -> None:
        """Clear all working memory entries."""
        self._store.clear()
