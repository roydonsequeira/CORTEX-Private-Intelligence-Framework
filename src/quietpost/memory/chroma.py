"""One shared ChromaDB client per on-disk path.

Semantic and procedural memory both persist to ``settings.chroma_path``. Two
``chromadb.PersistentClient`` instances on the same directory each keep their
own view of segment state; after writes to both, queries start failing with
"Error creating hnsw segment reader: Nothing found on disk" (reproduced with
chromadb 1.5 on Windows). Chroma requires a single client per path, so every
memory tier obtains its client here.
"""

import threading
from pathlib import Path
from typing import Any

import chromadb

_clients: dict[str, Any] = {}
_lock = threading.Lock()


def get_chroma_client(path: str | Path) -> Any:
    """Return the process-wide PersistentClient for ``path``, creating it once."""
    resolved = Path(path).resolve()
    key = str(resolved).casefold()
    with _lock:
        client = _clients.get(key)
        if client is None:
            resolved.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(resolved))
            _clients[key] = client
        return client
