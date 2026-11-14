"""One shared ChromaDB client per on-disk path.

Semantic and procedural memory both persist to ``settings.chroma_path``. Two
``chromadb.PersistentClient`` instances on the same directory each keep their
own view of segment state; after writes to both, queries start failing with
"Error creating hnsw segment reader: Nothing found on disk" (reproduced with
chromadb 1.5 on Windows). Chroma requires a single client per path, so every
memory tier obtains its client here.

The same error has a second cause in chromadb 1.5.9. Chroma writes a collection's
vector index to disk only every 1,000 items, so until then its in-process cache
holds the only copy, and a process rebuilds an index only the first time it reads
that collection. Once the cache evicts an index, every read of its collection
fails. The cache holds a fifth of the open-file limit, split over 64 shards. On
Windows (512 // 5 = 102) that is about one index per shard, so in about one store
in 64 the two memory collections share a shard and evict each other, again after
every restart. _raise_open_file_limit gives each shard room for several indexes.
"""

import contextlib
import sys
import threading
from pathlib import Path
from typing import Any

import chromadb

if sys.platform == "win32":
    import ctypes
else:
    import resource

# Chroma then caches 2048 // 5 = 409 indexes, at least 6 per shard. Windows allows
# no more than 2048.
_OPEN_FILE_LIMIT = 2048

_clients: dict[str, Any] = {}
_lock = threading.Lock()


def _raise_open_file_limit() -> None:
    """Raise the open-file limit Chroma sizes its index cache from, if it is lower."""
    if sys.platform == "win32":
        # The C runtime chromadb reads the limit from; Python's own files don't use it.
        crt = ctypes.windll.msvcrt
        if crt._getmaxstdio() < _OPEN_FILE_LIMIT:
            crt._setmaxstdio(_OPEN_FILE_LIMIT)
    else:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        limit = _OPEN_FILE_LIMIT if hard == resource.RLIM_INFINITY else min(_OPEN_FILE_LIMIT, hard)
        if soft < limit:
            # A sandbox may refuse; the store still opens, with Chroma's smaller cache.
            with contextlib.suppress(OSError, ValueError):
                resource.setrlimit(resource.RLIMIT_NOFILE, (limit, hard))


def get_chroma_client(path: str | Path) -> Any:
    """Return the process-wide PersistentClient for ``path``, creating it once."""
    resolved = Path(path).resolve()
    key = str(resolved).casefold()
    with _lock:
        client = _clients.get(key)
        if client is None:
            resolved.mkdir(parents=True, exist_ok=True)
            _raise_open_file_limit()
            client = chromadb.PersistentClient(path=str(resolved))
            _clients[key] = client
        return client
