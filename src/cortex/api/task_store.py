"""Pluggable persistence for async task results.

The API queues fire-and-forget agent tasks and reports their status/result by
``task_id``. In-memory storage is fine for a single localhost process but loses
everything on restart; the SQLite backend makes results durable with zero extra
infrastructure. Select the backend with ``settings.task_store``.
"""

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Protocol

import aiosqlite

from cortex.config.settings import Settings


class TaskStore(Protocol):
    """Async key/value store for task records keyed by task_id."""

    async def initialize(self) -> None:
        """Prepare the backing store (create tables, etc.)."""
        ...

    async def set(self, task_id: str, record: dict[str, Any]) -> None:
        """Insert or replace the record for a task."""
        ...

    async def get(self, task_id: str) -> dict[str, Any] | None:
        """Return the record for a task, or None if unknown."""
        ...


class InMemoryTaskStore:
    """Process-local task store. Fast, but results are lost on restart.

    Keeps the most recent ``max_records`` tasks so a long-running server does not
    grow without bound.
    """

    def __init__(self, max_records: int = 1000) -> None:
        self._records: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._max_records = max_records

    async def initialize(self) -> None:
        """No-op: the in-memory store needs no setup."""
        return None

    async def set(self, task_id: str, record: dict[str, Any]) -> None:
        """Store the record in the process dictionary, evicting the oldest."""
        self._records[task_id] = record
        self._records.move_to_end(task_id)
        while len(self._records) > self._max_records:
            self._records.popitem(last=False)

    async def get(self, task_id: str) -> dict[str, Any] | None:
        """Return the record from the process dictionary."""
        return self._records.get(task_id)


class SQLiteTaskStore:
    """Durable task store backed by SQLite, so results survive restarts."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)

    async def initialize(self) -> None:
        """Create the tasks table if it does not exist."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                  task_id TEXT PRIMARY KEY,
                  record JSON NOT NULL
                )
                """
            )
            await db.commit()

    async def set(self, task_id: str, record: dict[str, Any]) -> None:
        """Upsert the JSON-encoded record for a task."""
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT INTO tasks (task_id, record) VALUES (?, ?)
                ON CONFLICT(task_id) DO UPDATE SET record = excluded.record
                """,
                (task_id, json.dumps(record)),
            )
            await db.commit()

    async def get(self, task_id: str) -> dict[str, Any] | None:
        """Return the decoded record for a task, or None if unknown."""
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "SELECT record FROM tasks WHERE task_id = ?", (task_id,)
            )
            row = await cursor.fetchone()
        if row is None:
            return None
        return dict(json.loads(row[0]))


def create_task_store(settings: Settings) -> TaskStore:
    """Return the task store selected by settings.task_store."""
    if settings.task_store == "sqlite":
        return SQLiteTaskStore(settings.task_db_path)
    return InMemoryTaskStore()
