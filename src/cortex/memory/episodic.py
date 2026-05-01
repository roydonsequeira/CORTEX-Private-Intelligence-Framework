"""Episodic memory — chronological session history persisted in SQLite."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import aiosqlite

from cortex.memory.base import BaseMemory, MemoryEntry, MemoryQuery
from cortex.models.provider import Message
from cortex.observability.tracing import get_tracer

_tracer = get_tracer(__name__)


class EpisodicMemory(BaseMemory):
    """SQLite-backed conversation history, scoped by session_id."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)

    async def initialize(self) -> None:
        """Create the episodes table and indexes if they do not exist."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS episodes (
                  id TEXT PRIMARY KEY,
                  session_id TEXT NOT NULL,
                  role TEXT NOT NULL,
                  content TEXT NOT NULL,
                  metadata JSON,
                  created_at TEXT NOT NULL
                )
                """
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_session ON episodes(session_id)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_created_at ON episodes(created_at)"
            )
            await db.commit()

    async def store(self, entry: MemoryEntry) -> str:
        """Insert a memory entry into the episodes table."""
        metadata = dict(entry.metadata)
        session_id = str(metadata.get("session_id", "default"))
        role = str(metadata.get("role", "assistant"))
        with _tracer.start_as_current_span("memory.episodic.store") as span:
            span.set_attribute("session_id", session_id)
            span.set_attribute("db.system", "sqlite")
            async with aiosqlite.connect(self._db_path) as db:
                await db.execute(
                    """
                    INSERT INTO episodes (id, session_id, role, content, metadata, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        entry.id,
                        session_id,
                        role,
                        entry.content,
                        json.dumps(metadata),
                        entry.timestamp.isoformat(),
                    ),
                )
                await db.commit()
        return entry.id

    async def retrieve(self, query: MemoryQuery) -> list[MemoryEntry]:
        """Retrieve recent episodes, optionally filtered by session_id."""
        await self.initialize()
        sql = """
            SELECT id, session_id, role, content, metadata, created_at
            FROM episodes
        """
        params: list[object] = []
        clauses: list[str] = []
        if query.session_id:
            clauses.append("session_id = ?")
            params.append(query.session_id)
        if query.text:
            clauses.append("content LIKE ?")
            params.append(f"%{query.text}%")
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(query.top_k)

        with _tracer.start_as_current_span("memory.episodic.retrieve") as span:
            span.set_attribute("db.system", "sqlite")
            span.set_attribute("memory.top_k", query.top_k)
            if query.session_id:
                span.set_attribute("session_id", query.session_id)
            async with aiosqlite.connect(self._db_path) as db:
                rows = await (await db.execute(sql, params)).fetchall()

        entries: list[MemoryEntry] = []
        for row in rows:
            metadata = json.loads(row[4] or "{}")
            metadata.update({"session_id": row[1], "role": row[2]})
            entries.append(
                MemoryEntry(
                    id=row[0],
                    content=row[3],
                    metadata=metadata,
                    timestamp=datetime.fromisoformat(row[5]),
                    memory_type="episodic",
                )
            )
        return entries

    async def get_session_history(self, session_id: str) -> list[Message]:
        """Return ordered conversation history for a session."""
        await self.initialize()
        async with aiosqlite.connect(self._db_path) as db:
            rows = await (
                await db.execute(
                    """
                    SELECT role, content
                    FROM episodes
                    WHERE session_id = ?
                    ORDER BY created_at ASC
                    """,
                    (session_id,),
                )
            ).fetchall()
        return [Message(role=row[0], content=row[1]) for row in rows]

    async def list_session_ids(self) -> list[str]:
        """Return all known session IDs ordered by most recent activity."""
        await self.initialize()
        async with aiosqlite.connect(self._db_path) as db:
            rows = await (
                await db.execute(
                    """
                    SELECT session_id, MAX(created_at) AS last_seen
                    FROM episodes
                    GROUP BY session_id
                    ORDER BY last_seen DESC
                    """
                )
            ).fetchall()
        return [str(row[0]) for row in rows]

    async def delete_session(self, session_id: str) -> None:
        """Delete all episodic memory for a session."""
        await self.initialize()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("DELETE FROM episodes WHERE session_id = ?", (session_id,))
            await db.commit()

    async def prune_old_sessions(self, days: int = 30) -> None:
        """Delete episodes older than the provided age in days."""
        cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("DELETE FROM episodes WHERE created_at < ?", (cutoff,))
            await db.commit()

    async def delete(self, entry_id: str) -> None:
        """Delete an episode by id."""
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("DELETE FROM episodes WHERE id = ?", (entry_id,))
            await db.commit()


def make_episode(session_id: str, role: str, content: str) -> MemoryEntry:
    """Build a MemoryEntry suitable for EpisodicMemory.store()."""
    return MemoryEntry(
        id=uuid4().hex,
        content=content,
        metadata={"session_id": session_id, "role": role},
        timestamp=datetime.now(UTC),
        memory_type="episodic",
    )
