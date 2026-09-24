"""Unit tests for the pluggable task store backends."""

from pathlib import Path

import pytest

from cortex.api.task_store import (
    InMemoryTaskStore,
    SQLiteTaskStore,
    create_task_store,
)
from cortex.config.settings import Settings


@pytest.mark.asyncio
async def test_in_memory_store_roundtrip() -> None:
    """The in-memory store returns what was stored and None for unknown ids."""
    store = InMemoryTaskStore()
    await store.initialize()
    await store.set("t1", {"task_id": "t1", "status": "done", "result": {"x": 1}})

    assert await store.get("t1") == {"task_id": "t1", "status": "done", "result": {"x": 1}}
    assert await store.get("missing") is None


@pytest.mark.asyncio
async def test_sqlite_store_survives_restart(tmp_path: Path) -> None:
    """A record written by one SQLite store instance is readable by a fresh one."""
    db_path = tmp_path / "tasks.db"
    store = SQLiteTaskStore(db_path)
    await store.initialize()
    await store.set("t1", {"task_id": "t1", "status": "done", "result": {"answer": 42}})

    # Simulate a process restart with a brand-new store over the same file.
    restarted = SQLiteTaskStore(db_path)
    await restarted.initialize()

    assert await restarted.get("t1") == {
        "task_id": "t1",
        "status": "done",
        "result": {"answer": 42},
    }


@pytest.mark.asyncio
async def test_sqlite_store_upserts_status(tmp_path: Path) -> None:
    """Re-setting a task id replaces its record (queued -> running -> done)."""
    store = SQLiteTaskStore(tmp_path / "tasks.db")
    await store.initialize()
    await store.set("t1", {"task_id": "t1", "status": "queued"})
    await store.set("t1", {"task_id": "t1", "status": "done", "result": {"ok": True}})

    record = await store.get("t1")
    assert record is not None
    assert record["status"] == "done"


def test_factory_selects_backend(tmp_path: Path) -> None:
    """create_task_store returns the backend named by settings.task_store."""
    memory_settings = Settings(task_store="memory")
    sqlite_settings = Settings(task_store="sqlite", task_db_path=tmp_path / "tasks.db")

    assert isinstance(create_task_store(memory_settings), InMemoryTaskStore)
    assert isinstance(create_task_store(sqlite_settings), SQLiteTaskStore)


@pytest.mark.asyncio
async def test_in_memory_store_evicts_oldest_records() -> None:
    """The in-memory store keeps only the most recent records."""
    store = InMemoryTaskStore(max_records=2)
    for task_id in ("a", "b", "c"):
        await store.set(task_id, {"task_id": task_id})

    assert await store.get("a") is None
    assert await store.get("c") == {"task_id": "c"}
