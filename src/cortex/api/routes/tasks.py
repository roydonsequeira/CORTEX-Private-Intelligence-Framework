"""Async task queue routes."""

import itertools
import uuid
from typing import Any, Literal, cast

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

router = APIRouter(prefix="/tasks", tags=["tasks"])

# Lower rank runs first; the sequence number keeps FIFO order within a priority
# and stops the queue from ever comparing two task dicts.
_PRIORITY_RANK = {"high": 0, "normal": 1, "low": 2}
_sequence = itertools.count()


class TaskRequest(BaseModel):
    """Fire-and-forget agent task request."""

    task: str = Field(min_length=1, max_length=8192, pattern=r"\S")
    priority: Literal["low", "normal", "high"] = "normal"
    orchestration: Literal["single", "supervisor", "lats"] = "single"


@router.post("")
async def create_task(payload: TaskRequest, request: Request) -> dict[str, str]:
    """Queue an async agent task; poll ``GET /tasks/{task_id}`` for the result."""
    task_id = uuid.uuid4().hex
    await request.app.state.task_store.set(task_id, {"task_id": task_id, "status": "queued"})
    await request.app.state.task_queue.put(
        (
            _PRIORITY_RANK[payload.priority],
            next(_sequence),
            {
                "task_id": task_id,
                "task": payload.task,
                "priority": payload.priority,
                "orchestration": payload.orchestration,
            },
        )
    )
    return {"task_id": task_id, "status": "queued"}


@router.get("/{task_id}")
async def get_task(task_id: str, request: Request) -> dict[str, Any]:
    """Return a task result or status."""
    result = await request.app.state.task_store.get(task_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Task not found.")
    return cast(dict[str, Any], result)
