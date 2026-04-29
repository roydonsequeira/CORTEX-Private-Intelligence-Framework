"""Async task queue routes."""

import uuid
from typing import Any, Literal, cast

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

router = APIRouter(prefix="/tasks", tags=["tasks"])


class TaskRequest(BaseModel):
    """Fire-and-forget agent task request."""

    task: str = Field(min_length=1, max_length=8192)
    priority: Literal["low", "normal", "high"] = "normal"
    callback_url: str | None = None


@router.post("")
async def create_task(payload: TaskRequest, request: Request) -> dict[str, str]:
    """Queue an async agent task."""
    task_id = uuid.uuid4().hex
    request.app.state.task_results[task_id] = {"task_id": task_id, "status": "queued"}
    await request.app.state.task_queue.put(
        {
            "task_id": task_id,
            "task": payload.task,
            "priority": payload.priority,
            "callback_url": payload.callback_url,
        }
    )
    return {"task_id": task_id, "status": "queued"}


@router.get("/{task_id}")
async def get_task(task_id: str, request: Request) -> dict[str, Any]:
    """Return a task result or status."""
    result = request.app.state.task_results.get(task_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Task not found.")
    return cast(dict[str, Any], result)
