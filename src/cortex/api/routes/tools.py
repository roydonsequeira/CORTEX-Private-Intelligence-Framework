"""Tool introspection and direct execution routes."""

from typing import Any, cast

from fastapi import APIRouter, HTTPException, Request

router = APIRouter(prefix="/tools", tags=["tools"])


@router.get("")
async def list_tools(request: Request) -> list[dict[str, Any]]:
    """Return all registered tool schemas."""
    return [
        cast(dict[str, Any], schema.model_dump(mode="json"))
        for schema in request.app.state.tool_registry.list_tools()
    ]


@router.post("/{name}/execute")
async def execute_tool(name: str, payload: dict[str, Any], request: Request) -> dict[str, Any]:
    """Directly execute a registered tool for debugging (off by default).

    It bypasses the agent and its safety prompt, so it is only available when
    ``debug_tool_endpoint: true`` is set explicitly.
    """
    settings = getattr(request.app.state, "settings", None)
    if not getattr(settings, "debug_tool_endpoint", False):
        raise HTTPException(
            status_code=404,
            detail="Direct tool execution is disabled. Set debug_tool_endpoint: true to enable it.",
        )
    result = await request.app.state.tool_registry.execute(name, **payload)
    return cast(dict[str, Any], result.model_dump(mode="json"))
