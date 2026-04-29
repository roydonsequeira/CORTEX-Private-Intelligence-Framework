"""Tool introspection and direct execution routes."""

from typing import Any, cast

from fastapi import APIRouter, Request

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
    """Directly execute a registered tool for debugging."""
    result = await request.app.state.tool_registry.execute(name, **payload)
    return cast(dict[str, Any], result.model_dump(mode="json"))
