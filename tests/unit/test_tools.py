"""Unit tests for tool registry and built-in tools."""

import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from cortex.tools.base import BaseTool, ToolResult, ToolSchema
from cortex.tools.builtin.code_exec import CodeExecutionTool
from cortex.tools.builtin.filesystem import FileSystemTool
from cortex.tools.builtin.web_fetch import WebFetchTool
from cortex.tools.registry import ToolRegistry


@pytest.mark.asyncio
async def test_filesystem_tool_rejects_path_traversal(tmp_path: Path) -> None:
    """FileSystemTool rejects paths containing parent traversal."""
    tool = FileSystemTool(allowed_root=tmp_path)
    result = await tool.execute(action="read_file", path="../secret.txt")
    assert result.success is False
    assert "must not contain" in (result.error or "")


@pytest.mark.asyncio
async def test_filesystem_tool_read_write_round_trip(tmp_path: Path) -> None:
    """FileSystemTool writes and reads back an allowed file."""
    tool = FileSystemTool(allowed_root=tmp_path)
    write = await tool.execute(action="write_file", path="notes/test.txt", content="hello")
    read = await tool.execute(action="read_file", path="notes/test.txt")
    assert write.success is True
    assert read.output == "hello"


@pytest.mark.asyncio
async def test_code_execution_tool_basic_computation() -> None:
    """CodeExecutionTool executes simple computation in a sandbox."""
    result = await CodeExecutionTool().execute(code="print(2 + 2)")
    assert result.success is True
    assert "4" in result.output


@pytest.mark.asyncio
async def test_code_execution_tool_blocks_imports() -> None:
    """RestrictedPython blocks imports such as os."""
    result = await CodeExecutionTool().execute(code="import os\nprint(os.getcwd())")
    assert result.success is False


@pytest.mark.asyncio
async def test_code_execution_tool_timeout() -> None:
    """CodeExecutionTool enforces a timeout."""
    tool = CodeExecutionTool(timeout_seconds=0.01)
    result = await tool.execute(code="while True:\n    pass")
    assert result.success is False
    assert result.error == "Execution timed out."


@pytest.mark.asyncio
async def test_web_fetch_tool_converts_html_to_markdown() -> None:
    """WebFetchTool converts HTML to readable markdown."""
    tool = WebFetchTool()
    request = httpx.Request("GET", "https://example.com/page")
    robots = httpx.Response(404, text="", request=request)
    html = httpx.Response(
        200,
        text="<html><body><h1>Hello</h1><p>World</p></body></html>",
        request=request,
    )
    with patch.object(tool._client, "get", new=AsyncMock(side_effect=[robots, html])):
        result = await tool.execute(url="https://example.com/page", format="markdown")
    await tool.aclose()
    assert result.success is True
    assert "# Hello" in result.output
    assert "World" in result.output


@pytest.mark.asyncio
async def test_tool_registry_validates_schema() -> None:
    """ToolRegistry returns a ToolResult error for invalid kwargs."""

    class EchoTool(BaseTool):
        schema = ToolSchema(
            name="echo",
            description="Echo text.",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            },
        )

        async def execute(self, **kwargs: object) -> ToolResult:
            return ToolResult(
                tool_name=self.schema.name,
                success=True,
                output=str(kwargs["text"]),
                execution_time_ms=0.0,
            )

    registry = ToolRegistry()
    registry.register(EchoTool())
    result = await registry.execute("echo", wrong="value")
    assert result.success is False
    assert "Invalid arguments" in (result.error or "")


def test_tool_registry_auto_discover_picks_up_plugins(tmp_path: Path) -> None:
    """ToolRegistry.auto_discover registers valid plugin classes."""
    plugin = tmp_path / "echo_plugin.py"
    plugin.write_text(
        textwrap.dedent(
            """
            from typing import ClassVar
            from cortex.tools.base import BaseTool, ToolResult, ToolSchema

            class EchoPlugin(BaseTool):
                schema: ClassVar[ToolSchema] = ToolSchema(
                    name="echo_plugin",
                    description="Echo plugin.",
                    parameters={"type": "object", "properties": {}, "additionalProperties": False},
                )

                async def execute(self, **kwargs: object) -> ToolResult:
                    return ToolResult(tool_name="echo_plugin", success=True, output="ok", execution_time_ms=0.0)
            """
        )
    )

    registry = ToolRegistry()
    registry.auto_discover(tmp_path)
    assert registry.get("echo_plugin") is not None


def test_tool_registry_auto_discover_skips_bad_plugins(tmp_path: Path) -> None:
    """ToolRegistry.auto_discover warns and continues on bad plugins."""
    (tmp_path / "bad.py").write_text("raise RuntimeError('bad plugin')")
    registry = ToolRegistry()
    registry.auto_discover(tmp_path)
    assert registry.list_tools() == []
