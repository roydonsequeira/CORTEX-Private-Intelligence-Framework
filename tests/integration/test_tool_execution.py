"""Integration tests for end-to-end tool execution through ToolRegistry."""

from pathlib import Path

import pytest

from cortex.tools.builtin.code_exec import CodeExecutionTool
from cortex.tools.builtin.filesystem import FileSystemTool
from cortex.tools.registry import ToolRegistry


@pytest.mark.asyncio
async def test_registry_executes_python_exec_end_to_end() -> None:
    """ToolRegistry.execute dispatches python_exec end-to-end."""
    registry = ToolRegistry()
    registry.register(CodeExecutionTool())
    result = await registry.execute("python_exec", code="print(2 + 2)")
    assert result.success is True
    assert "4" in result.output


@pytest.mark.asyncio
async def test_registry_executes_filesystem_round_trip(tmp_path: Path) -> None:
    """ToolRegistry.execute dispatches filesystem write/read end-to-end."""
    registry = ToolRegistry()
    registry.register(FileSystemTool(allowed_root=tmp_path))
    write = await registry.execute(
        "filesystem",
        action="write_file",
        path="test.txt",
        content="hello",
    )
    read = await registry.execute("filesystem", action="read_file", path="test.txt")
    assert write.success is True
    assert read.output == "hello"
