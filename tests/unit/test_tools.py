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
async def test_code_execution_tool_blocks_module_traversal_escape() -> None:
    """Sandbox blocks the json -> codecs -> sys -> os module-traversal escape."""
    result = await CodeExecutionTool().execute(
        code="os = json.codecs.sys.modules.get('os')\nprint(os.getcwd())"
    )
    assert result.success is False
    assert "getcwd" not in result.output


@pytest.mark.asyncio
async def test_code_execution_tool_allows_subscripting() -> None:
    """Sandbox permits ordinary subscripting (guarded _getitem_)."""
    result = await CodeExecutionTool().execute(code="print([10, 20, 30][1])")
    assert result.success is True
    assert "20" in result.output


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


@pytest.mark.asyncio
async def test_filesystem_reads_utf8_regardless_of_locale(tmp_path: Path) -> None:
    """Non-ASCII UTF-8 files read correctly (Windows' default codec is cp1252)."""
    (tmp_path / "notes.md").write_bytes("café — naïve ✓".encode())
    tool = FileSystemTool(allowed_root=tmp_path)
    result = await tool.execute(action="read_file", path="notes.md")
    assert result.success is True
    assert result.output == "café — naïve ✓"


@pytest.mark.asyncio
async def test_filesystem_normalises_root_paths(tmp_path: Path) -> None:
    """'/', '.', and a leading slash address the workspace root, not the disk root."""
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    tool = FileSystemTool(allowed_root=tmp_path)
    for raw in ("/", ".", ""):
        listing = await tool.execute(action="list_directory", path=raw)
        assert listing.success is True
        assert "a.txt" in listing.output
    read = await tool.execute(action="read_file", path="/a.txt")
    assert read.output == "x"


@pytest.mark.asyncio
async def test_filesystem_denies_system_paths(tmp_path: Path) -> None:
    """Absolute system paths are refused with a clear 'outside the workspace' message."""
    tool = FileSystemTool(allowed_root=tmp_path)
    for raw in ("/etc/passwd", "C:\\Windows\\win.ini", "//server/share/x.txt"):
        result = await tool.execute(action="read_file", path=raw)
        assert result.success is False, raw
        assert "outside the CORTEX workspace" in (result.error or ""), raw


@pytest.mark.asyncio
async def test_filesystem_never_overwrites_without_explicit_flag(tmp_path: Path) -> None:
    """Existing files survive a write unless overwrite=true is passed."""
    (tmp_path / "README.md").write_text("original", encoding="utf-8")
    tool = FileSystemTool(allowed_root=tmp_path)

    refused = await tool.execute(action="write_file", path="README.md", content="")
    assert refused.success is False
    assert "already exists" in (refused.error or "")
    assert (tmp_path / "README.md").read_text(encoding="utf-8") == "original"

    replaced = await tool.execute(
        action="write_file", path="README.md", content="new", overwrite=True
    )
    assert replaced.success is True
    assert (tmp_path / "README.md").read_text(encoding="utf-8") == "new"


@pytest.mark.asyncio
async def test_filesystem_refuses_writes_to_hidden_paths(tmp_path: Path) -> None:
    """.git, .venv, .env and other dot-paths are never written."""
    tool = FileSystemTool(allowed_root=tmp_path)
    for raw in (".env.txt", ".git/config.txt", "notes/.secret.md"):
        result = await tool.execute(action="write_file", path=raw, content="x")
        assert result.success is False, raw
        assert "hidden" in (result.error or ""), raw


@pytest.mark.asyncio
async def test_filesystem_missing_file_is_a_clean_error(tmp_path: Path) -> None:
    """A missing file yields a failed result, not an exception."""
    tool = FileSystemTool(allowed_root=tmp_path)
    result = await tool.execute(action="read_file", path="nope.txt")
    assert result.success is False
    assert "not found" in (result.error or "")


@pytest.mark.asyncio
async def test_web_fetch_reports_http_errors_accurately() -> None:
    """A 404 is reported as an HTTP error, not as 'offline mode'."""
    tool = WebFetchTool()
    request = httpx.Request("GET", "https://example.com/missing")
    robots = httpx.Response(404, text="", request=request)
    missing = httpx.Response(404, text="nope", request=request)
    with patch.object(tool._client, "get", new=AsyncMock(side_effect=[robots, missing])):
        result = await tool.execute(url="example.com/missing")
    await tool.aclose()
    assert result.success is False
    assert "HTTP 404" in (result.error or "")


@pytest.mark.asyncio
async def test_web_fetch_rejects_non_web_urls() -> None:
    """Non-http(s) schemes are refused before any request is made."""
    tool = WebFetchTool()
    result = await tool.execute(url="file:///etc/passwd")
    await tool.aclose()
    assert result.success is False


def test_doc_chunker_splits_crlf_documents_on_paragraphs() -> None:
    """CRLF files (Windows) are chunked by paragraph, not cut into blind windows."""
    from cortex.tools.builtin.doc_search import _chunk_document

    first = "Memory tiers: working, episodic, semantic, procedural."
    second = "The sandbox runs code in a separate process."
    crlf = f"{first}\r\n\r\n{second}\r\n"

    chunks = _chunk_document(crlf, chunk_size=60, overlap=10)

    assert chunks == [first, second]


@pytest.mark.asyncio
async def test_calculator_handles_functions_and_caret() -> None:
    """The calculator supports sqrt, constants, and ^ as exponent."""
    from cortex.tools.plugins.calculator import CalculatorTool

    tool = CalculatorTool()
    assert (await tool.execute(expression="sqrt(144) + 2^3")).output == "20"
    assert (await tool.execute(expression="(17 * 23) + 5")).output == "396"


@pytest.mark.asyncio
async def test_calculator_refuses_runaway_exponents() -> None:
    """9**9**9 would freeze the API process; it is refused instead."""
    from cortex.tools.plugins.calculator import CalculatorTool

    tool = CalculatorTool()
    for expression in ("9**9**9", "(10**9999)**9999", "factorial(10**9)"):
        result = await tool.execute(expression=expression)
        assert result.success is False, expression


@pytest.mark.asyncio
async def test_executor_does_not_retry_deterministic_failures() -> None:
    """A tool that returns a failure is not re-run (only raised errors are retried)."""
    from unittest.mock import MagicMock

    from cortex.agent.executor import Executor

    registry = MagicMock(spec=ToolRegistry)
    registry.execute = AsyncMock(
        return_value=ToolResult(
            tool_name="python_exec",
            success=False,
            output="",
            error="SyntaxError",
            execution_time_ms=0.0,
        )
    )
    result = await Executor()._execute_with_retry("python_exec", registry, code="x=")
    assert result.success is False
    assert registry.execute.await_count == 1
