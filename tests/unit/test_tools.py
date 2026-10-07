"""Unit tests for tool registry and built-in tools."""

import os
import sys
import textwrap
from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest

from cortex.tools.base import BaseTool, ToolResult, ToolSchema
from cortex.tools.builtin.code_exec import CodeExecutionTool
from cortex.tools.builtin.filesystem import FileSystemTool
from cortex.tools.builtin.web_fetch import WebFetchTool
from cortex.tools.registry import ToolRegistry


def _web_tool(responses: dict[str, httpx.Response]) -> tuple[WebFetchTool, list[str]]:
    """A WebFetchTool whose HTTP client serves ``responses`` by URL path (404 otherwise)."""
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(request.url.path)
        return responses.get(request.url.path, httpx.Response(404))

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return WebFetchTool(client=client), requested


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
    html = "<html><body><h1>Hello</h1><p>World</p></body></html>"
    tool, _ = _web_tool(
        {"/page": httpx.Response(200, text=html, headers={"content-type": "text/html"})}
    )
    result = await tool.execute(url="https://example.com/page", format="markdown")
    await tool.aclose()
    assert result.success is True
    assert "# Hello" in result.output
    assert "World" in result.output


@pytest.mark.asyncio
async def test_web_fetch_includes_the_page_title() -> None:
    """A title that exists only in <head> is kept: example.com's page has no heading."""
    html = (
        "<html><head><title>Example &amp;\n Domain</title></head>"
        "<body><p>This domain is for use in documentation examples.</p></body></html>"
    )
    tool, _ = _web_tool(
        {"/": httpx.Response(200, text=html, headers={"content-type": "text/html"})}
    )
    result = await tool.execute(url="https://example.com/")
    await tool.aclose()
    assert result.output.startswith("Title: Example & Domain\n\n")
    assert "documentation examples" in result.output


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
async def test_filesystem_never_reads_or_lists_hidden_paths(tmp_path: Path) -> None:
    """.env and .git are refused for reading and listing too, not only writing."""
    (tmp_path / ".env").write_text("TOKEN=secret", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    tool = FileSystemTool(allowed_root=tmp_path)
    for action, raw in (("read_file", ".env"), ("list_directory", ".git"), ("file_exists", ".env")):
        result = await tool.execute(action=action, path=raw)
        assert result.success is False, raw
        assert "hidden" in (result.error or ""), raw
        assert "secret" not in result.output


@pytest.mark.asyncio
async def test_filesystem_refuses_writes_into_cortex_code(tmp_path: Path) -> None:
    """With the workspace widened over CORTEX itself, its code and plugins stay read-only."""
    plugins = tmp_path / "plugins"
    plugins.mkdir()
    tool = FileSystemTool(allowed_root=tmp_path, protected_dirs=[plugins])
    result = await tool.execute(action="write_file", path="plugins/helper.py", content="x = 1")
    assert result.success is False
    assert "CORTEX's own code or plugins folder" in (result.error or "")
    assert not (plugins / "helper.py").exists()
    ok = await tool.execute(action="write_file", path="notes.py", content="x = 1")
    assert ok.success is True


def test_filesystem_from_settings_protects_the_package_and_plugins(tmp_path: Path) -> None:
    """from_settings protects the installed cortex package and the plugins folder."""
    from cortex.config.settings import Settings

    settings = Settings(allowed_root=tmp_path, plugins_dir=tmp_path / "plugins")
    tool = FileSystemTool.from_settings(settings)
    protected = {str(path) for path in tool._protected_dirs}
    assert str((tmp_path / "plugins").resolve()) in protected
    assert any(path.name == "cortex" and (path / "cli.py").exists() for path in tool._protected_dirs)


@pytest.mark.asyncio
async def test_doc_search_never_indexes_hidden_or_outside_files(tmp_path: Path) -> None:
    """Indexing hands content to the model, so it follows the file tool's rules."""
    from cortex.tools.builtin.doc_search import DocumentSearchTool

    (tmp_path / ".env").write_text("TOKEN=secret", encoding="utf-8")
    tool = DocumentSearchTool(_FakeSemanticMemory(), allowed_root=tmp_path)  # type: ignore[arg-type]
    hidden = await tool.execute(action="index_document", path=".env")
    assert hidden.success is False
    assert "hidden" in (hidden.error or "")
    outside = await tool.execute(action="index_document", path=str(tmp_path.parent / "x.md"))
    assert outside.success is False


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
    tool, _ = _web_tool({})
    result = await tool.execute(url="example.com/missing")
    await tool.aclose()
    assert result.success is False
    assert "HTTP 404" in (result.error or "")


@pytest.mark.asyncio
async def test_web_fetch_stops_reading_at_the_download_cap() -> None:
    """A huge response is cut at the cap while downloading, not after reading it all."""
    from cortex.tools.builtin import web_fetch

    produced = 0

    async def endless() -> AsyncIterator[bytes]:
        nonlocal produced
        for _ in range(10_000):  # up to 640 MB if it were read to the end
            produced += 65_536
            yield b"a" * 65_536

    tool, _ = _web_tool(
        {"/big": httpx.Response(200, content=endless(), headers={"content-type": "text/plain"})}
    )
    result = await tool.execute(url="https://example.com/big")
    await tool.aclose()
    assert result.success is True
    assert result.output.endswith("[content truncated]")
    assert produced <= web_fetch._MAX_DOWNLOAD_BYTES + 65_536


@pytest.mark.asyncio
async def test_web_fetch_refuses_binary_files_without_downloading() -> None:
    """A PDF or image link gets a clear message instead of decoded garbage."""
    tool, _ = _web_tool(
        {"/paper.pdf": httpx.Response(200, content=b"%PDF-1.7", headers={"content-type": "application/pdf"})}
    )
    result = await tool.execute(url="https://example.com/paper.pdf")
    await tool.aclose()
    assert result.success is False
    assert "application/pdf file, not a web page" in (result.error or "")


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


class _FakeSemanticMemory:
    """Stores entries in a list; retrieve returns them all (no embeddings)."""

    def __init__(self) -> None:
        self.entries: dict[str, object] = {}

    async def store(self, entry: object) -> str:
        self.entries[entry.id] = entry  # type: ignore[attr-defined]
        return entry.id  # type: ignore[attr-defined,no-any-return]

    async def retrieve(self, query: object) -> list[object]:
        return list(self.entries.values())


@pytest.mark.asyncio
async def test_doc_search_with_path_indexes_that_document_first(tmp_path: Path) -> None:
    """'Search README.md for X' works even when the model skipped index_document."""
    from cortex.tools.builtin.doc_search import DocumentSearchTool

    (tmp_path / "README.md").write_text("Memory tiers: working, episodic, semantic.")
    (tmp_path / "other.md").write_text("Unrelated notes about lunch.")
    memory = _FakeSemanticMemory()
    tool = DocumentSearchTool(memory, allowed_root=tmp_path)  # type: ignore[arg-type]
    await tool.execute(action="index_document", path="other.md")

    result = await tool.execute(action="search", query="memory tiers", path="README.md")

    assert result.success is True
    assert "episodic" in result.output
    assert "lunch" not in result.output  # results are limited to the named document


@pytest.mark.asyncio
async def test_doc_search_index_with_query_also_searches(tmp_path: Path) -> None:
    """index_document with a query returns the top matches in the same call."""
    from cortex.tools.builtin.doc_search import DocumentSearchTool

    (tmp_path / "README.md").write_text("The sandbox runs code in a separate process.")
    tool = DocumentSearchTool(_FakeSemanticMemory(), allowed_root=tmp_path)  # type: ignore[arg-type]

    result = await tool.execute(action="index_document", path="README.md", query="sandbox")

    assert result.success is True
    assert result.output.startswith("Indexed 1 chunks.")
    assert "separate process" in result.output


@pytest.mark.asyncio
async def test_doc_search_accepts_out_of_range_numbers_from_the_model(tmp_path: Path) -> None:
    """chunk_size=100 (seen live from qwen2.5:7b) is clamped, not rejected by the schema."""
    from cortex.tools.builtin.doc_search import DocumentSearchTool

    (tmp_path / "README.md").write_text("CORTEX uses qwen2.5:7b and nomic-embed-text.")
    registry = ToolRegistry()
    registry.register(DocumentSearchTool(_FakeSemanticMemory(), allowed_root=tmp_path))  # type: ignore[arg-type]

    result = await registry.execute(
        "doc_search",
        query="models used by CORTEX",
        path="README.md",
        chunk_size=100,
        overlap=50,
        top_k=50,
    )

    assert result.success is True, result.error
    assert "nomic-embed-text" in result.output


def test_doc_search_clamps_chunk_arguments() -> None:
    """Out-of-range sizes are clamped; overlap never exceeds half a chunk."""
    from cortex.tools.builtin.doc_search import _chunk_args, _int_arg

    assert _chunk_args({}) == (512, 64)
    assert _chunk_args({"chunk_size": 100, "overlap": 50}) == (128, 50)
    assert _chunk_args({"chunk_size": 9000, "overlap": 5000}) == (4096, 1024)
    assert _chunk_args({"chunk_size": 200, "overlap": 190}) == (200, 100)
    assert _int_arg(0, 5, 1, 20) == 1
    assert _int_arg("7", 5, 1, 20) == 7


@pytest.mark.asyncio
async def test_python_exec_repairs_double_escaped_newlines() -> None:
    """Code sent with literal backslash-n line breaks still runs."""
    code = "def f(a, b):" + "\n" + "    return a + b" + "\n" + "print(f(1, 2))"
    result = await CodeExecutionTool().execute(code=code)
    assert result.success is True
    assert result.output == "3"


def test_escape_repair_leaves_valid_code_alone() -> None:
    """A real escape inside a string literal is not touched."""
    from cortex.tools.builtin.code_exec import _repair_escaped_newlines

    code = 'print("a' + "\n" + 'b")'
    assert _repair_escaped_newlines(code) == code


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8000/health",
        "http://localhost:11434/api/tags",
        "http://169.254.169.254/latest/meta-data/",
        "http://192.168.1.1/",
        "http://[::1]/",
    ],
)
async def test_web_fetch_refuses_local_and_private_hosts(url: str) -> None:
    """web_fetch cannot be turned against this machine or the local network (SSRF)."""
    tool, requested = _web_tool({})
    result = await tool.execute(url=url)
    await tool.aclose()
    assert result.success is False
    assert "local or private network" in (result.error or "")
    assert requested == []


@pytest.mark.asyncio
async def test_web_fetch_refuses_redirect_to_a_private_host() -> None:
    """A public page that redirects to localhost is not followed."""
    redirect = httpx.Response(302, headers={"location": "http://127.0.0.1:8000/"})
    tool, requested = _web_tool({"/page": redirect})
    result = await tool.execute(url="https://93.184.215.14/page")
    await tool.aclose()
    assert result.success is False
    assert "redirect target" in (result.error or "")
    assert requested == ["/robots.txt", "/page"]


def test_filesystem_infers_a_missing_action() -> None:
    """Models often omit `action`; content means write, a folder path means list."""
    from cortex.tools.builtin.filesystem import infer_action

    assert infer_action({"path": "notes.txt", "content": "x"}) == "write_file"
    assert infer_action({"path": "."}) == "list_directory"
    assert infer_action({"path": "README.md"}) == "read_file"
    assert infer_action({"action": "file_exists", "path": "a.txt"}) == "file_exists"


@pytest.mark.asyncio
async def test_filesystem_without_action_reads_the_file(tmp_path: Path) -> None:
    """{"path": ...} alone passes validation and reads the file."""
    (tmp_path / "a.txt").write_text("hello")
    registry = ToolRegistry()
    registry.register(FileSystemTool(allowed_root=tmp_path))
    result = await registry.execute("filesystem", path="a.txt")
    assert result.success is True
    assert result.output == "hello"


@pytest.mark.asyncio
async def test_python_exec_repairs_escaped_quotes() -> None:
    """name = \\"Roydon\\" (escaped quotes, escaped newlines) still runs."""
    code = 'name = \\"Roydon\\"' + "\n" + "print(name[::-1])"
    result = await CodeExecutionTool().execute(code=code)
    assert result.success is True
    assert result.output == "nodyoR"


def test_executor_prompt_describes_python_file_access_per_backend() -> None:
    """The model is told open() reads workspace files only when the sandbox allows it."""
    from cortex.agent.executor import Executor
    from cortex.agent.kernel import AgentState

    state = AgentState(session_id="s", user_input="count words in README.md")
    reads = Executor(python_reads_workspace=True).build_context(state)[0].content
    no_files = Executor(python_reads_workspace=False).build_context(state)[0].content

    assert "open('name.txt')" in reads
    assert "Code cannot open files" in no_files
    assert "{python_files}" not in reads + no_files
    assert "never state the result a failed tool call was meant to compute" in reads


@pytest.mark.asyncio
async def test_filesystem_refuses_a_hard_link_to_another_file(tmp_path: Path) -> None:
    """A file with a second name elsewhere is neither read nor replaced (#57)."""
    root = tmp_path / "ws"
    root.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("outside", encoding="utf-8")
    try:
        os.link(outside, root / "linked.txt")
    except OSError:
        pytest.skip("hard links unsupported here")
    tool = FileSystemTool(allowed_root=root)

    read = await tool.execute(action="read_file", path="linked.txt")
    write = await tool.execute(action="write_file", path="linked.txt", content="x", overwrite=True)

    assert read.success is False and "hard link" in (read.error or "")
    assert write.success is False
    assert outside.read_text(encoding="utf-8") == "outside"


@pytest.mark.asyncio
async def test_filesystem_overwrite_replaces_the_whole_file(tmp_path: Path) -> None:
    """Content is written exactly; a shorter overwrite leaves no tail of the old file."""
    tool = FileSystemTool(allowed_root=tmp_path)
    await tool.execute(action="write_file", path="notes/a.txt", content="a long first line\n")
    result = await tool.execute(action="write_file", path="notes/a.txt", content="ok", overwrite=True)
    assert result.success is True
    assert (tmp_path / "notes" / "a.txt").read_bytes() == b"ok"


def _swap_in_junction(root: Path, outside: Path) -> None:
    (root / "sub").rename(root / "sub-old")
    if sys.platform == "win32":  # a junction needs no special rights, unlike a symlink
        import _winapi

        _winapi.CreateJunction(str(outside), str(root / "sub"))


@pytest.fixture
def junction_layout(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "ws"
    (root / "sub").mkdir(parents=True)
    (root / "sub" / "notes.txt").write_text("inside", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "notes.txt").write_text("OUTSIDE", encoding="utf-8")
    return root, outside


@pytest.mark.skipif(sys.platform != "win32", reason="Windows junctions")
@pytest.mark.asyncio
async def test_windows_read_through_a_junction_swapped_in_after_the_check(
    junction_layout: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    from cortex.tools.builtin import filesystem as fs_module
    from cortex.tools.workspace import ensure_not_hidden

    root, outside = junction_layout

    def racing(root_arg: Path, path: Path) -> None:
        ensure_not_hidden(root_arg, path)
        _swap_in_junction(root, outside)

    monkeypatch.setattr(fs_module, "ensure_not_hidden", racing)
    tool = FileSystemTool(allowed_root=root)
    result = await tool.execute(action="read_file", path="sub/notes.txt")

    assert (root / "sub").is_junction(), "the swap did not happen"
    assert result.success is False and "OUTSIDE" not in result.output


@pytest.mark.skipif(sys.platform != "win32", reason="Windows junctions")
@pytest.mark.asyncio
async def test_windows_new_file_through_a_junction_is_removed_unwritten(
    junction_layout: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The open happens before the handle can be checked, so a file it created outside is deleted."""
    from cortex.tools import workspace as workspace_module

    root, outside = junction_layout
    make_folders = workspace_module._make_folders

    def racing(root_arg: Path, folder: Path, display: str) -> None:
        make_folders(root_arg, folder, display)
        _swap_in_junction(root, outside)

    monkeypatch.setattr(workspace_module, "_make_folders", racing)
    result = await FileSystemTool(allowed_root=root).execute(
        action="write_file", path="sub/new.txt", content="agent"
    )

    assert (root / "sub").is_junction(), "the swap did not happen"
    assert result.success is False
    assert not (outside / "new.txt").exists()
