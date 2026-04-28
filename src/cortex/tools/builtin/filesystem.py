"""Filesystem tool — safe read/write/list/exists under an allowed root."""

import json
import time
from pathlib import Path

import anyio

from cortex.config.settings import Settings
from cortex.tools.base import BaseTool, ToolResult, ToolSchema

_MAX_READ_BYTES = 1_048_576
_MAX_WRITE_BYTES = 524_288


class FileSystemTool(BaseTool):
    """Read, write, and list files under a constrained workspace root."""

    schema = ToolSchema(
        name="filesystem",
        description=(
            "Read, write, and list files on the local filesystem. "
            "All paths are relative to the workspace root."
        ),
        parameters={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["read_file", "write_file", "list_directory", "file_exists"],
                },
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["action", "path"],
            "additionalProperties": False,
        },
    )

    def __init__(
        self,
        allowed_root: str | Path = ".",
        allowed_extensions: list[str] | None = None,
    ) -> None:
        self._allowed_root = Path(allowed_root).resolve()
        self._allowed_extensions = set(
            allowed_extensions or [".txt", ".md", ".json", ".csv", ".py"]
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> "FileSystemTool":
        """Create a FileSystemTool from CORTEX settings."""
        return cls(settings.allowed_root, settings.allowed_write_extensions)

    async def execute(self, **kwargs: object) -> ToolResult:
        """Execute a filesystem action with path validation at point of use."""
        start = time.monotonic()
        action = str(kwargs["action"])
        try:
            path = self._resolve_path(str(kwargs["path"]))
            if action == "read_file":
                output = await self._read_file(path)
            elif action == "write_file":
                output = await self._write_file(path, str(kwargs.get("content", "")))
            elif action == "list_directory":
                output = await self._list_directory(path)
            elif action == "file_exists":
                output = json.dumps(path.exists())
            else:
                output = f"Unsupported action: {action}"
                return _result(False, output, time.monotonic() - start)
            return _result(True, output, time.monotonic() - start)
        except OSError as exc:
            return _result(False, "", time.monotonic() - start, str(exc))

    def _resolve_path(self, raw_path: str) -> Path:
        """Resolve and validate a relative path under the allowed root."""
        candidate = Path(raw_path)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise OSError("Path must be relative and must not contain '..'.")
        resolved = (self._allowed_root / candidate).resolve()
        if not resolved.is_relative_to(self._allowed_root):
            raise OSError("Path escapes the allowed root.")
        return resolved

    async def _read_file(self, path: Path) -> str:
        """Read a text file up to the maximum allowed size."""
        if not path.is_file():
            raise OSError("Path is not a file.")
        if path.stat().st_size > _MAX_READ_BYTES:
            raise OSError("File exceeds 1MB read limit.")
        return await anyio.Path(path).read_text()

    async def _write_file(self, path: Path, content: str) -> str:
        """Write content to an allowed extension under the allowed root."""
        if path.suffix not in self._allowed_extensions:
            raise OSError(f"Extension '{path.suffix}' is not allowed for writes.")
        if len(content.encode("utf-8")) > _MAX_WRITE_BYTES:
            raise OSError("Content exceeds 512KB write limit.")
        path.parent.mkdir(parents=True, exist_ok=True)
        await anyio.Path(path).write_text(content)
        return f"Wrote {len(content)} characters to {path.relative_to(self._allowed_root)}"

    async def _list_directory(self, path: Path) -> str:
        """List entries in a directory."""
        if not path.is_dir():
            raise OSError("Path is not a directory.")
        entries = sorted(p.name for p in path.iterdir())
        return json.dumps(entries)


def _result(
    success: bool,
    output: str,
    elapsed_seconds: float,
    error: str | None = None,
) -> ToolResult:
    """Build a ToolResult for filesystem operations."""
    return ToolResult(
        tool_name="filesystem",
        success=success,
        output=output,
        error=error,
        execution_time_ms=elapsed_seconds * 1000,
    )
