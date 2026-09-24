"""Filesystem tool — safe read/write/list/exists under an allowed root."""

import json
import re
import time
from pathlib import Path

import anyio

from cortex.config.settings import Settings
from cortex.tools.base import BaseTool, ToolResult, ToolSchema

_MAX_READ_BYTES = 1_048_576
_MAX_WRITE_BYTES = 524_288
# "C:/..." or "c:\..." (after backslashes are normalised to "/").
_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:(/|$)")


def infer_action(kwargs: dict[str, object]) -> str:
    """The requested action, inferred when the model left it out.

    Seen: {"path": "notes.txt", "content": "x"} and {"path": "C:/.../hosts"} with
    no action, which failed validation and left the model asking the user which
    action to use. Content means a write; a directory-like path means a listing.
    """
    if kwargs.get("action"):
        return str(kwargs["action"])
    if "content" in kwargs:
        return "write_file"
    path = str(kwargs.get("path", "")).strip()
    if path in ("", ".", "./", "/") or path.endswith(("/", "\\")):
        return "list_directory"
    return "read_file"


class FileSystemTool(BaseTool):
    """Read, write, and list files under a constrained workspace root."""

    schema = ToolSchema(
        name="filesystem",
        description=(
            "Read, write, and list files in the CORTEX workspace. Paths are relative "
            "to the workspace root. Writing never replaces an existing file unless "
            "overwrite is true, which requires the user to have explicitly asked for it."
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
                "overwrite": {"type": "boolean"},
            },
            # Models often omit `action`; it is inferred (see infer_action).
            "required": ["path"],
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
        action = infer_action(kwargs)
        try:
            path = self._resolve_path(str(kwargs["path"]))
            if action == "read_file":
                output = await self._read_file(path)
            elif action == "write_file":
                output = await self._write_file(
                    path, str(kwargs.get("content", "")), overwrite=kwargs.get("overwrite") is True
                )
            elif action == "list_directory":
                output = await self._list_directory(path)
            elif action == "file_exists":
                output = json.dumps(path.exists())
            else:
                output = f"Unsupported action: {action}"
                return _result(False, output, time.monotonic() - start)
            return _result(True, output, time.monotonic() - start)
        except (OSError, ValueError) as exc:
            # ValueError covers UnicodeDecodeError on binary files.
            return _result(False, "", time.monotonic() - start, str(exc))

    def _resolve_path(self, raw_path: str) -> Path:
        """Resolve and validate a relative path under the allowed root.

        Models often address the workspace root as "/", "./" or "" and prefix
        relative paths with "/"; those are normalised to the root. Real absolute
        paths (drive letters, UNC) and any ".." traversal are rejected.
        """
        cleaned = raw_path.strip().replace("\\", "/")
        denied = OSError(
            f"Access denied: '{raw_path}' is outside the CORTEX workspace. "
            "Only paths relative to the workspace root are allowed."
        )
        if cleaned in ("", ".", "/", "./", "~"):
            cleaned = "."
        elif cleaned.startswith("/") and not cleaned.startswith("//"):
            # "/README.md" usually means the workspace root, but "/etc/passwd"
            # is a real absolute path: only accept it if it exists in the workspace.
            relative = cleaned.lstrip("/")
            if not (self._allowed_root / relative).exists():
                raise denied
            cleaned = relative
        candidate = Path(cleaned)
        if ".." in candidate.parts:
            raise OSError("Path must be relative and must not contain '..'.")
        # Checked textually as well: on Linux, Path("C:/Windows") has no drive and
        # is not absolute, so a Windows-style path would otherwise be treated as a
        # relative folder named "C:" instead of being denied.
        if (
            candidate.is_absolute()
            or candidate.drive
            or cleaned.startswith("//")
            or _WINDOWS_DRIVE.match(cleaned)
        ):
            raise denied
        resolved = (self._allowed_root / candidate).resolve()
        if not resolved.is_relative_to(self._allowed_root):
            raise denied
        return resolved

    async def _read_file(self, path: Path) -> str:
        """Read a UTF-8 text file up to the maximum allowed size."""
        if not path.exists():
            raise OSError(f"File not found: {self._display(path)}")
        if not path.is_file():
            raise OSError(f"Not a file: {self._display(path)}")
        if path.stat().st_size > _MAX_READ_BYTES:
            raise OSError("File exceeds 1MB read limit.")
        raw = await anyio.Path(path).read_bytes()
        if b"\x00" in raw[:4096]:
            raise OSError(f"{self._display(path)} looks like a binary file; only text can be read.")
        return raw.decode("utf-8", errors="replace")

    async def _write_file(self, path: Path, content: str, overwrite: bool = False) -> str:
        """Write UTF-8 content to an allowed extension under the allowed root.

        Existing files are never replaced unless ``overwrite`` is set, and hidden
        paths (.git, .venv, .env, the .cortex data dir) are never written. This
        keeps a manipulated or confused model from clobbering the workspace — a
        "delete everything" prompt cannot turn into "overwrite everything".
        """
        relative = path.relative_to(self._allowed_root)
        if any(part.startswith(".") for part in relative.parts):
            raise OSError(f"Writing to hidden paths is not allowed: {relative}")
        if path.exists() and not overwrite:
            raise OSError(
                f"{relative} already exists. It was not changed. Only set overwrite=true "
                "if the user explicitly asked to replace this file."
            )
        if path.suffix not in self._allowed_extensions:
            allowed = ", ".join(sorted(self._allowed_extensions))
            raise OSError(f"Extension '{path.suffix}' is not allowed for writes (allowed: {allowed}).")
        if len(content.encode("utf-8")) > _MAX_WRITE_BYTES:
            raise OSError("Content exceeds 512KB write limit.")
        path.parent.mkdir(parents=True, exist_ok=True)
        await anyio.Path(path).write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} characters to {self._display(path)}"

    async def _list_directory(self, path: Path) -> str:
        """List entries in a directory (directories are suffixed with '/')."""
        if not path.is_dir():
            raise OSError(f"Not a directory: {self._display(path)}")
        entries = sorted(
            (p.name + "/" if p.is_dir() else p.name)
            for p in path.iterdir()
            if not p.name.startswith(".")
        )
        return json.dumps(entries)

    def _display(self, path: Path) -> str:
        """Render a path relative to the workspace root for messages."""
        try:
            return str(path.relative_to(self._allowed_root)) or "."
        except ValueError:
            return path.name


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
