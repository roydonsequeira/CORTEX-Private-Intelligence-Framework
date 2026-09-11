"""Python execution tool — RestrictedPython sandbox with timeout and output limits."""

import asyncio
import contextlib
import io
import json
import math
import multiprocessing
import time
import types
from datetime import datetime
from typing import Any

from RestrictedPython import compile_restricted
from RestrictedPython.Eval import default_guarded_getitem
from RestrictedPython.Guards import guarded_iter_unpack_sequence, safe_builtins
from RestrictedPython.PrintCollector import PrintCollector

from cortex.config.settings import Settings
from cortex.tools.base import BaseTool, ToolResult, ToolSchema

_MAX_OUTPUT_CHARS = 8192


class CodeExecutionTool(BaseTool):
    """Execute Python code in a restricted, best-effort sandbox.

    Code is compiled with RestrictedPython and run in a separate spawned process
    with a hard timeout. Attribute access is guarded so that sandboxed code cannot
    traverse from the whitelisted stdlib helpers into arbitrary modules (e.g. the
    ``json -> codecs -> sys -> sys.modules['os']`` escape). RestrictedPython is not
    a security boundary against a determined adversary; for untrusted workloads run
    CORTEX's executor inside the provided Docker container for OS-level isolation.
    """

    schema = ToolSchema(
        name="python_exec",
        description=(
            "Execute Python code in a sandboxed environment. No file access, no network, "
            "no subprocess. Use for computation, data transformation, and analysis."
        ),
        parameters={
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
            "additionalProperties": False,
        },
    )

    def __init__(self, timeout_seconds: float = 10.0) -> None:
        self._timeout_seconds = timeout_seconds

    @classmethod
    def from_settings(cls, settings: Settings) -> "CodeExecutionTool":
        """Create a CodeExecutionTool from CORTEX settings."""
        return cls(settings.code_exec_timeout_seconds)

    async def execute(self, **kwargs: object) -> ToolResult:
        """Compile and execute Python code inside a restricted globals dict."""
        start = time.monotonic()
        code = str(kwargs["code"])
        success, output, error = await asyncio.to_thread(
            _run_code_in_subprocess, code, self._timeout_seconds
        )
        if success:
            return ToolResult(
                tool_name=self.schema.name,
                success=True,
                output=_truncate(output),
                execution_time_ms=(time.monotonic() - start) * 1000,
            )
        return _error(error or "Code execution failed.", start)


def _run_code_in_subprocess(code: str, timeout_seconds: float) -> tuple[bool, str, str | None]:
    """Run code in a subprocess and terminate it on hard timeout."""
    ctx = multiprocessing.get_context("spawn")
    queue: Any = ctx.Queue()
    process = ctx.Process(target=_sandbox_worker, args=(code, queue))
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(1)
        return False, "", "Execution timed out."
    if queue.empty():
        return False, "", "Code execution produced no result."
    success, output, error = queue.get()
    return bool(success), str(output), None if error is None else str(error)


def _sandbox_worker(code: str, queue: Any) -> None:
    """Execute restricted code in an isolated worker process."""
    try:
        queue.put((True, _run_code(code), None))
    except (SyntaxError, NameError, TypeError, ValueError, ImportError) as exc:
        queue.put((False, "", str(exc)))


def _run_code(code: str) -> str:
    """Run RestrictedPython code and return collected output."""
    byte_code = compile_restricted(code, "<cortex-python-exec>", "exec")
    globals_dict = _safe_globals()
    locals_dict: dict[str, Any] = {}
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(byte_code, globals_dict, locals_dict)  # noqa: S102
    printed = locals_dict.get("_print") or globals_dict.get("_print")
    collected = printed() if callable(printed) else ""
    direct_stdout = stdout.getvalue()
    result = locals_dict.get("_")
    parts = [part for part in (direct_stdout, collected) if part]
    if result is not None:
        parts.append(result if isinstance(result, str) else repr(result))
    return "\n".join(parts).strip()


def _guarded_getattr(obj: object, name: str, default: Any = None) -> Any:
    """Attribute guard: block dunders and any access that yields a module object.

    Returning module objects is the primary sandbox-escape vector: from a single
    whitelisted module (e.g. ``json``) an attacker can otherwise reach ``codecs``,
    then ``sys``, then ``sys.modules['os']``. Blocking underscore-prefixed names
    and module returns closes that traversal.
    """
    if name.startswith("_"):
        raise AttributeError(f"access to '{name}' is not permitted in the sandbox")
    value = getattr(obj, name, default)
    if isinstance(value, types.ModuleType):
        raise AttributeError(f"access to module '{name}' is not permitted in the sandbox")
    return value


def _safe_globals() -> dict[str, Any]:
    """Return the restricted globals dict used for all code execution."""
    builtins = dict(safe_builtins)
    builtins.update(
        {
            "len": len,
            "range": range,
            "enumerate": enumerate,
            "str": str,
            "int": int,
            "float": float,
            "list": list,
            "dict": dict,
            "set": set,
            "bool": bool,
        }
    )
    return {
        "__builtins__": builtins,
        "_print_": PrintCollector,
        "_getattr_": _guarded_getattr,
        "_getitem_": default_guarded_getitem,
        "_getiter_": iter,
        "_iter_unpack_sequence_": guarded_iter_unpack_sequence,
        "json": json,
        "math": math,
        "datetime": datetime,
    }


def _truncate(output: str) -> str:
    """Truncate sandbox output to the configured hard limit."""
    if len(output) <= _MAX_OUTPUT_CHARS:
        return output
    return output[:_MAX_OUTPUT_CHARS] + "\n[output truncated]"


def _error(message: str, start: float) -> ToolResult:
    """Build an error ToolResult for code execution."""
    return ToolResult(
        tool_name="python_exec",
        success=False,
        output="",
        error=message,
        execution_time_ms=(time.monotonic() - start) * 1000,
    )
