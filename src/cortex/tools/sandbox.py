"""Pluggable code-execution sandboxes.

Two backends implement the same ``CodeSandbox`` contract:

- ``restricted`` — the default. RestrictedPython compiled code run in a spawned
  process with a hard timeout and a guarded globals dict. Best-effort, in-process,
  and appropriate for trusted/local use.
- ``container`` — each snippet runs real CPython in an ephemeral Docker container
  with the network disabled, a read-only root filesystem, all Linux capabilities
  dropped, ``no-new-privileges``, a tmpfs workdir, and CPU/memory/pid limits. The
  OS process boundary is the isolation layer, so this is the backend for untrusted
  or multi-tenant workloads.

Select the backend with ``settings.code_sandbox``. (A capability-free ``wasm``
backend via Pyodide/wasmtime is a planned third option.)
"""

import ast
import base64
import contextlib
import importlib
import io
import json
import math
import multiprocessing
import operator
import types
import warnings
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from pydantic import BaseModel
from RestrictedPython import compile_restricted
from RestrictedPython.Eval import default_guarded_getitem
from RestrictedPython.Guards import (
    guarded_iter_unpack_sequence,
    guarded_unpack_sequence,
    safe_builtins,
)
from RestrictedPython.PrintCollector import PrintCollector

from cortex.config.settings import Settings
from cortex.observability.tracing import get_tracer

_tracer = get_tracer(__name__)


class SandboxResult(BaseModel):
    """Outcome of running a code snippet in a sandbox."""

    success: bool
    output: str
    error: str | None = None


class CodeSandbox(ABC):
    """A backend that runs a Python snippet under some isolation guarantee."""

    @abstractmethod
    async def run(self, code: str, timeout_seconds: float) -> SandboxResult:
        """Execute code and return its collected stdout or an error."""
        raise NotImplementedError


class RestrictedSandbox(CodeSandbox):
    """Best-effort in-process sandbox using RestrictedPython in a spawned process."""

    async def run(self, code: str, timeout_seconds: float) -> SandboxResult:
        """Run RestrictedPython-compiled code in a spawned, time-limited process."""
        import asyncio

        with _tracer.start_as_current_span("sandbox.restricted.run") as span:
            span.set_attribute("sandbox.backend", "restricted")
            success, output, error = await asyncio.to_thread(
                _run_code_in_subprocess, code, timeout_seconds
            )
        return SandboxResult(success=success, output=output, error=error)


class ContainerSandbox(CodeSandbox):
    """OS-isolated sandbox: each snippet runs in a locked-down Docker container."""

    def __init__(
        self,
        image: str = "python:3.12-slim",
        mem_limit: str = "256m",
        pids_limit: int = 128,
        cpus: float = 0.5,
        client: Any | None = None,
    ) -> None:
        self._image = image
        self._mem_limit = mem_limit
        self._pids_limit = pids_limit
        self._cpus = cpus
        self._client = client

    def _get_client(self) -> Any:
        """Return a Docker client, importing the SDK lazily."""
        if self._client is not None:
            return self._client
        try:
            import docker
        except ImportError as exc:  # optional dependency
            raise RuntimeError(
                "code_sandbox='container' requires the docker SDK: "
                "pip install 'cortex-agent[container]'"
            ) from exc
        self._client = docker.from_env()
        return self._client

    async def run(self, code: str, timeout_seconds: float) -> SandboxResult:
        """Run code in an ephemeral, network-disabled, read-only container."""
        import asyncio

        with _tracer.start_as_current_span("sandbox.container.run") as span:
            span.set_attribute("sandbox.backend", "container")
            span.set_attribute("sandbox.image", self._image)
            return await asyncio.to_thread(self._run_in_container, code, timeout_seconds)

    def _run_in_container(self, code: str, timeout_seconds: float) -> SandboxResult:
        """Blocking Docker run with hard resource and capability limits."""
        client = self._get_client()
        encoded = base64.b64encode(code.encode("utf-8")).decode("ascii")
        command = [
            "python",
            "-c",
            f"import base64;exec(base64.b64decode('{encoded}').decode('utf-8'))",
        ]
        container = client.containers.run(
            self._image,
            command=command,
            detach=True,
            network_disabled=True,
            read_only=True,
            mem_limit=self._mem_limit,
            pids_limit=self._pids_limit,
            nano_cpus=int(self._cpus * 1_000_000_000),
            cap_drop=["ALL"],
            security_opt=["no-new-privileges:true"],
            tmpfs={"/sandbox": "rw,size=64m,noexec"},
            working_dir="/sandbox",
            user="nobody",
        )
        try:
            try:
                result = container.wait(timeout=timeout_seconds)
            except Exception:  # docker read-timeout on a snippet that ran too long
                with contextlib.suppress(Exception):
                    container.kill()
                return SandboxResult(success=False, output="", error="Execution timed out.")
            exit_code = int(result.get("StatusCode", 1))
            logs = container.logs(stdout=True, stderr=True).decode("utf-8", "replace")
            if exit_code == 0:
                return SandboxResult(success=True, output=logs.strip())
            return SandboxResult(
                success=False, output="", error=logs.strip() or "Code execution failed."
            )
        finally:
            with contextlib.suppress(Exception):
                container.remove(force=True)


def create_sandbox(settings: Settings) -> CodeSandbox:
    """Return the code sandbox selected by settings.code_sandbox."""
    if settings.code_sandbox == "container":
        return ContainerSandbox(
            image=settings.code_sandbox_image,
            mem_limit=settings.code_sandbox_mem_limit,
            pids_limit=settings.code_sandbox_pids_limit,
            cpus=settings.code_sandbox_cpus,
        )
    return RestrictedSandbox()


# --- RestrictedPython execution primitives (used by RestrictedSandbox) ---------


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
    except Exception as exc:
        # All errors from untrusted code (including RestrictedPython guard
        # violations, which raise AttributeError) are reported as a failed
        # result rather than crashing the worker with a traceback.
        queue.put((False, "", f"{type(exc).__name__}: {exc}"))


_REPL_VALUE_NAME = "cortex_repl_value"
_NO_OUTPUT_MESSAGE = (
    "(code ran successfully but produced no output — use print() to show results)"
)


def _run_code(code: str) -> str:
    """Run RestrictedPython code and return collected output.

    A single namespace is used for globals and locals so that a top-level
    function definition is visible when the same snippet calls it (with separate
    dicts, ``def f(): ...`` then ``f()`` raises "name 'f' is not defined").

    Like a REPL, a bare expression on the last line (``fibonacci(20)``) has its
    value reported — LLM-written snippets routinely forget to ``print`` it.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        try:
            byte_code = compile_restricted(
                _capture_last_expression(code), "<cortex-python-exec>", "exec"
            )
        except SyntaxError as exc:
            # RestrictedPython packs its policy violations into a tuple of lines.
            detail = exc.args[0] if exc.args else exc
            if isinstance(detail, tuple):
                detail = "; ".join(str(item) for item in detail)
            raise SyntaxError(str(detail)) from None
    namespace = _safe_globals()
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(byte_code, namespace)  # noqa: S102
    printed = namespace.get("_print")
    collected = printed() if callable(printed) else ""
    direct_stdout = stdout.getvalue()
    parts = [part for part in (direct_stdout, collected) if part]
    value = namespace.get(_REPL_VALUE_NAME)
    if value is not None:
        parts.append(value if isinstance(value, str) else repr(value))
    output = "\n".join(parts).strip()
    return output or _NO_OUTPUT_MESSAGE


def _capture_last_expression(code: str) -> str | ast.Module:
    """Rewrite a trailing bare expression into an assignment the runner can read.

    ``print(...)`` calls are left alone (their value is always None). Code that
    does not parse is returned unchanged so RestrictedPython reports the error.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    if not tree.body or not isinstance(tree.body[-1], ast.Expr):
        return code
    last = tree.body[-1]
    if (
        isinstance(last.value, ast.Call)
        and isinstance(last.value.func, ast.Name)
        and last.value.func.id == "print"
    ):
        return code
    tree.body[-1] = ast.copy_location(
        ast.Assign(
            targets=[ast.Name(id=_REPL_VALUE_NAME, ctx=ast.Store())],
            value=last.value,
        ),
        last,
    )
    ast.fix_missing_locations(tree)
    return tree


# Pure-computation stdlib modules the sandbox permits `import` for. None of these
# expose the filesystem, network, subprocess, or interpreter internals, and the
# attribute guard still blocks any traversal from them into other modules — so
# `import os` stays blocked and the json -> codecs -> sys -> os escape still fails.
_SAFE_MODULES = frozenset(
    {
        "math", "cmath", "statistics", "random", "decimal", "fractions",
        "json", "datetime", "itertools", "functools", "operator", "re",
        "string", "textwrap", "collections", "heapq", "bisect", "calendar",
        "uuid", "hashlib", "base64", "unicodedata", "typing", "enum", "dataclasses",
        "time",
        # Imported lazily from C by datetime.strptime/time.strptime; that import
        # runs through the sandbox's __import__, so it must be allowed explicitly.
        "_strptime",
    }
)


def _safe_import(
    name: str,
    _globals: Any = None,
    _locals: Any = None,
    fromlist: tuple[str, ...] = (),
    level: int = 0,
) -> Any:
    """A restricted ``__import__`` allowing only a whitelist of safe stdlib modules.

    LLM-generated code routinely writes ``import math`` / ``import json``; without
    this, every such snippet fails with "__import__ not found". Only pure-
    computation modules in ``_SAFE_MODULES`` are permitted; anything else (os, sys,
    subprocess, socket, …) raises ImportError.
    """
    if level != 0:
        raise ImportError("relative imports are not permitted in the sandbox")
    root = name.split(".")[0]
    if root not in _SAFE_MODULES:
        raise ImportError(f"import of '{name}' is not permitted in the sandbox")
    module = importlib.import_module(name)
    # Match __import__ semantics: bare `import a.b` binds the top package `a`.
    return module if fromlist else importlib.import_module(root)


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


def _guarded_hasattr(obj: object, name: str) -> bool:
    """``hasattr`` routed through the same guard as attribute reads."""
    try:
        _guarded_getattr(obj, name)
    except AttributeError:
        return False
    return hasattr(obj, name)


def _guarded_write(obj: Any) -> Any:
    """Write guard for ``obj.attr = v`` / ``obj[k] = v``.

    Each snippet runs in its own short-lived process, so mutating ordinary
    objects (lists, dicts, instances of snippet-defined classes) cannot affect
    CORTEX. Writes onto modules are still refused, and underscore attributes are
    already rejected at compile time by RestrictedPython.
    """
    if isinstance(obj, types.ModuleType):
        raise TypeError("modifying modules is not permitted in the sandbox")
    return obj


_INPLACE_OPS: dict[str, Any] = {
    "+=": operator.iadd,
    "-=": operator.isub,
    "*=": operator.imul,
    "/=": operator.itruediv,
    "//=": operator.ifloordiv,
    "%=": operator.imod,
    "**=": operator.ipow,
    "@=": operator.imatmul,
    "<<=": operator.ilshift,
    ">>=": operator.irshift,
    "&=": operator.iand,
    "|=": operator.ior,
    "^=": operator.ixor,
}


def _inplace_var(op: str, target: Any, value: Any) -> Any:
    """Implement augmented assignment (``total += i``) for RestrictedPython."""
    try:
        return _INPLACE_OPS[op](target, value)
    except KeyError:
        raise SyntaxError(f"unsupported augmented assignment operator {op!r}") from None


def _safe_globals() -> dict[str, Any]:
    """Return the restricted globals dict used for all code execution."""
    builtins = dict(safe_builtins)
    builtins.update(
        {
            "__import__": _safe_import,
            "__metaclass__": type,
            "getattr": _guarded_getattr,
            "hasattr": _guarded_hasattr,
            "isinstance": isinstance,
            "issubclass": issubclass,
            "type": type,
            "object": object,
            "super": super,
            "property": property,
            "staticmethod": staticmethod,
            "classmethod": classmethod,
            "iter": iter,
            "next": next,
            "frozenset": frozenset,
            "format": format,
            "bin": bin,
            "hex": hex,
            "oct": oct,
            "ord": ord,
            "chr": chr,
            "repr": repr,
            "hash": hash,
            "callable": callable,
            "slice": slice,
            "complex": complex,
            "bytes": bytes,
            "len": len,
            "range": range,
            "enumerate": enumerate,
            "str": str,
            "int": int,
            "float": float,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "bool": bool,
            "abs": abs,
            "min": min,
            "max": max,
            "sum": sum,
            "round": round,
            "sorted": sorted,
            "reversed": reversed,
            "zip": zip,
            "map": map,
            "filter": filter,
            "all": all,
            "any": any,
            "divmod": divmod,
            "pow": pow,
        }
    )
    return {
        "__builtins__": builtins,
        "__metaclass__": type,
        "__name__": "cortex_sandbox",
        "_print_": PrintCollector,
        "_write_": _guarded_write,
        "_inplacevar_": _inplace_var,
        "_getattr_": _guarded_getattr,
        "_getitem_": default_guarded_getitem,
        "_getiter_": iter,
        "_iter_unpack_sequence_": guarded_iter_unpack_sequence,
        "_unpack_sequence_": guarded_unpack_sequence,
        "json": json,
        "math": math,
        "datetime": datetime,
    }
