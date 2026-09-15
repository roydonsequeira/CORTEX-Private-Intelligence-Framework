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

import base64
import contextlib
import io
import json
import math
import multiprocessing
import types
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from pydantic import BaseModel
from RestrictedPython import compile_restricted
from RestrictedPython.Eval import default_guarded_getitem
from RestrictedPython.Guards import guarded_iter_unpack_sequence, safe_builtins
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
