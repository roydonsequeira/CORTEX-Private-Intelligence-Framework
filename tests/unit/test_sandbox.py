"""Unit tests for the pluggable code sandboxes."""

from typing import Any
from unittest.mock import MagicMock

import pytest

from cortex.config.settings import Settings
from cortex.tools.sandbox import (
    ContainerSandbox,
    RestrictedSandbox,
    create_sandbox,
)


@pytest.mark.asyncio
async def test_restricted_sandbox_runs_and_blocks_escape() -> None:
    """The restricted backend computes and still blocks the module-traversal escape."""
    ok = await RestrictedSandbox().run("print(6 * 7)", timeout_seconds=5.0)
    assert ok.success is True
    assert "42" in ok.output

    escape = await RestrictedSandbox().run(
        "os = json.codecs.sys.modules.get('os')\nprint(os.getcwd())", timeout_seconds=5.0
    )
    assert escape.success is False
    assert "getcwd" not in escape.output


@pytest.mark.asyncio
async def test_restricted_sandbox_defines_and_calls_function() -> None:
    """A snippet may define a function and call it (single-namespace exec)."""
    code = "def f():\n    return 42\nprint(f())"
    result = await RestrictedSandbox().run(code, timeout_seconds=5.0)
    assert result.success is True
    assert "42" in result.output


@pytest.mark.asyncio
async def test_restricted_sandbox_supports_tuple_unpacking() -> None:
    """Tuple-unpacking assignment works (needs the _unpack_sequence_ guard)."""
    result = await RestrictedSandbox().run("a, b = 1, 2\nprint(a + b)", timeout_seconds=5.0)
    assert result.success is True
    assert "3" in result.output


@pytest.mark.asyncio
async def test_restricted_sandbox_runs_iterative_fibonacci() -> None:
    """The canonical demo snippet (unpacking + loop + function) runs correctly."""
    code = (
        "def fibonacci(n):\n"
        "    a, b = 0, 1\n"
        "    for _ in range(n - 1):\n"
        "        a, b = b, a + b\n"
        "    return a\n"
        "print(fibonacci(20))"
    )
    result = await RestrictedSandbox().run(code, timeout_seconds=5.0)
    assert result.success is True
    assert "4181" in result.output


@pytest.mark.asyncio
async def test_restricted_sandbox_allows_safe_imports() -> None:
    """Common pure-computation modules can be imported (LLMs write `import math`)."""
    result = await RestrictedSandbox().run(
        "import math\nprint(math.comb(20, 2))", timeout_seconds=5.0
    )
    assert result.success is True
    assert "190" in result.output

    from_import = await RestrictedSandbox().run(
        "from math import factorial\nprint(factorial(5))", timeout_seconds=5.0
    )
    assert from_import.success is True
    assert "120" in from_import.output


@pytest.mark.asyncio
async def test_restricted_sandbox_blocks_unsafe_imports() -> None:
    """Importing os/sys/subprocess is refused even with imports enabled."""
    for module in ("os", "sys", "subprocess"):
        result = await RestrictedSandbox().run(f"import {module}", timeout_seconds=5.0)
        assert result.success is False
        assert "not permitted" in (result.error or "")


@pytest.mark.asyncio
async def test_restricted_sandbox_import_does_not_reopen_escape() -> None:
    """An imported safe module cannot be traversed into a forbidden module."""
    result = await RestrictedSandbox().run(
        "import uuid\nprint(uuid.os.getcwd())", timeout_seconds=5.0
    )
    assert result.success is False
    assert "getcwd" not in result.output


def _fake_docker_client(exit_code: int, logs: bytes) -> tuple[Any, MagicMock]:
    """Return a fake docker client and the container mock it produces."""
    container = MagicMock()
    container.wait.return_value = {"StatusCode": exit_code}
    container.logs.return_value = logs
    client = MagicMock()
    client.containers.run.return_value = container
    return client, container


@pytest.mark.asyncio
async def test_container_sandbox_applies_isolation_flags() -> None:
    """The container backend passes the hardening flags to docker run."""
    client, container = _fake_docker_client(exit_code=0, logs=b"42\n")
    sandbox = ContainerSandbox(client=client, mem_limit="128m", pids_limit=64, cpus=0.25)

    result = await sandbox.run("print(6 * 7)", timeout_seconds=5.0)

    assert result.success is True
    assert result.output == "42"
    kwargs = client.containers.run.call_args.kwargs
    assert kwargs["network_disabled"] is True
    assert kwargs["read_only"] is True
    assert kwargs["cap_drop"] == ["ALL"]
    assert kwargs["security_opt"] == ["no-new-privileges:true"]
    assert kwargs["mem_limit"] == "128m"
    assert kwargs["pids_limit"] == 64
    assert kwargs["nano_cpus"] == 250_000_000
    assert kwargs["user"] == "nobody"
    container.remove.assert_called_once_with(force=True)


@pytest.mark.asyncio
async def test_container_sandbox_reports_nonzero_exit_as_error() -> None:
    """A non-zero container exit becomes a failed SandboxResult carrying the logs."""
    client, _ = _fake_docker_client(exit_code=1, logs=b"Traceback: boom")
    sandbox = ContainerSandbox(client=client)

    result = await sandbox.run("raise ValueError('boom')", timeout_seconds=5.0)

    assert result.success is False
    assert "boom" in (result.error or "")


@pytest.mark.asyncio
async def test_container_sandbox_times_out() -> None:
    """A docker wait timeout is surfaced as a timeout error and the container killed."""
    container = MagicMock()
    container.wait.side_effect = Exception("Read timed out")
    client = MagicMock()
    client.containers.run.return_value = container
    sandbox = ContainerSandbox(client=client)

    result = await sandbox.run("while True: pass", timeout_seconds=0.5)

    assert result.success is False
    assert result.error == "Execution timed out."
    container.kill.assert_called_once()


def test_create_sandbox_selects_backend() -> None:
    """create_sandbox honours settings.code_sandbox."""
    assert isinstance(create_sandbox(Settings(code_sandbox="restricted")), RestrictedSandbox)
    assert isinstance(create_sandbox(Settings(code_sandbox="container")), ContainerSandbox)
