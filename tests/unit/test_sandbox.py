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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("total = 0\nfor i in range(5):\n    total += i\nprint(total)", "10"),
        ("d = {}\nd['k'] = 1\nprint(d)", "{'k': 1}"),
        ("a = [1, 2, 3]\na[1] = 9\nprint(a)", "[1, 9, 3]"),
        (
            "class P:\n    def __init__(self, n):\n        self.n = n\n"
            "    def double(self):\n        return self.n * 2\nprint(P(4).double())",
            "8",
        ),
        ("print(hasattr('x', 'upper'), bin(10), hex(255))", "True 0b1010 0xff"),
        ("n = 0\nwhile n < 3:\n    n += 1\nprint(n)", "3"),
    ],
)
async def test_restricted_sandbox_runs_common_llm_code(code: str, expected: str) -> None:
    """Patterns LLMs write constantly (+=, item assignment, classes) are supported."""
    result = await RestrictedSandbox().run(code, timeout_seconds=10.0)
    assert result.success is True, result.error
    assert expected in result.output


@pytest.mark.asyncio
async def test_restricted_sandbox_supports_datetime_strptime() -> None:
    """datetime.strptime lazily imports _strptime from C; date maths must still work."""
    code = (
        "from datetime import datetime\n"
        "a = datetime.strptime('2024-01-01', '%Y-%m-%d')\n"
        "b = datetime.strptime('2024-12-25', '%Y-%m-%d')\n"
        "print((b - a).days)"
    )
    result = await RestrictedSandbox().run(code, timeout_seconds=10.0)
    assert result.success is True, result.error
    assert result.output == "359"


@pytest.mark.asyncio
async def test_restricted_sandbox_reports_trailing_expression() -> None:
    """A bare last expression is reported like a REPL (models often omit print)."""
    code = (
        "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n"
        "        a, b = b, a + b\n    return a\nfib(20)"
    )
    result = await RestrictedSandbox().run(code, timeout_seconds=10.0)
    assert result.success is True
    assert result.output == "6765"


@pytest.mark.asyncio
async def test_restricted_sandbox_does_not_leak_loop_underscore() -> None:
    """A top-level `for _ in ...` loop variable is not reported as output."""
    result = await RestrictedSandbox().run(
        "for _ in range(3):\n    pass\nprint('ok')", timeout_seconds=10.0
    )
    assert result.output == "ok"


@pytest.mark.asyncio
async def test_restricted_sandbox_explains_empty_output() -> None:
    """Code that prints nothing says so instead of returning an empty string."""
    result = await RestrictedSandbox().run("def f():\n    return 1", timeout_seconds=10.0)
    assert result.success is True
    assert "print()" in result.output


@pytest.mark.asyncio
async def test_restricted_sandbox_blocks_module_writes_and_getattr_escape() -> None:
    """Writes to modules and getattr traversal into modules stay blocked."""
    write = await RestrictedSandbox().run("math.pi = 3", timeout_seconds=10.0)
    assert write.success is False
    escape = await RestrictedSandbox().run("getattr(json, 'codecs')", timeout_seconds=10.0)
    assert escape.success is False
    assert "not permitted" in (escape.error or "")


def test_trailing_assignment_is_reported_when_nothing_printed() -> None:
    """`result = 2**100 // 3` reports the value instead of 'no output'."""
    from cortex.tools.sandbox import _run_code

    assert _run_code("result = 2**100 // 3") == "result = 422550200076076467165567735125"
    assert _run_code("x = 5\nx += 2") == "x = 7"
    assert _run_code("print(1)\ny = 3") == "1"  # printed output wins
