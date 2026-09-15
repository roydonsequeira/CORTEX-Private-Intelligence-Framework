"""Python execution tool — delegates to a pluggable CodeSandbox backend."""

import time

from cortex.config.settings import Settings
from cortex.tools.base import BaseTool, ToolResult, ToolSchema
from cortex.tools.sandbox import CodeSandbox, RestrictedSandbox, create_sandbox

_MAX_OUTPUT_CHARS = 8192


class CodeExecutionTool(BaseTool):
    """Execute Python code in a sandbox chosen by configuration.

    The backend is pluggable (see cortex.tools.sandbox): the default
    ``restricted`` backend compiles with RestrictedPython and runs in a spawned,
    time-limited process with a guarded globals dict — best-effort, in-process,
    for trusted/local use. The ``container`` backend runs each snippet in an
    ephemeral, network-disabled, read-only Docker container with dropped
    capabilities and CPU/memory/pid limits, giving OS-level isolation for
    untrusted workloads.
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

    def __init__(
        self, timeout_seconds: float = 10.0, sandbox: CodeSandbox | None = None
    ) -> None:
        self._timeout_seconds = timeout_seconds
        self._sandbox = sandbox or RestrictedSandbox()

    @classmethod
    def from_settings(cls, settings: Settings) -> "CodeExecutionTool":
        """Create a CodeExecutionTool with the sandbox backend from CORTEX settings."""
        return cls(settings.code_exec_timeout_seconds, sandbox=create_sandbox(settings))

    async def execute(self, **kwargs: object) -> ToolResult:
        """Run the submitted code in the configured sandbox backend."""
        start = time.monotonic()
        code = str(kwargs["code"])
        result = await self._sandbox.run(code, self._timeout_seconds)
        if result.success:
            return ToolResult(
                tool_name=self.schema.name,
                success=True,
                output=_truncate(result.output),
                execution_time_ms=(time.monotonic() - start) * 1000,
            )
        return _error(result.error or "Code execution failed.", start)


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
