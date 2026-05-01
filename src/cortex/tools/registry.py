"""Tool registry — auto-discovery, schema validation, and instrumented execution."""

import importlib.util
import time
from pathlib import Path
from typing import Any

import jsonschema
import structlog

from cortex.config.settings import Settings
from cortex.exceptions import CortexToolError
from cortex.observability.tracing import get_tracer
from cortex.tools.base import BaseTool, ToolResult, ToolSchema

logger = structlog.get_logger(__name__)
_tracer = get_tracer(__name__)


class ToolRegistry:
    """Holds all registered tools and dispatches execution with OTel instrumentation."""

    def __init__(self, settings: Settings | None = None, timeout_seconds: float = 30.0) -> None:
        self._tools: dict[str, BaseTool] = {}
        self._timeout_seconds = (
            settings.tool_timeout_seconds if settings is not None else timeout_seconds
        )

    def register(self, tool: BaseTool) -> None:
        """Register a tool instance by its schema name."""
        self._tools[tool.schema.name] = tool
        logger.debug("tool_registered", name=tool.schema.name)

    def auto_discover(self, plugins_dir: Path) -> None:
        """Scan plugins_dir for .py files and register any BaseTool subclasses found."""
        if not plugins_dir.exists():
            return
        for path in plugins_dir.glob("*.py"):
            if path.name.startswith("_"):
                continue
            try:
                spec = importlib.util.spec_from_file_location(path.stem, path)
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (
                        isinstance(attr, type)
                        and issubclass(attr, BaseTool)
                        and attr is not BaseTool
                    ):
                        self.register(attr())
            except Exception as exc:
                logger.warning("plugin_load_failed", path=str(path), error=str(exc))

    def get(self, name: str) -> BaseTool | None:
        """Return the tool registered under name, or None if not found."""
        return self._tools.get(name)

    def list_tools(self) -> list[ToolSchema]:
        """Return schemas for all registered tools."""
        return [t.schema for t in self._tools.values()]

    def to_ollama_tools(self) -> list[dict[str, Any]]:
        """Format all tool schemas for the Ollama tool-call API."""
        return [t.to_ollama_format() for t in self._tools.values()]

    async def execute(self, tool_name: str, **kwargs: object) -> ToolResult:
        """Find, validate, and execute a tool. Returns a ToolResult on success or timeout."""
        tool = self._tools.get(tool_name)
        if tool is None:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                output="",
                error=f"Tool '{tool_name}' not found in registry.",
                execution_time_ms=0.0,
            )

        import asyncio

        validation_error = self._validate(tool, kwargs)
        if validation_error is not None:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                output="",
                error=validation_error,
                execution_time_ms=0.0,
            )

        start = time.monotonic()
        with _tracer.start_as_current_span("tool.execute") as span:
            span.set_attribute("tool_name", tool_name)
            span.set_attribute("tool.timeout_seconds", self._timeout_seconds)
            try:
                result = await asyncio.wait_for(
                    tool.execute(**kwargs), timeout=self._timeout_seconds
                )
            except TimeoutError:
                elapsed = (time.monotonic() - start) * 1000
                return ToolResult(
                    tool_name=tool_name,
                    success=False,
                    output="",
                    error="Execution timed out.",
                    execution_time_ms=elapsed,
                )
            except Exception as exc:
                elapsed = (time.monotonic() - start) * 1000
                raise CortexToolError(f"Tool '{tool_name}' raised: {exc}") from exc
        return result

    def _validate(self, tool: BaseTool, kwargs: dict[str, object]) -> str | None:
        """Validate tool kwargs against the tool JSON Schema."""
        try:
            jsonschema.validate(instance=kwargs, schema=tool.schema.parameters)
        except jsonschema.ValidationError as exc:
            return f"Invalid arguments for tool '{tool.schema.name}': {exc.message}"
        return None
