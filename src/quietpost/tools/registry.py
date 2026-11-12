"""Tool registry — plugin loading, schema validation, and instrumented execution."""

import importlib
import importlib.util
import pkgutil
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import jsonschema
import structlog

from quietpost.config.settings import Settings
from quietpost.exceptions import QuietpostToolError
from quietpost.observability.tracing import get_tracer
from quietpost.tools.base import BaseTool, ToolResult, ToolSchema

logger = structlog.get_logger(__name__)
_tracer = get_tracer(__name__)

# The plugins that ship with Quietpost, inside the package itself.
_SHIPPED_PLUGINS_PACKAGE = "quietpost.tools.plugins"
_SHIPPED_PLUGINS_DIR = Path(__file__).resolve().parent / "plugins"


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

    def load_plugins(self, plugins_dir: Path | None = None) -> None:
        """Register the plugins that ship with Quietpost, then those in plugins_dir.

        The shipped ones are imported from the package, so they load in an
        installed copy too, wherever Quietpost is started from. Run this after the
        built-in tools are registered: a plugin never replaces a registered tool.
        """
        for module_info in pkgutil.iter_modules([str(_SHIPPED_PLUGINS_DIR)]):
            if module_info.name.startswith("_"):
                continue
            name = f"{_SHIPPED_PLUGINS_PACKAGE}.{module_info.name}"
            try:
                module = importlib.import_module(name)
            except Exception as exc:
                logger.warning("plugin_load_failed", path=name, error=str(exc))
                continue
            self._register_plugin_tools(module, name)
        if plugins_dir is not None and not _is_shipped_plugins_dir(plugins_dir):
            self.auto_discover(plugins_dir)

    def auto_discover(self, plugins_dir: Path) -> None:
        """Register the tools defined in plugins_dir's .py files (see load_plugins)."""
        if not plugins_dir.is_dir():
            return
        for path in sorted(plugins_dir.glob("*.py")):
            if path.name.startswith("_"):
                continue
            try:
                spec = importlib.util.spec_from_file_location(path.stem, path)
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            except Exception as exc:
                logger.warning("plugin_load_failed", path=str(path), error=str(exc))
                continue
            self._register_plugin_tools(module, str(path))

    def _register_plugin_tools(self, module: ModuleType, source: str) -> None:
        """Register each tool class the plugin module defines.

        A class it only imports is skipped: a plugin that imported FileSystemTool
        used to register a second copy, rooted at the current folder, in place of
        the one confined to the workspace. For the same reason a tool name that is
        already taken keeps its tool, and the plugin's tool is skipped.
        """
        for attr in vars(module).values():
            if not (
                isinstance(attr, type)
                and issubclass(attr, BaseTool)
                and attr.__module__ == module.__name__
            ):
                continue
            try:
                name = attr.schema.name
                if name in self._tools:
                    logger.warning("plugin_tool_name_taken", name=name, plugin=source)
                    continue
                self.register(attr())
            except Exception as exc:
                logger.warning("plugin_load_failed", path=source, error=str(exc))

    def get(self, name: str) -> BaseTool | None:
        """Return the tool registered under name, or None if not found."""
        return self._tools.get(name)

    def list_tools(self) -> list[ToolSchema]:
        """Return schemas for all registered tools."""
        return [t.schema for t in self._tools.values()]

    def to_ollama_tools(self) -> list[dict[str, Any]]:
        """Format all tool schemas for the Ollama tool-call API."""
        return [t.to_ollama_format() for t in self._tools.values()]

    async def execute(self, tool_name: str, /, **kwargs: object) -> ToolResult:
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
                raise QuietpostToolError(f"Tool '{tool_name}' raised: {exc}") from exc
        return result

    def _validate(self, tool: BaseTool, kwargs: dict[str, object]) -> str | None:
        """Validate tool kwargs against the tool JSON Schema."""
        try:
            jsonschema.validate(instance=kwargs, schema=tool.schema.parameters)
        except jsonschema.ValidationError as exc:
            return f"Invalid arguments for tool '{tool.schema.name}': {exc.message}"
        return None


def _is_shipped_plugins_dir(path: Path) -> bool:
    """True when path is the package's own plugins folder, already loaded."""
    try:
        return path.samefile(_SHIPPED_PLUGINS_DIR)
    except OSError:
        return False
