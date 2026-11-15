"""Built-in tool registration helpers."""

import contextlib

from quietpost.config.settings import Settings
from quietpost.memory.semantic import SemanticMemory
from quietpost.tools.builtin.code_exec import CodeExecutionTool
from quietpost.tools.builtin.doc_search import DocumentSearchTool
from quietpost.tools.builtin.filesystem import FileSystemTool
from quietpost.tools.builtin.web_fetch import WebFetchTool
from quietpost.tools.registry import ToolRegistry


def register_builtin_tools(
    registry: ToolRegistry,
    settings: Settings,
    semantic_memory: SemanticMemory,
) -> None:
    """Register all built-in tools, then plugin discovery can run afterward."""
    # The workspace is the agent's only folder; create it so the first
    # "list my files" gets an empty listing instead of "not a directory".
    with contextlib.suppress(OSError):
        settings.allowed_root.mkdir(parents=True, exist_ok=True)
    registry.register(FileSystemTool.from_settings(settings))
    registry.register(CodeExecutionTool.from_settings(settings))
    registry.register(WebFetchTool())
    registry.register(DocumentSearchTool(semantic_memory, allowed_root=settings.allowed_root))
