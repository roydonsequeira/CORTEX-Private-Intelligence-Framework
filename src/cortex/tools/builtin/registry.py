"""Built-in tool registration helpers."""

from cortex.config.settings import Settings
from cortex.memory.semantic import SemanticMemory
from cortex.tools.builtin.code_exec import CodeExecutionTool
from cortex.tools.builtin.doc_search import DocumentSearchTool
from cortex.tools.builtin.filesystem import FileSystemTool
from cortex.tools.builtin.web_fetch import WebFetchTool
from cortex.tools.registry import ToolRegistry


def register_builtin_tools(
    registry: ToolRegistry,
    settings: Settings,
    semantic_memory: SemanticMemory,
) -> None:
    """Register all built-in tools, then plugin discovery can run afterward."""
    registry.register(FileSystemTool.from_settings(settings))
    registry.register(CodeExecutionTool.from_settings(settings))
    registry.register(WebFetchTool())
    registry.register(DocumentSearchTool(semantic_memory, allowed_root=settings.allowed_root))
