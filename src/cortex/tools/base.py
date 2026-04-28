"""Base abstractions for the CORTEX tool system."""

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from pydantic import BaseModel


class ToolSchema(BaseModel):
    """JSON Schema descriptor for a tool, used by the model to select and call tools."""

    name: str
    description: str
    parameters: dict[str, Any]


class ToolResult(BaseModel):
    """Outcome of executing a tool, success or failure."""

    tool_name: str
    success: bool
    output: str
    error: str | None = None
    execution_time_ms: float


class BaseTool(ABC):
    """Abstract base class for all CORTEX tools, built-in and plugin."""

    schema: ClassVar[ToolSchema]

    @abstractmethod
    async def execute(self, **kwargs: object) -> ToolResult:
        """Execute the tool with the provided arguments."""
        ...

    def to_ollama_format(self) -> dict[str, Any]:
        """Convert this tool's schema to the Ollama tool-call dict format."""
        return {
            "type": "function",
            "function": {
                "name": self.schema.name,
                "description": self.schema.description,
                "parameters": self.schema.parameters,
            },
        }
