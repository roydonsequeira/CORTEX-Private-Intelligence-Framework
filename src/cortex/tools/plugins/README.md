# CORTEX Tool Plugins

Drop a `.py` file in this directory and restart CORTEX. The `ToolRegistry` scans
this directory, imports each plugin module, instantiates any `BaseTool` subclass,
and registers it by `schema.name`.

Each plugin must:

- Inherit from `cortex.tools.base.BaseTool`
- Define `schema: ClassVar[ToolSchema]`
- Implement `async execute(self, **kwargs) -> ToolResult`
- Use a JSON Schema object in `schema.parameters`

Minimal example:

```python
from typing import ClassVar

from cortex.tools.base import BaseTool, ToolResult, ToolSchema


class EchoTool(BaseTool):
    schema: ClassVar[ToolSchema] = ToolSchema(
        name="echo",
        description="Echo input text.",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
    )

    async def execute(self, **kwargs: object) -> ToolResult:
        return ToolResult(
            tool_name=self.schema.name,
            success=True,
            output=str(kwargs["text"]),
            execution_time_ms=0.0,
        )
```

See `calculator.py` for a reference plugin.
