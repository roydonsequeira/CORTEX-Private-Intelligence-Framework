# Quietpost Tool Plugins

The plugins in this folder ship with Quietpost and load on every start, from the
installed package. Quietpost imports each module here and registers every
`BaseTool` subclass the module defines, by `schema.name`.

To add a tool to Quietpost itself, put its `.py` file here. To add one only to your
own install, put the file in a folder of your own, set `plugins_dir` to that folder
in `quietpost.yaml`, and restart Quietpost.

Your plugins load after the built-in tools and the plugins here. A plugin never
replaces a tool that is already registered: a tool whose name is taken is skipped
with a `plugin_tool_name_taken` warning, so give yours a new name. Classes a plugin
only imports are not registered. Plugin code runs unsandboxed, with Quietpost's own
permissions, so add only code you trust.

Each plugin must:

- Inherit from `quietpost.tools.base.BaseTool`
- Define `schema: ClassVar[ToolSchema]`
- Implement `async execute(self, **kwargs) -> ToolResult`
- Use a JSON Schema object in `schema.parameters`

Minimal example:

```python
from typing import ClassVar

from quietpost.tools.base import BaseTool, ToolResult, ToolSchema


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
