"""Reference plugin: safe arithmetic calculator."""

import ast
import operator
import time
from collections.abc import Callable
from typing import ClassVar

from cortex.tools.base import BaseTool, ToolResult, ToolSchema

_OPS: dict[type[ast.operator | ast.unaryop], Callable[..., float | int]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


class CalculatorTool(BaseTool):
    """Simple arithmetic calculator plugin."""

    schema: ClassVar[ToolSchema] = ToolSchema(
        name="calculator",
        description="Evaluate a simple arithmetic expression without eval().",
        parameters={
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
            "additionalProperties": False,
        },
    )

    async def execute(self, **kwargs: object) -> ToolResult:
        """Evaluate a simple arithmetic expression."""
        start = time.monotonic()
        try:
            result = _eval_expr(str(kwargs["expression"]))
            return ToolResult(
                tool_name=self.schema.name,
                success=True,
                output=str(result),
                execution_time_ms=(time.monotonic() - start) * 1000,
            )
        except (SyntaxError, ValueError, TypeError, ZeroDivisionError) as exc:
            return ToolResult(
                tool_name=self.schema.name,
                success=False,
                output="",
                error=str(exc),
                execution_time_ms=(time.monotonic() - start) * 1000,
            )


def _eval_expr(expression: str) -> float | int:
    """Evaluate arithmetic AST nodes only."""
    node = ast.parse(expression, mode="eval")
    return _eval_node(node.body)


def _eval_node(node: ast.AST) -> float | int:
    """Evaluate a restricted arithmetic AST node."""
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value
    if isinstance(node, ast.BinOp):
        op = _OPS.get(type(node.op))
        if op is not None:
            return op(_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _OPS.get(type(node.op))
        if op is not None:
            return op(_eval_node(node.operand))
    raise ValueError("Only arithmetic expressions are allowed.")
