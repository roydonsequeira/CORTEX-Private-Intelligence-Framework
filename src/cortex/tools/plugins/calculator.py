"""Reference plugin: safe arithmetic calculator."""

import ast
import math
import operator
import time
from collections.abc import Callable
from typing import Any, ClassVar

from cortex.tools.base import BaseTool, ToolResult, ToolSchema

_OPS: dict[type[ast.operator | ast.unaryop], Callable[..., Any]] = {
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

_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "floor": math.floor,
    "ceil": math.ceil,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "factorial": math.factorial,
    "min": min,
    "max": max,
}
_CONSTANTS: dict[str, float] = {"pi": math.pi, "e": math.e, "tau": math.tau}

# The calculator runs in the API process, so unbounded work here would freeze the
# server: `9**9**9` or `factorial(10**9)` never returns. Bound the operands.
_MAX_EXPONENT = 10_000
_MAX_FACTORIAL = 5_000
_MAX_DIGITS = 10_000
_MAX_BITS = 40_000


class CalculatorTool(BaseTool):
    """Simple arithmetic calculator plugin."""

    schema: ClassVar[ToolSchema] = ToolSchema(
        name="calculator",
        description=(
            "Evaluate an arithmetic expression, e.g. '(17 * 23) + sqrt(144)'. Supports "
            "+ - * / // % ** (or ^), parentheses, pi, e, and sqrt, log, exp, sin, cos, "
            "tan, floor, ceil, round, abs, factorial, min, max."
        ),
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
                output=_format(result),
                execution_time_ms=(time.monotonic() - start) * 1000,
            )
        except (SyntaxError, ValueError, TypeError, ZeroDivisionError, OverflowError) as exc:
            return ToolResult(
                tool_name=self.schema.name,
                success=False,
                output="",
                error=str(exc) or type(exc).__name__,
                execution_time_ms=(time.monotonic() - start) * 1000,
            )


def _eval_expr(expression: str) -> float | int:
    """Evaluate arithmetic AST nodes only."""
    normalised = (
        expression.strip()
        .replace("^", "**")
        .replace("×", "*")
        .replace("÷", "/")
        .replace("−", "-")
    )
    if normalised.endswith("="):
        normalised = normalised[:-1]
    node = ast.parse(normalised, mode="eval")
    result = _eval_node(node.body)
    if isinstance(result, int) and result.bit_length() > _MAX_DIGITS * 3.33:
        raise ValueError("Result is too large to display.")
    if not isinstance(result, int | float):
        raise ValueError("Expression did not produce a number.")
    return result


def _eval_node(node: ast.AST) -> Any:
    """Evaluate a restricted arithmetic AST node."""
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value
    if isinstance(node, ast.Name) and node.id in _CONSTANTS:
        return _CONSTANTS[node.id]
    if isinstance(node, ast.BinOp):
        op = _OPS.get(type(node.op))
        if op is not None:
            left = _eval_node(node.left)
            right = _eval_node(node.right)
            if isinstance(node.op, ast.Pow):
                if abs(right) > _MAX_EXPONENT:
                    raise ValueError(f"Exponent too large (limit {_MAX_EXPONENT}).")
                if (
                    isinstance(left, int)
                    and isinstance(right, int)
                    and left.bit_length() * max(right, 0) > _MAX_BITS
                ):
                    raise ValueError("Result is too large to compute.")
            return op(left, right)
    if isinstance(node, ast.UnaryOp):
        op = _OPS.get(type(node.op))
        if op is not None:
            return op(_eval_node(node.operand))
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCTIONS
        and not node.keywords
    ):
        args = [_eval_node(arg) for arg in node.args]
        if node.func.id == "factorial" and args and abs(args[0]) > _MAX_FACTORIAL:
            raise ValueError(f"factorial argument too large (limit {_MAX_FACTORIAL}).")
        return _FUNCTIONS[node.func.id](*args)
    raise ValueError(
        "Only arithmetic expressions are allowed (numbers, + - * / // % **, "
        "parentheses, and math functions such as sqrt)."
    )


def _format(value: Any) -> str:
    """Render a result, dropping float noise such as 12.000000000000002."""
    if isinstance(value, float):
        if value.is_integer() and abs(value) < 1e16:
            return str(int(value))
        return f"{value:.12g}"
    return str(value)
