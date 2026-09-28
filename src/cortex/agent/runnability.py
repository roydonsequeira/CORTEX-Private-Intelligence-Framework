"""Decide whether code shown earlier in a conversation can run in the sandbox.

When a user asks to run a program CORTEX just wrote — a pygame game, a
calculator that waits for ``input()`` — a small model tends to paste the whole
program again instead of explaining that it cannot run here. This check lets
the kernel answer that case directly and consistently, without the model.
"""

import ast
import re
import sys

from cortex.models.provider import Message
from cortex.tools.sandbox import _SAFE_MODULES

_RUN_REQUEST = re.compile(r"\b(run|execute|exec|output)\b", re.IGNORECASE)
_CODE_BLOCK = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)(?:```|\Z)", re.DOTALL | re.IGNORECASE)
_IMPORT = re.compile(r"^\s*(?:from|import)\s+([A-Za-z_]\w*)", re.MULTILINE)
_GUI_MODULES = {"tkinter", "turtle", "curses"}
# Import names whose pip package is named differently.
_PIP_NAMES = {
    "bs4": "beautifulsoup4",
    "cv2": "opencv-python",
    "PIL": "pillow",
    "sklearn": "scikit-learn",
    "yaml": "pyyaml",
    "dotenv": "python-dotenv",
    "dateutil": "python-dateutil",
}
_MAX_REQUEST_CHARS = 200
_INPUT_REASON = "it waits for you to type input, and the sandbox has no keyboard"
# Builtins RestrictedPython refuses to compile a call to.
_BLOCKED_CALLS = frozenset({"eval", "exec", "compile"})


def cannot_run_reply(user_input: str, history: list[Message]) -> str | None:
    """Return a ready answer if the user asks to run earlier code that cannot run here.

    Returns None when the request is not a run request, no code was shown in
    the previous assistant message, or the code looks runnable in the sandbox.
    """
    if (
        len(user_input) > _MAX_REQUEST_CHARS
        or "```" in user_input
        or not _RUN_REQUEST.search(user_input)
    ):
        return None
    last_answer = next((m.content for m in reversed(history) if m.role == "assistant"), "")
    code = "\n".join(_CODE_BLOCK.findall(last_answer))
    if not code.strip():
        return None
    reasons, packages = _blockers(code)
    if not reasons:
        return None
    if reasons == [_INPUT_REASON] and re.search(r"\d", user_input):
        # "Run it with 5 and 3": the model can swap input() for those values.
        return None
    install = f"pip install {' '.join(packages)}\n" if packages else ""
    return (
        f"I can't run this program here: {'; '.join(reasons)}. CORTEX runs code in a "
        "locked-down sandbox with no window, no keyboard and no network, "
        "and only safe standard-library modules, so it can only execute code that "
        "computes and prints a result.\n\n"
        "To run it on your machine, save the code as a `.py` file and run:\n\n"
        f"```bash\n{install}python program.py\n```\n\n"
        'Want me to save it to a file? Say, for example, "save it to program.py".'
    )


# Sandbox refusals are policy, not mistakes in the code: a blocked import, a
# forbidden name or attribute, a write through the read-only open().
_SANDBOX_REFUSAL = re.compile(
    r"not permitted in the sandbox|is an invalid (?:variable|attribute) name"
    r"|calls are not allowed|in the sandbox is read-only|cannot be opened in the sandbox",
    re.IGNORECASE,
)
_ERROR_PREFIX = re.compile(r"^(?:\w+(?:Error|Exception):\s*)?(?:Line \d+:\s*)?")


def refused_user_code_reply(code: object, error: str | None, user_input: str) -> str | None:
    """The answer when the sandbox refused code the user supplied themselves.

    Seen live (battery cases I05, I17): asked to run code the sandbox refused,
    the model then did the same thing with the filesystem tool and answered
    with a directory listing, as if the code had run. When the refused code
    is the user's own (it appears in their message), the run ends with this
    answer. Code the model wrote itself can still be fixed and retried.
    """
    if not isinstance(code, str) or not error or not _SANDBOX_REFUSAL.search(error):
        return None
    compact = _compact(code)
    if len(compact) < 8 or compact not in _compact(user_input):
        return None
    reason = _ERROR_PREFIX.sub("", error.split(";")[0].strip()).rstrip(".")
    return (
        f"The sandbox refused to run this code: {reason}. CORTEX runs Python in a "
        "locked-down sandbox (safe standard-library modules only; no system commands, "
        "network access or file writes), so it can only run code that computes and "
        "prints a result."
    )


def _compact(text: str) -> str:
    """Text without whitespace or semicolons and with one quote style, to match code."""
    return re.sub(r"[\s;]+", "", text).replace("'", '"')


def _blocked_constructs(code: str) -> list[str]:
    """Reasons for calls and attributes the sandbox refuses at compile time.

    Seen live: asked to "run" a calculator it had written with
    eval(expression, ..., math.__dict__), the model retried twice, failed, and
    answered with an unrelated request for clarification. Parsed, not searched,
    so a comment that mentions eval() does not count.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    calls = sorted(
        {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _BLOCKED_CALLS
        }
    )
    internals = sorted(
        {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and node.attr.startswith("__")
        }
    )
    reasons = []
    if calls:
        names = " and ".join(f"`{name}()`" for name in calls)
        reasons.append(f"it calls {names}, which the sandbox blocks for safety")
    if internals:
        names = ", ".join(f"`{name}`" for name in internals)
        reasons.append(f"it uses Python internals ({names}) that the sandbox blocks")
    return reasons


def _blockers(code: str) -> tuple[list[str], list[str]]:
    """Return (human-readable reasons, pip packages) that stop code running here."""
    reasons: list[str] = []
    blocked_stdlib: list[str] = []
    packages: list[str] = []
    for module in dict.fromkeys(_IMPORT.findall(code)):
        if module in _SAFE_MODULES or module == "__future__":
            continue
        if module == "pygame":
            reasons.append("it's a pygame game that opens a window and reads the keyboard")
            packages.append("pygame")
        elif module in _GUI_MODULES:
            reasons.append(f"it opens a graphical window ({module})")
        elif module in sys.stdlib_module_names:
            blocked_stdlib.append(module)
        else:
            reasons.append(f"it needs the `{module}` package, which isn't available in the sandbox")
            packages.append(_PIP_NAMES.get(module, module))
    reasons.extend(_blocked_constructs(code))
    if re.search(r"\binput\s*\(", code):
        reasons.append(_INPUT_REASON)
    if blocked_stdlib and not reasons:
        # Only worth naming when nothing bigger (a window, a package) is in the way.
        names = ", ".join(f"`{module}`" for module in blocked_stdlib)
        reasons.append(f"it uses {names}, which the sandbox blocks for safety")
    return reasons, packages
