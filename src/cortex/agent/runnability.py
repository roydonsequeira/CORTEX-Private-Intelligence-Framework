"""Decide whether code shown earlier in a conversation can run in the sandbox.

When a user asks to run a program CORTEX just wrote — a pygame game, a
calculator that waits for ``input()`` — a small model tends to paste the whole
program again instead of explaining that it cannot run here. This check lets
the kernel answer that case directly and consistently, without the model.
"""

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
        "locked-down sandbox with no window, no keyboard, no network or file access, "
        "and only safe standard-library modules, so it can only execute code that "
        "computes and prints a result.\n\n"
        "To run it on your machine, save the code as a `.py` file and run:\n\n"
        f"```bash\n{install}python program.py\n```\n\n"
        'Want me to save it to a file? Say, for example, "save it to program.py".'
    )


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
    if re.search(r"\binput\s*\(", code):
        reasons.append(_INPUT_REASON)
    if blocked_stdlib and not reasons:
        # Only worth naming when nothing bigger (a window, a package) is in the way.
        names = ", ".join(f"`{module}`" for module in blocked_stdlib)
        reasons.append(f"it uses {names}, which the sandbox blocks for safety")
    return reasons, packages
