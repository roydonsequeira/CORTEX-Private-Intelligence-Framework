"""Tests for detecting 'run it' requests on code the sandbox cannot run."""

import pytest

from cortex.agent.planner import _tidy_steps
from cortex.agent.runnability import cannot_run_reply, refused_user_code_reply
from cortex.models.provider import Message

_SNAKE = """Here is a snake game:

```python
import pygame
import random
import sys

pygame.init()
```
"""

_CALCULATOR = """```python
def add(a, b):
    return a + b

a = float(input("First number: "))
b = float(input("Second number: "))
print(add(a, b))
```"""

_PRIMES = """```python
import math

print([n for n in range(2, 20) if all(n % d for d in range(2, int(math.sqrt(n)) + 1))])
```"""


def _history(answer: str) -> list[Message]:
    return [
        Message(role="user", content="write a python program"),
        Message(role="assistant", content=answer),
    ]


def test_pygame_run_request_gets_a_short_explanation() -> None:
    reply = cannot_run_reply("run the code and send me the output", _history(_SNAKE))

    assert reply is not None
    assert "pygame" in reply
    assert "pip install pygame" in reply
    assert "import random" not in reply  # the program is not pasted again
    assert "`sys`" not in reply  # the window is the reason worth naming


def test_input_program_run_request_is_explained() -> None:
    reply = cannot_run_reply("run it", _history(_CALCULATOR))

    assert reply is not None
    assert "type input" in reply
    assert "pip install" not in reply


def test_input_program_with_given_values_goes_to_the_model() -> None:
    assert cannot_run_reply("run it with 5 and 3", _history(_CALCULATOR)) is None


def test_runnable_code_goes_to_the_model() -> None:
    assert cannot_run_reply("run it", _history(_PRIMES)) is None


def test_non_run_request_goes_to_the_model() -> None:
    assert cannot_run_reply("add a score counter", _history(_SNAKE)) is None


def test_run_request_without_earlier_code_goes_to_the_model() -> None:
    assert cannot_run_reply("run it", _history("Sure, what should I run?")) is None


def test_new_code_in_the_request_goes_to_the_model() -> None:
    assert cannot_run_reply("run this:\n```python\nprint(1)\n```", _history(_SNAKE)) is None


def test_unknown_package_is_named_with_install_command() -> None:
    reply = cannot_run_reply("execute it", _history("```python\nimport numpy as np\n```"))

    assert reply is not None
    assert "pip install numpy" in reply


def test_install_command_uses_pip_package_names() -> None:
    code = "```python\nimport requests\nfrom bs4 import BeautifulSoup\n```"
    reply = cannot_run_reply("run it", _history(code))

    assert reply is not None
    assert "pip install requests beautifulsoup4" in reply


def test_blocked_stdlib_module_is_named() -> None:
    reply = cannot_run_reply("run it", _history("```python\nimport os\nprint(os.getcwd())\n```"))

    assert reply is not None
    assert "`os`" in reply


def test_refused_user_code_gets_a_plain_answer() -> None:
    """Live I05: the sandbox's refusal of the user's own code is the answer."""
    reply = refused_user_code_reply(
        "import subprocess; subprocess.run(['dir'])",
        "ImportError: import of 'subprocess' is not permitted in the sandbox",
        "Run this Python: import subprocess; subprocess.run(['dir'])",
    )

    assert reply is not None
    assert reply.startswith(
        "The sandbox refused to run this code: import of 'subprocess' is not permitted"
    )


def test_refused_code_matching_ignores_formatting() -> None:
    """The model may split `a; b` over lines or switch quote style."""
    reply = refused_user_code_reply(
        'import subprocess\nprint(subprocess.run(["dir"]))',
        "ImportError: import of 'subprocess' is not permitted in the sandbox",
        "Run this Python: import subprocess; print(subprocess.run(['dir']))",
    )

    assert reply is not None


@pytest.mark.parametrize(
    ("code", "error", "request_text"),
    [
        # The model's own code: it may fix it or use another tool.
        (
            "import os\nprint(os.listdir('.'))",
            "ImportError: import of 'os' is not permitted in the sandbox",
            "Use Python to list the files here",
        ),
        # An ordinary bug, not a refusal.
        ("print(1 / 0)", "ZeroDivisionError: division by zero", "Run this Python: print(1 / 0)"),
    ],
)
def test_model_code_and_ordinary_errors_go_back_to_the_model(
    code: str, error: str, request_text: str
) -> None:
    assert refused_user_code_reply(code, error, request_text) is None


def test_eval_calculator_run_request_is_explained() -> None:
    """Live: a calculator built on eval(..., math.__dict__) got a confused answer."""
    code = (
        "```python\nimport math\n\ndef calculator(expression):\n"
        '    return eval(expression, {"__builtins__": None}, math.__dict__)\n```'
    )
    reply = cannot_run_reply("run it", _history(code))

    assert reply is not None
    assert "`eval()`" in reply
    assert "`__dict__`" in reply
    assert "python program.py" in reply


def test_eval_in_a_comment_or_string_is_not_a_blocker() -> None:
    """Only real calls count: code that mentions eval() still goes to the model."""
    code = "```python\n# avoid eval() here\nprint('eval() is unsafe')\n```"

    assert cannot_run_reply("run it", _history(code)) is None


def test_main_guard_program_goes_to_the_model() -> None:
    """The sandbox runs `if __name__ == "__main__":` programs, so they are not refused."""
    code = '```python\ndef main():\n    print(42)\n\nif __name__ == "__main__":\n    main()\n```'

    assert cannot_run_reply("run it", _history(code)) is None


def test_plan_steps_are_cut_before_code_and_capped() -> None:
    steps = _tidy_steps(["Write the game:\n```python\nimport pygame\n```", "x" * 500])

    assert steps[0] == "Write the game:"
    assert len(steps[1]) <= 160


def test_plan_step_loses_literal_newline_escapes() -> None:
    steps = _tidy_steps(["Answer directly: write the complete code\\n"])

    assert steps == ["Answer directly: write the complete code"]


def test_plan_of_only_code_falls_back_to_direct_answer() -> None:
    assert _tidy_steps(["```python\nprint(1)\n```"]) == ["Answer directly from knowledge"]


def test_plan_steps_without_words_are_dropped() -> None:
    """A planner that answers ("2") instead of planning falls back to a direct answer."""
    assert _tidy_steps(["2"]) == ["Answer directly from knowledge"]
    assert _tidy_steps(["[2, 3, 5]", "Run it with python_exec"]) == ["Run it with python_exec"]
    assert _tidy_steps(["नमस्ते का जवाब दें"]) == ["नमस्ते का जवाब दें"]
