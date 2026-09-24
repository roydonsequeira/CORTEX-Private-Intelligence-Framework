"""Tests for detecting 'run it' requests on code the sandbox cannot run."""

from cortex.agent.planner import _tidy_steps
from cortex.agent.runnability import cannot_run_reply
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


def test_blocked_stdlib_module_is_named() -> None:
    reply = cannot_run_reply("run it", _history("```python\nimport os\nprint(os.getcwd())\n```"))

    assert reply is not None
    assert "`os`" in reply


def test_plan_steps_are_cut_before_code_and_capped() -> None:
    steps = _tidy_steps(["Write the game:\n```python\nimport pygame\n```", "x" * 500])

    assert steps[0] == "Write the game:"
    assert len(steps[1]) <= 160


def test_plan_step_loses_literal_newline_escapes() -> None:
    steps = _tidy_steps(["Answer directly: write the complete code\\n"])

    assert steps == ["Answer directly: write the complete code"]


def test_plan_of_only_code_falls_back_to_direct_answer() -> None:
    assert _tidy_steps(["```python\nprint(1)\n```"]) == ["Answer directly from knowledge"]
