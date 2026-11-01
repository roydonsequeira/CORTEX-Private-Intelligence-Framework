"""Unit tests for tolerant parsing of small-model structured output."""

from quietpost.models.parsing import (
    extract_json,
    extract_string_list,
    is_tool_call_text,
    strip_reasoning,
)


def test_extract_json_handles_fences_and_prose() -> None:
    assert extract_json('```json\n["a", "b"]\n```') == ["a", "b"]
    assert extract_json('Sure! Here it is: {"progress": false} hope that helps') == {
        "progress": False
    }
    assert extract_json("no json here") is None


def test_is_tool_call_text_only_when_the_reply_is_nothing_else() -> None:
    call = '<tool_call>\n{"name": "python_exec", "arguments": {"code": "print(1)"}}\n</tool_call>'
    assert is_tool_call_text(call)
    assert is_tool_call_text(f"\n{call}\n{call}\n")
    assert is_tool_call_text(f"<think>I should run it</think>{call}")
    # Cut off before the closing tag, as a reply that hit the token limit is.
    assert is_tool_call_text('<tool_call>\n{"name": "python_exec", "argu')
    # A real answer that shows the format, or that goes on after a closing tag, stays.
    assert not is_tool_call_text(f"Qwen writes a tool call like this:\n{call}")
    assert not is_tool_call_text(f"{call}\nThe answer is 4.")
    assert not is_tool_call_text(f"```\n{call}\n```")
    assert not is_tool_call_text("The answer is 4.")
    assert not is_tool_call_text("   ")


def test_strip_reasoning_removes_think_blocks() -> None:
    assert strip_reasoning("<think>hmm, let me see</think>The answer is 4.") == "The answer is 4."


def test_extract_string_list_accepts_wrappers_and_objects() -> None:
    assert extract_string_list('{"steps": ["one", "two"]}') == ["one", "two"]
    assert extract_string_list('[{"step": "one"}, {"step": "two"}]') == ["one", "two"]


def test_extract_string_list_falls_back_to_clean_lines() -> None:
    raw = "1. Compute the value\n2. Report it\n```"
    assert extract_string_list(raw) == ["Compute the value", "Report it"]
