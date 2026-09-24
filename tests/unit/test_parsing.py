"""Unit tests for tolerant parsing of small-model structured output."""

from cortex.models.parsing import extract_json, extract_string_list, strip_reasoning


def test_extract_json_handles_fences_and_prose() -> None:
    assert extract_json('```json\n["a", "b"]\n```') == ["a", "b"]
    assert extract_json('Sure! Here it is: {"progress": false} hope that helps') == {
        "progress": False
    }
    assert extract_json("no json here") is None


def test_strip_reasoning_removes_think_blocks() -> None:
    assert strip_reasoning("<think>hmm, let me see</think>The answer is 4.") == "The answer is 4."


def test_extract_string_list_accepts_wrappers_and_objects() -> None:
    assert extract_string_list('{"steps": ["one", "two"]}') == ["one", "two"]
    assert extract_string_list('[{"step": "one"}, {"step": "two"}]') == ["one", "two"]


def test_extract_string_list_falls_back_to_clean_lines() -> None:
    raw = "1. Compute the value\n2. Report it\n```"
    assert extract_string_list(raw) == ["Compute the value", "Report it"]
