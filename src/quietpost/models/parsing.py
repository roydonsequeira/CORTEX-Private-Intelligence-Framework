"""Tolerant parsing of structured output from small local models.

7-8B models asked for "ONLY a JSON array" routinely wrap it in a ```json fence,
prefix it with a sentence, or (for reasoning models) emit a <think> block
first. These helpers recover the JSON payload instead of failing the caller.
"""

import json
import re
from typing import Any

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_CODE_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
_TOOL_CALL_BLOCK = re.compile(r"<tool_call>.*?(?:</tool_call>|\Z)", re.DOTALL | re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    """Remove <think>…</think> blocks emitted by reasoning models."""
    return _THINK_BLOCK.sub("", text).strip()


def is_tool_call_text(text: str) -> bool:
    """True when a reply is nothing but tool calls written out as text.

    The server turns a tool call into structured data only when the request offered
    tools. Seen (qwen3:8b): asked for a final answer with no tools, after a run that
    had used them, the model wrote a <tool_call> block anyway and it reached the chat.
    Text that merely mentions the tag, such as an answer about the format, is left alone.
    """
    reply = strip_reasoning(text)
    return bool(reply) and not _TOOL_CALL_BLOCK.sub("", reply).strip()


def extract_json(raw: str) -> Any | None:
    """Return the first JSON value found in raw model output, or None."""
    text = strip_reasoning(raw)
    fenced = _CODE_FENCE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    for opener, closer in (("[", "]"), ("{", "}")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except (json.JSONDecodeError, ValueError):
                continue
    return None


def extract_string_list(raw: str, keys: tuple[str, ...] = ("steps", "plan", "items")) -> list[str]:
    """Parse a JSON list of strings, accepting ``{"steps": [...]}`` wrappers and
    list items that are objects (their first string value is used).

    Falls back to non-empty text lines, skipping bare JSON punctuation and code
    fences. Returns an empty list when nothing usable is found.
    """
    data = extract_json(raw)
    if isinstance(data, dict):
        for key in keys:
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if isinstance(data, list):
        items: list[str] = []
        for item in data:
            if isinstance(item, dict):
                item = next((v for v in item.values() if isinstance(v, str)), "")
            text = str(item).strip()
            if text:
                items.append(text)
        return items
    lines: list[str] = []
    for line in strip_reasoning(raw).splitlines():
        cleaned = line.strip().strip(",").strip('"').lstrip("0123456789.)-* ").strip()
        if not cleaned or cleaned.startswith("```") or cleaned in {"[", "]", "{", "}"}:
            continue
        lines.append(cleaned)
    return lines
