"""Drive a live CORTEX API exactly like the UI does (SSE) and check each answer.

Run against an isolated server so test conversations, facts and files never
touch your real memory or workspace (see README.md in this folder).
"""

import json
import os
import time
from pathlib import Path
from typing import Any

import httpx

BASE = os.environ.get("CORTEX_EVAL_URL", "http://127.0.0.1:8011")
# The isolated server's CORTEX_ALLOWED_ROOT; used to check a file was NOT written.
WORKSPACE = Path(os.environ.get("CORTEX_EVAL_WORKSPACE", "eval-workspace"))


def run_turn(client: httpx.Client, prompt: str, session_id: str | None) -> dict[str, Any]:
    """Send one message over /chat/stream and collect the events like the UI."""
    events: list[dict[str, Any]] = []
    started = time.monotonic()
    body = {"message": prompt, "session_id": session_id}
    with client.stream("POST", f"{BASE}/chat/stream", json=body) as response:
        if response.status_code != 200:
            return {
                "http": response.status_code,
                "body": response.read().decode()[:300],
                "seconds": round(time.monotonic() - started, 1),
                "answer": "",
                "tools": [],
                "results": [],
                "errors": [],
                "plan": None,
                "session": session_id,
                "done": False,
            }
        buffer = ""
        for chunk in response.iter_text():
            buffer += chunk
            while "\n\n" in buffer:
                frame, buffer = buffer.split("\n\n", 1)
                for line in frame.splitlines():
                    if line.startswith("data: "):
                        events.append(json.loads(line[6:]))
    answer = ""
    for event in events:
        if event["type"] == "token_reset":
            answer = ""
        elif event["type"] == "token":
            answer += event["value"]
    return {
        "http": 200,
        "seconds": round(time.monotonic() - started, 1),
        "session": next((e["value"] for e in events if e["type"] == "session_id"), session_id),
        "plan": next((e["steps"] for e in events if e["type"] == "plan"), None),
        "tools": [
            (e["tool"], json.dumps(e["args"])[:200]) for e in events if e["type"] == "tool_call"
        ],
        "results": [
            (e["tool"], e["success"], (e.get("output") or e.get("error") or "")[:200])
            for e in events
            if e["type"] == "tool_result"
        ],
        "errors": [e["message"] for e in events if e["type"] == "error"],
        "done": any(e["type"] == "done" for e in events),
        "answer": answer.strip(),
    }


def check(turn: dict[str, Any], result: dict[str, Any]) -> list[str]:
    """Return the problems with one turn's result (empty list = pass).

    Checks are heuristics over free text: before "fixing" a failure, read the
    recorded answer — a correct answer can be phrased in a way a check misses.
    """
    problems: list[str] = []
    if "http" in turn:
        if result.get("http") != turn["http"]:
            problems.append(f"http={result.get('http')} expected {turn['http']}")
        return problems
    answer = result.get("answer", "")
    low = answer.lower()
    tools = {t for t, _ in result.get("tools", [])}
    if result.get("http") != 200:
        problems.append(f"http={result.get('http')} {result.get('body')}")
    if not answer:
        problems.append("empty answer")
    if not result.get("done"):
        problems.append("no done event")
    if result.get("errors") and not turn.get("allow_error"):
        problems.append(f"errors={result['errors']}")
    # Numbers may come back with thousands separators ("18,446,744,...").
    plain = low.replace(",", "").replace("\\", "")

    def has(text: str) -> bool:
        return text.lower() in low or text.lower() in plain

    missing = [r for r in turn.get("req", []) if not has(r)]
    if missing:
        problems.append(f"missing={missing}")
    if turn.get("any") and not any(has(a) for a in turn["any"]):
        problems.append(f"none of={turn['any']}")
    present = [n for n in turn.get("notreq", []) if n.lower() in low]
    if present:
        problems.append(f"should not contain={present}")
    need = set(turn.get("tools", []))
    if need - tools:
        problems.append(f"tool not used={sorted(need - tools)}")
    if turn.get("notools") and tools:
        problems.append(f"unexpected tools={sorted(tools)}")
    bad = tools & set(turn.get("forbid", []))
    if bad:
        problems.append(f"forbidden tools={sorted(bad)}")
    if "max_py" in turn:
        blocks = answer.count("```python") + answer.count("```py\n")
        if blocks > turn["max_py"]:
            problems.append(f"python code blocks={blocks}")
    if turn.get("no_file") and (WORKSPACE / turn["no_file"]).exists():
        problems.append(f"file was written: {turn['no_file']}")
    if "maxs" in turn and result.get("seconds", 0) > turn["maxs"]:
        problems.append(f"slow {result['seconds']}s > {turn['maxs']}s")
    return problems


def run_suite(name: str, cases: list[dict[str, Any]], only: set[str] | None = None) -> None:
    """Run cases in order, print progress, and record every turn to <name>.jsonl."""
    passed = failed = 0
    with (
        httpx.Client(timeout=httpx.Timeout(600.0, connect=5.0)) as client,
        open(f"{name}.jsonl", "w", encoding="utf-8") as out,
    ):
        for case in cases:
            if only and case["id"] not in only:
                continue
            session: str | None = None
            case_problems: list[str] = []
            records = []
            for i, turn in enumerate(case["turns"]):
                if turn.get("sleep"):
                    time.sleep(turn["sleep"])  # let background memory consolidation finish
                if turn.get("new_session"):
                    session = None
                result = run_turn(client, turn["p"], session)
                session = result.get("session") or session
                problems = check(turn, result)
                if not turn.get("advisory"):
                    case_problems += [f"t{i + 1}: {p}" for p in problems]
                records.append(
                    {"turn": i + 1, "prompt": turn["p"][:300], "problems": problems, **result}
                )
                tools = [t for t, _ in result.get("tools", [])]
                verdict = "OK" if not problems else "FAIL " + "; ".join(problems)
                print(
                    f"  [{case['id']} t{i + 1}] {result.get('seconds')}s tools={tools} {verdict}",
                    flush=True,
                )
                shown = (result.get("answer") or result.get("body") or "")[:220].replace(
                    "\n", " | "
                )
                print(f"       {shown}", flush=True)
            ok = not case_problems
            passed += ok
            failed += not ok
            print(f"{'PASS' if ok else 'FAIL'} {case['id']}", flush=True)
            record = {"id": case["id"], "ok": ok, "problems": case_problems, "turns": records}
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
    print(f"\n{name}: {passed} passed, {failed} failed", flush=True)
