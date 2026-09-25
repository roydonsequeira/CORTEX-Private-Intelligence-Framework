"""Re-check recorded answers against the current case definitions (no model calls)."""

import json
import sys
from pathlib import Path

import battery_extra
import battery_plan
from harness import check

SUITES = {"battery_plan": battery_plan.CASES, "battery_extra": battery_extra.CASES}

for name, cases in SUITES.items():
    path = sys.argv[1] if len(sys.argv) > 1 and name in sys.argv[1] else f"{name}.jsonl"
    by_id = {c["id"]: c for c in cases}
    passed = failed = 0
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        continue
    for line in lines:
        record = json.loads(line)
        case = by_id[record["id"]]
        problems = []
        for turn, result in zip(case["turns"], record["turns"], strict=False):
            found = check(turn, result)
            if not turn.get("advisory"):
                problems += [f"t{result['turn']}: {p}" for p in found]
        if problems:
            failed += 1
            print(f"FAIL {record['id']}: {'; '.join(problems)}")
        else:
            passed += 1
    print(f"{name}: {passed} passed, {failed} failed (of {len(lines)} run)\n")
