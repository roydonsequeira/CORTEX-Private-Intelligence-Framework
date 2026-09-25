# Live test battery

187 cases that drive a **running** CORTEX exactly like the UI does (Server-Sent
Events against a real Ollama model), from basic questions to prompt-injection
attacks. The unit tests in `tests/` mock the model; this battery is how the bugs
that only a real 7B model produces were found — a model that says "I saved the
file" without saving it, repastes a program instead of saying it can't run, or
invents a number when its code printed nothing.

| Suite | Cases | What it covers |
|---|---|---|
| `battery_plan.py` | 39 | The six-level test plan: basics, single tools, writing code and "run it", memory, safety, multi-step |
| `battery_extra.py` | 137 | General knowledge, maths, Python, code generation + follow-ups, files, RAG, web fetch, memory, safety, reasoning |
| `battery_ops.py` | 11 | Ollama down, concurrent chats, `/chat/message`, a CPU-heavy snippet not blocking other chats, `cortex doctor`, CORS, Host check, debug endpoint, validation |

## Run it on isolated data

Always point the battery at a **separate** server with its own database, memory
store and workspace, so test conversations, learned facts and files never touch
your real `.cortex/` data.

```bash
# terminal 1 — an isolated server on :8011 (bash; PowerShell: $env:NAME = "value")
mkdir -p /tmp/cortex-eval/ws && cp README.md /tmp/cortex-eval/ws/
CORTEX_DB_PATH=/tmp/cortex-eval/cortex.db \
CORTEX_CHROMA_PATH=/tmp/cortex-eval/chroma \
CORTEX_TASK_DB_PATH=/tmp/cortex-eval/tasks.db \
CORTEX_ALLOWED_ROOT=/tmp/cortex-eval/ws \
cortex serve --port 8011

# terminal 2 — the battery
cd evals/live_battery
export CORTEX_EVAL_URL=http://127.0.0.1:8011 CORTEX_EVAL_WORKSPACE=/tmp/cortex-eval/ws
python battery_plan.py      # ~15 min on an RTX 3060 6 GB
python battery_extra.py     # ~35 min
python battery_ops.py       # ~2 min; starts a second server on :8012 for test 40
```

The workspace needs `README.md` in it (several cases read or index it). Run a
subset by id: `python battery_extra.py "C01 reverse|G04 localhost ssrf"`.

Each run writes `<suite>.jsonl` with every turn: prompt, plan, tool calls and
results, the answer and timings. `python rescore.py` re-checks recorded answers
against the current case definitions without calling the model.

## Reading the results

Checks are heuristics over free text (expected substrings, tools used, no code
repasted, a file *not* written, a time limit). **Before fixing a failure, read
the recorded answer**: a correct answer can be phrased in a way a check misses
(`\frac{1}{2}` for 1/2, "could not be found" for "not found"). A 7B model also
varies run to run, so re-run a failing case before concluding.

## Latest results (qwen2.5:7b, RTX 3060 Laptop 6 GB)

| Run | Plan (39) | Extra (137) | Ops (11) |
|---|---|---|---|
| First run, before fixes | 29 | — | — |
| After the first round of fixes | 38 | 125 | 11 |
| Final verification | **39** | **134** | **11** |

The three remaining misses were fixed and passed on re-run; a targeted re-run of
every case that had ever failed passed 45/45 over three rounds. The battery found
27 issues the unit tests had missed — including an SSRF hole in `web_fetch` and a
prompt injection hidden in quoted text — each fixed in code with a regression
test (see `CHANGELOG.md`, 1.1.0).
