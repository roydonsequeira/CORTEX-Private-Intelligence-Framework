# CORTEX — Enhancement & Upgrade Plan (v0.2 → v1.0)

> Companion to `CORTEX_implementation_plan.md`. That document built v0.1 (Phases 0–7,
> all complete). This document covers: the audit of what was built, the flaws found
> and fixed, open-source readiness, and a phased design for hardening and upgrades.

---

## 1. State of the project (audit, 2026-09-11)

The original 7-phase plan is **fully implemented**. Verification on a clean checkout:

| Gate | Result |
|---|---|
| `pytest tests/` | 50 passed |
| `ruff check src tests` | clean |
| `mypy src tests` (strict) | clean, 61 files |
| `ui` lint / typecheck / build config | clean |
| `docker compose config` | valid |

The architecture is coherent and the code quality is genuinely high: typed boundaries
(Pydantic) vs. internal structures, OTel spans on every external I/O, no global mutable
state on the kernel, honest abstractions. This is **well above** the typical "local AI
agent" GitHub project.

---

## 2. Flaws found — and what was fixed in this pass

### Fixed now (committed to working tree)

1. **CRITICAL — sandbox escape in `python_exec`.** `_safe_globals()` injected the real
   `json` module and set `_getattr_ = getattr` (unrestricted). Sandboxed code could walk
   `json.codecs.sys.modules['os']` and reach full OS access (proven with a PoC that ran
   `os.getcwd()`). This directly defeated the project's headline claim.
   **Fix:** a `_guarded_getattr` that blocks dunder names and any attribute access
   returning a module object; added `_getitem_` (which was missing, so subscripting was
   silently broken). Added two regression tests. Verified: escape blocked, legit code
   (subscripting, `math`, `json`, `datetime`, comprehensions) still works.

2. **No `LICENSE` file.** README and `pyproject.toml` declared MIT, but the repo had no
   license file — legally it was "all rights reserved." Added `LICENSE` (MIT).

3. **Dead / misleading infrastructure.** `docker-compose.yml` ran a standalone `chromadb`
   container on port 8001 that the app never used (it embeds `chromadb.PersistentClient`).
   Removed the service and its `depends_on`; documented that Chroma is embedded. Removed
   the obsolete `version:` keys from both compose files (deprecation warnings).

4. **Undocumented security posture.** Added a truthful "Security Model" section to the
   README describing what each guardrail does and does *not* guarantee.

### Known, not yet fixed (tracked below as upgrades)

- **Procedural memory is dead weight.** `MemoryManager.retrieve_tool_patterns()` exists
  and is populated-capable, but the kernel never reads patterns and never writes them
  after a run. The 4th memory tier currently does nothing. → v0.2.
- **Streaming is faked.** `_chunk_text()` slices the *finished* answer into 80-char
  pieces and emits them as `token` events. Ollama supports real token streaming
  (`stream: true`); the provider uses `stream: false`. → v0.2.
- **Task store is in-memory.** `/tasks` results live in a dict lost on restart; the plan
  always intended Redis. → v0.3.
- **No authentication, CORS is `*`.** Fine for localhost; unsafe the moment the API is
  exposed. → v0.3.
- **Rate-limiter bucket dict never evicts** — slow unbounded growth per unique IP. → v0.2.
- **`embed()` sends one HTTP request per text** instead of batching. → v0.2 (perf).
- **Config is read only from the CWD** `cortex.yaml`; running from another directory
  silently uses defaults. → v0.2 (quality-of-life).

---

## 3. Should you open-source this? — Yes.

With the fixes above, yes, and fairly confidently. Reasoning:

- **Quality bar is met.** Strict typing, tests, observability, ADRs, CI. It reads like
  something built deliberately, not a weekend hack — which is exactly the differentiator
  the README promises.
- **No secrets or PII.** No API keys, tokens, or personal data in the tree (your email
  appears only as the git author, which is normal for a public repo).
- **The one disqualifier is now resolved.** Shipping a project whose headline is
  "sandboxed execution" with a trivial sandbox escape would have been the kind of thing
  that gets a harsh top comment on HN/Reddit. That's fixed and regression-tested.

### Pre-release checklist (do before you tag v0.1.0)

- [x] Add `LICENSE`
- [x] Fix the sandbox escape + regression test
- [x] Remove dead Chroma container / compose warnings
- [x] Document the real security model
- [ ] Add `SECURITY.md` (how to report vulnerabilities privately)
- [ ] Add `CHANGELOG.md` (start with v0.1.0)
- [ ] Add a `CODE_OF_CONDUCT.md` (Contributor Covenant) — optional but expected
- [ ] Squash/clean commit history is fine as-is; tag `v0.1.0` via a GitHub Release
- [ ] Add real screenshots/GIF to replace the Jaeger placeholder in the README
- [ ] Enable GitHub "Private vulnerability reporting" in repo settings
- [ ] Confirm CI is green on GitHub (not just locally) before announcing
- [ ] Decide: keep `CORTEX_implementation_plan.md` in-repo (shows intent) or move to
      `docs/` — recommend `docs/` so the repo root stays lean

### Positioning advice

Be honest in the launch post that the in-process sandbox is best-effort and that
container isolation is the recommended mode for untrusted code. Security-literate readers
reward that candor; they punish overclaiming. The "Security Model" README section already
sets this tone.

---

## 4. Upgrade roadmap

Each item below is **Design → Implementation → Acceptance**. Ordered by value-for-effort.
Target: finish v0.2 this weekend; v0.3 and v1.0 are follow-ups.

### v0.2 — "Make the claims true" (highest priority)

#### 4.1 Real token streaming

- **Design.** The provider should expose `stream_complete()` that sets `stream: true` on
  `/api/chat` and yields deltas. The executor, when producing a final answer (no tool
  call), streams tokens straight to the event queue instead of buffering then chunking.
  Remove `_chunk_text`.
- **Implementation.** Add `async def stream_complete(...) -> AsyncIterator[str]` to
  `OllamaProvider` using `httpx`'s `client.stream("POST", ...)` and line-delimited JSON.
  Add a `ModelRouter.stream_complete`. In `Executor.step`, branch: if no tool calls are
  coming, stream; keep the non-streaming path for planner/reflector/evaluator. Guard with
  a `settings.stream_tokens` flag (default true) so `/chat/message` can still buffer.
- **Acceptance.** `/chat/stream` emits `token` events as the model generates, not after;
  an integration test asserts >1 token event arrives before `done` using a fake streaming
  provider.

#### 4.2 Activate procedural memory

- **Design.** Close the learning loop: after a successful run, record the task
  description + the ordered tool names used + step count as a `ToolPattern`. Before
  planning, retrieve similar patterns and pass them to the planner as hints ("Past similar
  tasks used these tools: …") so well-known task shapes skip cold planning.
- **Implementation.** In `AgentKernel.run`, on `status == "complete"`, call
  `memory_manager.store_tool_pattern(...)` built from `state.tool_results`. Before
  `planner.decompose`, call `retrieve_tool_patterns(user_input)` and thread the hint into
  the planner prompt. Add a `settings.procedural_memory_enabled` flag.
- **Acceptance.** Unit test: after one run, a pattern is retrievable for a similar task.
  The planner prompt includes the hint when a pattern exists (assert via a spy router).

#### 4.3 Rate-limiter eviction + batched embeddings + config discovery

- **Design / Implementation.**
  - Rate limiter: evict buckets idle longer than a TTL (e.g. 10 min) on each `_consume`,
    or cap dict size with an LRU. Prevents unbounded growth.
  - `embed()`: when Ollama's `/api/embed` (plural, batch) is available, send all inputs in
    one request; fall back to the per-item loop.
  - Config: resolve `cortex.yaml` by walking up from CWD (and honour a `CORTEX_CONFIG`
    env var) so the app works regardless of launch directory.
- **Acceptance.** Unit tests for eviction and for config resolution from a subdirectory.

### v0.3 — "Make it deployable beyond localhost"

#### 4.4 Optional API authentication + tight CORS

- **Design.** A `settings.api_key` (env-only). When set, a dependency requires
  `Authorization: Bearer <key>` on all non-`/health` routes; when unset, open (localhost
  default preserved). CORS origins become configurable, defaulting to `*` only in dev.
- **Acceptance.** 401 without key when configured; 200 with it; `/health` always open.

#### 4.5 Durable task queue

- **Design.** Abstract the task store behind an interface with two impls: the current
  in-memory dict, and a Redis-backed store (and optionally SQLite for a zero-dependency
  durable option). Background worker and `callback_url` webhook delivery unchanged.
- **Acceptance.** Task results survive an API restart when the durable backend is enabled.

#### 4.6 Open-source hygiene

- `SECURITY.md`, `CHANGELOG.md`, `CODE_OF_CONDUCT.md`, a `.github/dependabot.yml`, and a
  `pre-commit` config running ruff + mypy. Pin dependency lower *and* upper bounds for
  reproducible installs.

### v1.0 — "Real isolation and depth"

#### 4.7 OS-level code sandbox (the proper fix for §2.1)

- **Design.** Replace best-effort RestrictedPython-only execution with a pluggable
  executor backend:
  - `restricted` (current, default for trusted/local use),
  - `container` — run each snippet in an ephemeral, network-disabled,
    read-only-rootfs Docker container with CPU/memory/pids limits and a tmpfs workdir,
  - `wasm` — run via a WASM Python (Pyodide/`wasmtime`) for a capability-free sandbox with
    no host syscalls.
- **Implementation.** Define `CodeSandbox` ABC with `run(code, limits) -> Result`.
  Container backend uses the Docker SDK with `network_disabled=True`,
  `read_only=True`, `mem_limit`, `pids_limit`, `cap_drop=["ALL"]`, `--security-opt
  no-new-privileges`. Select via `settings.code_sandbox = restricted|container|wasm`.
- **Acceptance.** The module-traversal escape *and* a filesystem/network attempt both fail
  under `container`/`wasm`; a CPU bomb is killed by limits, not just the wall-clock timeout.

#### 4.8 LATS & supervisor depth

- LATS currently re-runs the whole ReAct loop per simulated branch (expensive) and
  evaluates via a separate LLM call. Upgrade: cache evaluations, reuse partial states, and
  add a configurable value-model vs. heuristic switch. Add a real end-to-end LATS test
  against a small local model behind an opt-in `@pytest.mark.ollama` marker.

#### 4.9 Live-model smoke tests

- An opt-in integration suite (marker `ollama`, skipped in default CI) that pulls a tiny
  model and asserts a full `/chat/message` round-trip, memory consolidation, and one tool
  call. Gives confidence the mocked units reflect reality.

---

## 5. Suggested commit sequence for this pass

```bash
git checkout -b harden/sandbox-and-release
git add src/cortex/tools/builtin/code_exec.py tests/unit/test_tools.py
git commit   # fix(security): close python_exec module-traversal sandbox escape
git add LICENSE
git commit   # chore: add MIT LICENSE file
git add infra/docker-compose.yml infra/docker-compose.dev.yml
git commit   # fix(infra): drop unused Chroma container and obsolete compose version
git add README.md CORTEX_enhancement_plan.md
git commit   # docs: document security model and add v0.2–v1.0 enhancement plan
```

Open a PR into `main`, let CI prove green on GitHub, then tag `v0.1.0`.

---

*CORTEX. Intelligence that stays yours — and now, a sandbox that actually holds.*
