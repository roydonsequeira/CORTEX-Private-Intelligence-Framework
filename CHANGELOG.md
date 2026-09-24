# Changelog

All notable changes to CORTEX are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Security

- Requests to delete, wipe or erase files or folders are refused
  deterministically: the planner knows no tool can delete, and a plan that
  would destroy data is replaced with a refusal before anything runs.
- The API is private by default: it binds to `127.0.0.1` (was `0.0.0.0`),
  allows browser calls only from the local UI origin (CORS was `*`, so any web
  page could drive a local agent that runs code), and rejects unknown `Host`
  names to block DNS rebinding (`allowed_hosts`). The Docker stack publishes
  every port, Ollama's included, on `127.0.0.1` only.
- `POST /tools/{name}/execute`, which runs a tool directly and bypasses the
  agent, is disabled unless `debug_tool_endpoint: true`.
- `cortex serve` warns when binding beyond loopback without an `api_key`.
- A plan that is a direct answer or a refusal now runs with no tools offered,
  and at most three tool calls run per step. Under accumulated memory, a
  "delete every file" prompt had produced 165 tool calls in one step.
- Prompt-injection hardening: destructive requests are refused, tool and file
  content is treated as untrusted data, `filesystem` never overwrites an
  existing file without `overwrite: true` and never writes hidden paths, and
  `doc_search` can only index files inside the workspace.

### Fixed

- `doc_search` chunks CRLF (Windows) documents by paragraph. They had no
  `"\n\n"` separators, so whole files were cut into blind 512-character windows
  that split sentences and tables; the chunk listing the memory tiers now ranks
  first for "which memory tiers does CORTEX have" instead of unrelated fragments.
- "Write Python code ... and run it" runs the code instead of only showing it.
- A run is bounded by `max_run_seconds` (120 s): a stuck request ends with a
  best-effort answer instead of running for minutes.
- Runs whose plan came from procedural hints no longer record a pattern, so an
  unnecessary tool choice cannot reinforce itself.
- The UI renders tables that small models wrap in a ```` ```markdown ```` fence.
- Tests no longer share the in-process Chroma store, which made one
  similarity test order-dependent.
- Procedural memory no longer steers the planner toward tools from unrelated
  tasks: only patterns from near-duplicate tasks (relevance >= 0.6,
  `procedural_min_relevance`) become hints, and hints are advisory. Unrelated
  file-writing patterns had turned a refusal into "Delete all files with the
  filesystem tool", and one bad run could reinforce itself.
- Models without tool support (e.g. gemma3, deepseek-r1) no longer fail every
  request with HTTP 400; CORTEX retries without tools and remembers the model.
- `/tasks` honours `priority` (high before normal before low, FIFO within a
  level); the never-implemented `callback_url` field is removed. The in-memory
  task store keeps the latest 1,000 tasks instead of growing without bound.
- The request-completion log line keeps its `request_id`, and client-supplied
  `X-Request-ID` values are accepted only as short plain tokens.
- A tool argument named `tool_name` no longer crashes the registry.
- Defaults: every chat role now uses `qwen2.5:7b` (one pull, tested end to end);
  the Docker quick start pulled only two of the three configured models and
  started degraded. `make pull-models` no longer needs `jq`, and scripts keep LF
  line endings on Windows checkouts (`.gitattributes`).
- The UI image receives `NEXT_PUBLIC_CORTEX_API_URL` at build time (Next.js
  inlines it; the runtime variable had no effect), installs with `npm ci`, and
  runs as the unprivileged `node` user.
- Code sandbox now runs ordinary Python that defines a function and calls it, and
  code that uses tuple unpacking. `_run_code` used separate globals/locals dicts
  (so `def f(): ...; f()` raised `name 'f' is not defined`), and the
  `_unpack_sequence_` guard was missing (so `a, b = 1, 2` raised
  `name '_unpack_sequence_' is not defined`). The worker also now reports any
  error from untrusted code as a failed result instead of crashing with a
  traceback.
- Code sandbox now permits `import` of a whitelist of pure-computation stdlib
  modules (math, json, random, statistics, itertools, …). LLM-generated code
  routinely writes `import math`, which previously always failed with
  "__import__ not found". Unsafe modules (os, sys, subprocess, …) stay blocked,
  and the attribute guard still prevents traversing an imported module into a
  forbidden one, so the escape and import-blocking regression tests are unchanged.
- Planner and executor prompts now push for the shortest plan (often one step)
  and stopping as soon as the answer is in hand, so small local models stop
  over-decomposing simple tasks into repetitive steps.
- Follow-up questions now work: the user's turn is stored, the session's last
  exchanges are replayed into context, and new session IDs are dashed UUIDs that
  round-trip through the API unchanged (a hex ID came back dashed and split the
  conversation in two).
- Long-term memory no longer fails with "Error creating hnsw segment reader:
  Nothing found on disk": semantic and procedural memory now share one ChromaDB
  client per path instead of opening two on the same directory.
- Tool results reach the model labelled (`[calculator result] 403`) and paired
  with the tool call that produced them; with bare values, qwen2.5:7b sometimes
  reported its own arithmetic instead (397). Identical repeated tool calls are
  not re-run, and only raised tool errors are retried.
- Sandbox supports `+=`, item and attribute assignment, classes,
  `datetime.strptime` and more builtins, and reports a trailing expression's
  value like a REPL. Escapes remain blocked.
- The calculator bounds exponents and factorials; `9**9**9` used to freeze the
  event loop and with it the whole API.
- `web_fetch` verifies TLS against the OS trust store (works behind HTTPS
  inspection), sends a descriptive User-Agent, and reports real errors (an HTTP
  404 is no longer "offline mode").
- File reads and writes are UTF-8 on every platform (Windows defaulted to
  cp1252), and paths outside the workspace get a clear "access denied".
- Ollama is addressed as 127.0.0.1: on Windows, `localhost` cost ~2 s per new
  connection through the IPv6 fallback (2,062 ms vs 23 ms).
- Planner, reflector, LATS and supervisor tolerate fenced or wrapped JSON from
  small models.

### Added

- `cortex` command line: `cortex serve` (starts the API inside the installed
  environment, refusing a busy port), `cortex doctor` (pre-flight check of
  Python, config, Ollama, pulled models, data directories, port and HTTPS) and
  `cortex reset-memory --yes`.
- Reliability for live use: startup model check and background warm-up;
  `ollama_timeout_seconds`, `ollama_num_ctx` and `ollama_keep_alive` settings;
  SSE keep-alive pings; a best-effort answer at the step limit or on a stall;
  time-boxed LATS escalation; actionable errors when Ollama is down or a model
  is not pulled; `/health` names missing models.
- Background memory consolidation that extracts only facts about the user,
  deduplicated by content.
- `telemetry_enabled` setting (default on). Set it `false` (e.g.
  `CORTEX_TELEMETRY_ENABLED=false`) to run without an OpenTelemetry collector:
  the trace/metric exporters are skipped entirely, so a local no-Docker run is
  free of repeated "connection refused" export warnings.
- Configurable `reasoning_model` and `code_model` settings, so every model
  capability (REASONING / FAST / CODE / EMBEDDING) is routed from config. This
  makes it possible to point the whole agent at a single small model.
- A low-resource, CPU-only demo stack (`infra/docker-compose.demo.yml`,
  `make demo` / `make pull-models-demo`) that runs on `llama3.2:1b` for laptops
  and quick live demos without a GPU.
- README architecture diagram (Mermaid), a Demo section with a GIF slot, and a
  demo recording guide (`docs/DEMO_RECORDING.md`).

### Changed

- A pull-request template, a `cortex doctor` field in the bug report, and
  least-privilege (`contents: read`) permissions for the CI workflow.
- Package metadata: authors, project URLs, keywords and classifiers; the unused
  `rich` dependency is removed.
- README quick start covers the native install first (Docker needs an NVIDIA
  GPU), and DEMO.md no longer relies on the debug tool endpoint.

## [1.0.0] - 2026-09-11

### Added

- Pluggable code-execution sandbox (`code_sandbox`): the default `restricted`
  in-process backend, and a `container` backend that runs each snippet in an
  ephemeral Docker container (network disabled, read-only rootfs, dropped
  capabilities, `no-new-privileges`, tmpfs workdir, CPU/memory/pid limits) for
  OS-level isolation of untrusted code. Install with `cortex-agent[container]`.
- Real token streaming from Ollama (`stream: true`) with a `StreamChunk` API on
  the provider and router; the executor streams final-answer tokens to SSE
  clients as they generate. Gated by `stream_tokens` (default on).
- Active procedural memory: successful runs record their tool sequence, and the
  planner receives hints from similar past tasks. Gated by
  `procedural_memory_enabled` (default on).
- Optional API-key authentication (`api_key`) requiring `Authorization: Bearer`
  on all routes except `/health` and docs, and configurable `cors_origins`.
- Durable task store: `task_store = memory | sqlite`, with a zero-dependency
  SQLite backend so task results survive an API restart.
- Config discovery — `cortex.yaml` is found by walking up from the working
  directory, and `CORTEX_CONFIG` can point at an explicit file.
- LATS depth: evaluation results are memoised per state, and a `lats.evaluator`
  switch selects a model or a cheap heuristic value estimate.
- Opt-in live-model smoke tests behind an `ollama` pytest marker (deselected by
  default) covering completion, streaming, a full kernel tool-call + memory run,
  and an end-to-end LATS run.

### Changed

- Embeddings are sent to Ollama's batch `/api/embed` endpoint in one request,
  falling back to the per-item endpoint on older Ollama.

### Fixed

- Rate-limiter token buckets idle beyond a TTL are evicted, bounding memory use.

## [0.1.0] - 2026-09-11

### Added

- Initial public release: local FastAPI runtime, Next.js operator UI, four-tier
  memory (working, episodic SQLite, semantic ChromaDB, procedural), plugin-first
  tool registry, ReAct executor with LATS and supervisor-worker orchestration,
  SSE streaming API, and OpenTelemetry observability.

### Fixed

- Closed a `python_exec` module-traversal sandbox escape and added regression
  tests; added the MIT `LICENSE`; removed an unused ChromaDB container from the
  Docker Compose files; documented the real security model.
