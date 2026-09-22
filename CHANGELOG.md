# Changelog

All notable changes to CORTEX are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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
