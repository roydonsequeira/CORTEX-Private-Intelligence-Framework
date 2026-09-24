# CORTEX — Private Intelligence Framework 

> "Intelligence that stays yours."

![Python](https://img.shields.io/badge/Python-3.12-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Docker](https://img.shields.io/badge/Docker-Compose-blue)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)
![Next.js](https://img.shields.io/badge/UI-Next.js-black)

## What is CORTEX?

Cloud AI is powerful, but everything you send to it can be logged, retained, inspected, or used to train future models. Prompts, files, private notes, source code, and research questions leave your machine and enter systems you do not control.

CORTEX is a fully local, privacy-first AI agent framework that runs on your own hardware. It combines Ollama models, a production FastAPI runtime, a Next.js operator UI, four-tier memory, sandboxed tools, LATS reasoning, supervisor-worker orchestration, and OpenTelemetry observability without requiring cloud API keys.

## Demo

<!--
  Record a ~3-minute local walkthrough (see docs/DEMO_RECORDING.md), save it as
  docs/assets/demo.gif, then uncomment the line below to embed it here:
-->
<!-- ![CORTEX live demo](docs/assets/demo.gif) -->

A 3-minute local walkthrough — live token streaming, a sandboxed `python_exec` tool call, cross-session memory, and the request trace in Jaeger. Recording steps: [docs/DEMO_RECORDING.md](docs/DEMO_RECORDING.md).

## Architecture

```mermaid
flowchart TB
    UI["Next.js Operator UI<br/>live SSE streaming"] -->|HTTP · SSE| API["FastAPI Gateway<br/>/chat · /tasks · /memory · /tools · /health"]
    API --> Kernel["Agent Kernel<br/>Planner · Executor (ReAct) · Reflector"]
    Kernel -. hard tasks .-> LATS["LATS<br/>tree search"]
    Kernel -. complex jobs .-> SUP["Supervisor–worker<br/>parallel agents"]

    Kernel --> Mem["Memory Stack"]
    Kernel --> Tools["Tool Registry"]
    Kernel --> Router["Model Router"]

    Mem --> Working["Working"]
    Mem --> Episodic["Episodic · SQLite"]
    Mem --> Semantic["Semantic · ChromaDB"]
    Mem --> Procedural["Procedural · ChromaDB"]

    Tools --> PyExec["python_exec<br/>sandboxed"]
    Tools --> FS["filesystem"]
    Tools --> Web["web_fetch"]
    Tools --> Doc["doc_search"]

    Router --> Ollama["Ollama<br/>llama3.1 · deepseek-r1 · nomic-embed"]
    API -. OTLP spans .-> Jaeger["Jaeger traces<br/>:16686"]
```

## Features

- **Multi-tier memory:** working → episodic SQLite → semantic ChromaDB → procedural tool patterns.
- **LATS-augmented ReAct:** Language Agent Tree Search for hard tasks where a linear loop stalls.
- **Supervisor-worker orchestration:** decomposes complex jobs into parallel sub-agent runs and aggregates results.
- **Sandboxed tool execution:** RestrictedPython code execution, filesystem guardrails, web fetching, document indexing.
- **Plugin-first tools:** drop a `BaseTool` subclass into `src/cortex/tools/plugins/` and restart.
- **Streaming API:** Server-Sent Events for plans, steps, tool calls, tokens, and completion events.
- **Operator UI:** terminal-inspired Next.js interface with memory search and execution trace panel.
- **OpenTelemetry by default:** FastAPI, model calls, tools, memory, and agent spans exported to Jaeger.
- **Release hardening:** request IDs, token bucket rate limiting, graceful shutdown, strict typing, CI.

## Quick Start

```bash
git clone https://github.com/roydonsequeira/CORTEX-Private-Intelligence-Framework.git
cd CORTEX-Private-Intelligence-Framework
make up
make pull-models
```

Open:
- UI: http://localhost:3000
- API docs: http://localhost:8000/docs
- Jaeger traces: http://localhost:16686

### Low-resource demo (no GPU)

For a laptop or quick live demo without a GPU, use the CPU-only stack, which points every model capability at a small model (`llama3.2:1b`):

```bash
make demo
make pull-models-demo
```

Same URLs as above. Answers are weaker than the full 8B stack — this shows the machinery (planning, streaming, tool calls, memory, tracing), not frontier-model quality. Warm the model with one query before presenting, since the first call loads it.

### Running natively (no Docker)

After `pip install -e .` in a virtual environment, use the `cortex` command. It runs inside the environment CORTEX is installed in, so it cannot accidentally pick up another Python's `uvicorn` from `PATH` (which fails with `No module named 'cortex'`):

```bash
cortex doctor   # checks Python, cortex.yaml, Ollama, pulled models, data dirs, port, HTTPS
cortex serve    # starts the API on http://localhost:8000 (warms the model in the background)
```

Without a Jaeger collector, set `telemetry_enabled: false` in `cortex.yaml` (or `CORTEX_TELEMETRY_ENABLED=false`) to skip trace/metric export. `cortex reset-memory --yes` wipes stored memory for a clean start.

## Configuration

All runtime configuration lives in `cortex.yaml` and can be overridden with `CORTEX_` environment variables. CORTEX finds `cortex.yaml` by walking up from the current working directory, so it works from any subdirectory; set `CORTEX_CONFIG` to point at an explicit config file.

| Key | Default | Description |
|---|---|---|
| `ollama_base_url` | `http://localhost:11434` | Ollama server URL |
| `ollama_model` | `llama3.1:8b` | Fast model (FAST capability) |
| `reasoning_model` | `deepseek-r1:8b` | Planner / LATS model (REASONING capability) |
| `code_model` | `qwen2.5-coder:7b` | Code model (CODE capability) |
| `embed_model` | `nomic-embed-text` | Embedding model |
| `ollama_timeout_seconds` | `300` | Per-request Ollama timeout (covers a cold model load) |
| `ollama_num_ctx` | `8192` | Context window pinned on every call |
| `ollama_keep_alive` | `30m` | How long Ollama keeps the model loaded between requests |
| `max_agent_steps` | `10` | ReAct step budget; at the limit CORTEX synthesizes a best-effort answer |
| `history_turns` | `6` | Prior turns of the session replayed into each request |
| `chroma_path` | `./.cortex/chroma` | Local Chroma persistence path |
| `db_path` | `./.cortex/cortex.db` | SQLite episodic memory path |
| `api_host` / `api_port` | `0.0.0.0` / `8000` | FastAPI bind address |
| `api_key` | `null` | When set, require `Authorization: Bearer <key>` on all routes except `/health` and docs |
| `cors_origins` | `["*"]` | Allowed CORS origins; narrow this before exposing the API |
| `task_store` | `memory` | Task result backend: `memory` (lost on restart) or `sqlite` (durable) |
| `task_db_path` | `./.cortex/tasks.db` | SQLite path for the durable task store |
| `telemetry_enabled` | `true` | Export OpenTelemetry traces/metrics; set `false` to run without a collector (no Jaeger) and no export warnings |
| `otel_endpoint` | `http://localhost:4317` | OTLP gRPC endpoint |
| `code_sandbox` | `restricted` | Code execution backend: `restricted` (in-process) or `container` (Docker isolation) |
| `stream_tokens` | `true` | Stream final-answer tokens from Ollama as they generate |
| `procedural_memory_enabled` | `true` | Learn tool-use patterns and feed them to the planner |
| `lats.enabled` | `false` | Enable LATS globally |
| `lats.max_depth` | `5` | LATS tree depth |
| `lats.n_branches` | `3` | LATS branch factor |
| `lats.budget` | `10` | LATS simulation budget |
| `lats.evaluator` | `model` | LATS state scoring: `model` (LLM) or `heuristic` (fast, no LLM) |
| `supervisor.max_workers` | `3` | Parallel worker limit |
| `rate_limit.enabled` | `true` | Enable API rate limiting |
| `rate_limit.requests_per_minute` | `200` | Default per-IP route limit |
| `rate_limit.chat_requests_per_minute` | `60` | Per-IP `/chat` limit |

## Tool System

Create plugins by inheriting from `BaseTool`, defining a JSON schema, and implementing `execute()`:

```python
from typing import ClassVar
from cortex.tools.base import BaseTool, ToolResult, ToolSchema

class EchoTool(BaseTool):
    schema: ClassVar[ToolSchema] = ToolSchema(
        name="echo",
        description="Echo input text.",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
    )

    async def execute(self, **kwargs: object) -> ToolResult:
        return ToolResult(
            tool_name=self.schema.name,
            success=True,
            output=str(kwargs["text"]),
            execution_time_ms=0.0,
        )
```

Drop the file in `src/cortex/tools/plugins/` and restart CORTEX.

## Security Model

CORTEX is designed to run on hardware you control, and its guardrails are built for that threat model — chiefly to contain code the LLM generates when a prompt is malicious or injected.

- **Code execution** uses a pluggable sandbox backend selected by `code_sandbox`:
  - `restricted` (default) — compiled with RestrictedPython and run in a separate spawned process with a hard timeout. The attribute guard blocks dunder access and any traversal that would return a module object, closing escapes such as `json → codecs → sys → sys.modules['os']`. This is a best-effort in-process sandbox, **not** a guarantee against a determined adversary.
  - `container` — each snippet runs in an ephemeral Docker container with the network disabled, a read-only root filesystem, all Linux capabilities dropped, `no-new-privileges`, a tmpfs workdir, and CPU/memory/pid limits, so the OS process boundary is the real isolation layer. Use this for untrusted or multi-tenant workloads (`pip install 'cortex-agent[container]'`).
- **Filesystem** access is confined to a configurable workspace root, rejects `..` traversal and absolute paths, enforces read/write size caps, and restricts writable extensions.
- **Web fetch** honours `robots.txt`, caps response size, and fails closed to an offline message when the network is unavailable.
- **API** requests are validated with Pydantic, rate limited per IP with a token bucket, and refused while the server drains for graceful shutdown. Authentication is off by default for localhost; set `api_key` (e.g. via `CORTEX_API_KEY`) to require a bearer token on every route except `/health` and the docs, and narrow `cors_origins` before exposing the API beyond your machine.

Report security issues privately via a GitHub security advisory rather than a public issue.

## Memory Architecture

```mermaid
flowchart TD
    request[User Request] --> working[Working Memory]
    working --> episodic[Episodic Memory SQLite]
    episodic --> semantic[Semantic Memory ChromaDB]
    tools[Tool Success Patterns] --> procedural[Procedural Memory ChromaDB]
    semantic --> context[Relevant Memory Context]
    episodic --> context
    procedural --> planner[Planner Hints]
```

## Observability

CORTEX exports OpenTelemetry spans for HTTP requests, agent planning/execution/reflection, Ollama calls, SQLite memory, Chroma memory, tool execution, LATS, supervisor workers, and web fetches.

Jaeger is available at `http://localhost:16686`.

Screenshot placeholder:

```
docs/assets/jaeger-trace-placeholder.png
```

## API Examples

```bash
curl -X POST http://localhost:8000/chat/message \
  -H "Content-Type: application/json" \
  -d '{"message":"Tell me the time","session_id":null}'
```

See [DEMO.md](DEMO.md) for reproducible examples.

## Implementation Status

| Phase | Description | Status |
|---|---|---|
| Phase 0 | Foundation, config, Docker, telemetry | Complete |
| Phase 1 | Model provider, router, agent kernel | Complete |
| Phase 2 | Multi-tier memory architecture | Complete |
| Phase 3 | Tool registry and built-in tools | Complete |
| Phase 4 | FastAPI API and SSE streaming | Complete |
| Phase 5 | Next.js operator UI | Complete |
| Phase 6 | LATS and supervisor multi-agent orchestration | Complete |
| Phase 7 | Observability, hardening, docs, release | Complete |

## Roadmap

- [ ] Voice I/O (Whisper + TTS, fully local)
- [ ] Vision tool (LLaVA integration)
- [ ] Multi-modal document processing
- [ ] Agent-to-agent networking (libp2p)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and our [Code of Conduct](CODE_OF_CONDUCT.md). Good first PRs include new tool plugins, memory backends, model provider adapters, and UI trace visualizations. Architecture decisions live in [docs/architecture-decisions](docs/architecture-decisions). Changes are tracked in [CHANGELOG.md](CHANGELOG.md); report vulnerabilities privately per [SECURITY.md](SECURITY.md).

## License

MIT
