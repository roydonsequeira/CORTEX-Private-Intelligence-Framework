# CORTEX — Private Intelligence Framework
## End-to-End Local AI Agent | Complete Implementation Plan

> A fully offline, privacy-first autonomous agent framework. Runs entirely on your hardware.
> Built to be the definitive reference implementation for local AI agents.

---

## Why This Stands Out

Most "local AI agent" projects on GitHub are: one-file scripts, LangChain wrappers with a chatbot loop, or Jupyter notebooks. CORTEX is none of those.

What makes CORTEX different:
- **Multi-tier memory architecture** — working, episodic, semantic, and procedural memory as distinct subsystems
- **LATS-augmented ReAct loop** — Language Agent Tree Search for non-trivial tasks, not just flat tool chains
- **Hot-swappable model backends** — swap between Ollama models at runtime with no restart
- **Fully observable** — OpenTelemetry traces from day one, local Jaeger UI
- **Sandboxed code execution** — no tool runs unsandboxed, ever
- **Production-grade Python** — typed, tested, structured. Not a script. Not a notebook.
- **Plugin-first tool registry** — drop a `.py` file into `tools/plugins/`, it auto-registers
- **Docker Compose stack** — one command to spin up the entire infrastructure

This is the project that will get pinned by ML engineers on X. Not because it's the biggest model. Because the architecture is the most coherent and honest.

---

## Project Identity

```
Name:        CORTEX
Tagline:     "Intelligence that stays yours."
Stack:       Python 3.12 · FastAPI · Ollama · ChromaDB · Next.js · Docker
License:     MIT
Target:      GitHub stars, ML practitioners, privacy-focused devs
```

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CORTEX Runtime                                     │
│                                                                               │
│  ┌──────────────┐     ┌────────────────────────────────────────────────┐    │
│  │   Next.js UI │────▶│              FastAPI Gateway                    │    │
│  │  (Port 3000) │     │  /chat  /tasks  /memory  /tools  /traces       │    │
│  └──────────────┘     └────────────────┬───────────────────────────────┘    │
│                                        │                                      │
│                        ┌──────────────▼──────────────┐                      │
│                        │        Agent Kernel          │                      │
│                        │                              │                      │
│                        │  ┌──────────┐  ┌─────────┐  │                      │
│                        │  │  Planner │  │Reflector│  │                      │
│                        │  └────┬─────┘  └────┬────┘  │                      │
│                        │       │              │        │                      │
│                        │  ┌────▼──────────────▼────┐  │                      │
│                        │  │    LATS / ReAct Loop   │  │                      │
│                        │  │   (Executor Engine)    │  │                      │
│                        │  └────────────┬───────────┘  │                      │
│                        └──────────────┼──────────────┘                      │
│                                       │                                       │
│           ┌───────────────────────────┼───────────────────────────┐         │
│           │                           │                             │         │
│   ┌───────▼──────┐         ┌──────────▼─────────┐      ┌─────────▼──────┐  │
│   │ Memory Stack │         │   Tool Registry     │      │  Model Router  │  │
│   │              │         │                     │      │                │  │
│   │ • Working    │         │ • Filesystem        │      │ • Ollama       │  │
│   │ • Episodic   │         │ • Code Exec (jail)  │      │ • llama3.1:8b  │  │
│   │ • Semantic   │         │ • Web Scraper       │      │ • qwen2.5:14b  │  │
│   │ • Procedural │         │ • Doc Search        │      │ • deepseek-r1  │  │
│   └──────────────┘         │ • Shell (opt-in)    │      │ • nomic-embed  │  │
│   ┌──────────────┐         │ • Plugin Loader     │      └────────────────┘  │
│   │  ChromaDB    │         └─────────────────────┘                          │
│   │  (local)     │                                                            │
│   └──────────────┘                                                            │
│                                                                               │
│   ┌───────────────────────────────────────────────────────────────────────┐  │
│   │                    Observability Layer                                 │  │
│   │   OpenTelemetry Collector → Jaeger UI (Port 16686)                    │  │
│   └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
cortex/
├── cortex/                      # Core Python package
│   ├── __init__.py
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── kernel.py            # Main agent orchestrator
│   │   ├── loop.py              # ReAct + LATS execution loop
│   │   ├── planner.py           # Task decomposition (hierarchical)
│   │   ├── executor.py          # Step execution with retry logic
│   │   └── reflector.py        # Self-critique & correction pass
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── base.py              # MemoryStore ABC
│   │   ├── working.py           # In-context scratchpad
│   │   ├── episodic.py          # Session history (SQLite)
│   │   ├── semantic.py          # Vector store (ChromaDB)
│   │   └── procedural.py       # Tool-use patterns (learned)
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── registry.py          # Auto-discovery + schema validation
│   │   ├── base.py              # BaseTool ABC
│   │   ├── builtin/
│   │   │   ├── filesystem.py
│   │   │   ├── code_exec.py     # Sandboxed via restrictedpython
│   │   │   ├── web_fetch.py     # httpx, offline-capable
│   │   │   └── doc_search.py
│   │   └── plugins/             # Drop custom tools here
│   ├── models/
│   │   ├── __init__.py
│   │   ├── provider.py          # OllamaProvider ABC + impl
│   │   └── router.py            # Model selection logic
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py          # Pydantic Settings v2
│   ├── api/
│   │   ├── __init__.py
│   │   ├── server.py            # FastAPI app factory
│   │   ├── routes/
│   │   │   ├── chat.py
│   │   │   ├── tasks.py
│   │   │   ├── memory.py
│   │   │   └── tools.py
│   │   └── middleware/
│   │       ├── telemetry.py
│   │       └── rate_limit.py
│   └── observability/
│       ├── __init__.py
│       ├── tracing.py           # OTel setup
│       └── metrics.py
├── ui/                          # Next.js frontend
│   ├── src/
│   └── package.json
├── tests/
│   ├── unit/
│   ├── integration/
│   └── conftest.py
├── infra/
│   ├── docker-compose.yml
│   ├── docker-compose.dev.yml
│   └── ollama/
│       └── modelfile.sh         # Pull & configure models
├── cortex.yaml                  # Main config file
├── pyproject.toml               # Modern Python packaging
├── Makefile                     # Dev shortcuts
└── README.md
```

---

## Technology Decisions (with rationale)

| Component | Choice | Why |
|---|---|---|
| LLM Runtime | Ollama | Native GPU/CPU, REST API, model management |
| Primary Model | llama3.1:8b / qwen2.5:14b | Balance of speed and reasoning |
| Reasoning Model | deepseek-r1:8b | Chain-of-thought heavy tasks |
| Embeddings | nomic-embed-text | Best quality/size ratio, fully local |
| Vector Store | ChromaDB | Embedded, no server needed, Python-native |
| Session DB | SQLite (async via aiosqlite) | Zero-dependency persistence |
| Backend | FastAPI + uvicorn | Async-native, auto OpenAPI docs |
| Config | Pydantic Settings v2 + YAML | Type-safe, no magic |
| Observability | OpenTelemetry + Jaeger | Industry standard, not vendor lock-in |
| Containerisation | Docker Compose | One-command setup |
| Frontend | Next.js 14 (App Router) | Server components, clean SSE streaming |
| Testing | pytest + pytest-asyncio | Standard, no ceremony |

**Intentionally avoided:**
- LangChain/LlamaIndex — too much abstraction, poor debuggability
- Haystack — overkill for this scope
- Poetry — pyproject.toml with pip is sufficient

---

# IMPLEMENTATION PHASES

Each phase is designed to fit within a single Cursor agent context window. Paste the full prompt for each phase when starting that phase.

---

## PHASE 0 — Foundation & Infrastructure

**Goal:** Repository skeleton, packaging, Docker stack, model pull, config system, base telemetry.
**Estimated time:** 2–3 hours
**Deliverable:** `docker compose up` starts Ollama + ChromaDB + Jaeger. Config loads. Models pulled.

---

### Cursor Prompt — Phase 0

```
You are a senior AI systems engineer with 25+ years of experience building distributed, 
production-grade ML infrastructure. Write Python that would pass a strict code review at 
a top ML platform company. No boilerplate comments, no over-engineering, no LLM-style 
verbosity. Every function does one thing. Every module has a clear boundary.

PROJECT: CORTEX — a fully local, privacy-first AI agent framework.
PHASE 0: Foundation, packaging, infrastructure, and base configuration.

REQUIREMENTS:

1. Initialize a Python 3.12 project using pyproject.toml (no setup.py, no requirements.txt).
   - Package name: cortex-agent
   - Use src layout: src/cortex/
   - Dependencies: fastapi>=0.115, uvicorn[standard], pydantic>=2.7, pydantic-settings>=2.4, 
     httpx>=0.27, chromadb>=0.5, aiosqlite>=0.20, opentelemetry-sdk>=1.25, 
     opentelemetry-exporter-otlp>=1.25, opentelemetry-instrumentation-fastapi>=0.46b0, 
     structlog>=24.4, anyio>=4.6, rich>=13.8
   - Dev dependencies: pytest>=8, pytest-asyncio>=0.24, httpx[testing], ruff, mypy

2. Create cortex/config/settings.py using Pydantic Settings v2:
   - Load from cortex.yaml (primary) with env var overrides
   - Typed fields: ollama_base_url, ollama_model, embed_model, chroma_path, 
     db_path, api_host, api_port, log_level, otel_endpoint
   - No magic strings anywhere in the codebase — all config flows through this

3. Create cortex/config/__init__.py that exposes a get_settings() function with 
   lru_cache — settings are a singleton, not re-loaded per call.

4. Create cortex.yaml with sane defaults:
   ollama_base_url: http://localhost:11434
   ollama_model: llama3.1:8b
   embed_model: nomic-embed-text
   chroma_path: ./.cortex/chroma
   db_path: ./.cortex/cortex.db
   api_host: 0.0.0.0
   api_port: 8000
   log_level: INFO
   otel_endpoint: http://localhost:4317

5. Create cortex/observability/tracing.py:
   - setup_tracing(service_name: str) -> None
   - Initialise OTel SDK with OTLP gRPC exporter
   - Expose get_tracer(name: str) -> opentelemetry.trace.Tracer
   - If the exporter endpoint is unreachable, log a warning and continue — 
     telemetry must never crash the app

6. Create cortex/observability/metrics.py:
   - setup_metrics() -> None
   - Expose a counter for agent_steps_total and a histogram for llm_latency_seconds
   - Use OTel metrics SDK

7. Create cortex/observability/__init__.py that calls both setup functions on import.

8. Create infra/docker-compose.yml:
   Services:
   - ollama: ollama/ollama:latest, GPU passthrough if available (nvidia runtime, 
     fallback to CPU), port 11434, volume ./infra/ollama_data:/root/.ollama
   - chromadb: chromadb/chroma:latest, port 8001, volume ./infra/chroma_data:/chroma/chroma
   - jaeger: jaegertracing/all-in-one:latest, ports 16686:16686 (UI) and 4317:4317 (OTLP gRPC)
   - cortex-api: build from Dockerfile, depends_on ollama and chromadb, port 8000:8000
   All services on a shared bridge network: cortex-net

9. Create infra/docker-compose.dev.yml (override):
   - cortex-api: volume mount ./src:/app/src for hot-reload
   - Set CORTEX_LOG_LEVEL=DEBUG

10. Create a Dockerfile for cortex-api:
    - Base: python:3.12-slim
    - Copy pyproject.toml and src/, install, expose 8000
    - Entrypoint: uvicorn cortex.api.server:create_app --factory --host 0.0.0.0 --port 8000

11. Create infra/ollama/pull_models.sh:
    #!/bin/bash
    # Pull all required models into the Ollama container
    models=("llama3.1:8b" "nomic-embed-text" "deepseek-r1:8b")
    for model in "${models[@]}"; do
        echo "Pulling $model..."
        curl -s -X POST http://localhost:11434/api/pull \
          -d "{\"name\": \"$model\"}" | jq '.status' -r
    done

12. Create a Makefile with targets:
    - make up: docker compose up -d
    - make dev: docker compose -f docker-compose.yml -f docker-compose.dev.yml up
    - make pull-models: bash infra/ollama/pull_models.sh
    - make test: pytest tests/
    - make lint: ruff check src/ && mypy src/
    - make clean: docker compose down -v && rm -rf .cortex/

13. Create a structlog-based logging setup in cortex/observability/logging.py:
    - configure_logging(level: str) -> None
    - Use structlog with JSON renderer in production, ConsoleRenderer in dev
    - All modules should use: logger = structlog.get_logger(__name__)

14. Write tests/unit/test_config.py:
    - Test settings load from cortex.yaml
    - Test env var override
    - Test get_settings() returns same instance (singleton check)

STYLE RULES:
- All public functions have type annotations and a single-line docstring (no multi-line walls of text)
- No commented-out code
- No print() calls — use structlog everywhere
- Private helpers prefixed with underscore
- Exceptions are specific (not bare Exception) and caught at boundaries, not deep in logic
- Use pathlib.Path, not os.path
```

---

## PHASE 1 — Model Provider & Agent Kernel

**Goal:** Ollama provider abstraction, model router, ReAct loop skeleton, base tool ABC.
**Estimated time:** 3–4 hours
**Deliverable:** `AgentKernel.run("tell me the time")` executes a single-step agent loop.

---

### Cursor Prompt — Phase 1

```
CORTEX — Phase 1: Model Provider, Router, and Agent Kernel

Context: Phase 0 is complete. We have config, telemetry, Docker stack. Now we build 
the intelligence layer. This is the most important phase — every subsequent phase 
builds on top of what you define here. Get the abstractions right.

REQUIREMENTS:

1. cortex/models/provider.py — OllamaProvider
   
   class Message(BaseModel):
     role: Literal["system", "user", "assistant", "tool"]
     content: str
     tool_calls: list[ToolCall] | None = None
     tool_results: list[ToolResult] | None = None
   
   class GenerationConfig(BaseModel):
     temperature: float = 0.7
     top_p: float = 0.9
     max_tokens: int = 4096
     stop: list[str] | None = None
   
   class ModelResponse(BaseModel):
     content: str
     model: str
     input_tokens: int
     output_tokens: int
     latency_ms: float
     raw: dict  # full Ollama response, never discard

   class OllamaProvider:
     - __init__(base_url: str, timeout: float = 120.0)
     - async complete(model: str, messages: list[Message], config: GenerationConfig, 
                       tools: list[ToolSchema] | None = None) -> ModelResponse
     - async embed(model: str, text: str | list[str]) -> list[list[float]]
     - async list_models() -> list[str]
     - async health_check() -> bool
     
   Implementation notes:
   - Use httpx.AsyncClient (reuse across calls, don't create per-request)
   - Wrap every network call in a try/except that re-raises as CortexModelError
   - Instrument with OTel spans: name the span "ollama.complete" with model as attribute
   - Record llm_latency_seconds metric on every complete() call

2. cortex/models/router.py — ModelRouter
   
   ModelCapability enum: REASONING, FAST, EMBEDDING, CODE
   
   class ModelRouter:
     - __init__(provider: OllamaProvider, settings: Settings)
     - route(capability: ModelCapability) -> str (returns model name)
     - async complete(...) -> ModelResponse (delegates to provider with routed model)
     - Default routing: REASONING→deepseek-r1:8b, FAST→llama3.1:8b, 
       EMBEDDING→nomic-embed-text, CODE→qwen2.5-coder:7b

3. cortex/tools/base.py — BaseTool

   class ToolSchema(BaseModel):
     name: str
     description: str
     parameters: dict  # JSON Schema for parameters
   
   class ToolResult(BaseModel):
     tool_name: str
     success: bool
     output: str
     error: str | None = None
     execution_time_ms: float
   
   class BaseTool(ABC):
     schema: ClassVar[ToolSchema]  # defined per subclass
     
     @abstractmethod
     async def execute(self, **kwargs) -> ToolResult: ...
     
     def to_ollama_format(self) -> dict:
       # Convert schema to Ollama tool format
       ...

4. cortex/agent/kernel.py — AgentKernel
   
   This is the main orchestrator. Keep it clean — it should read like a 
   state machine, not spaghetti.
   
   class AgentState(BaseModel):
     session_id: str
     user_input: str
     plan: list[str]
     steps_taken: int
     messages: list[Message]
     tool_results: list[ToolResult]
     final_answer: str | None = None
     status: Literal["planning", "executing", "reflecting", "complete", "failed"]
   
   class AgentKernel:
     - __init__(router: ModelRouter, tool_registry: ToolRegistry, 
                memory_manager: MemoryManager, settings: Settings)
     - async run(user_input: str, session_id: str | None = None) -> AgentState
       This is the main entry point. It:
       1. Creates an AgentState
       2. Calls planner.decompose(user_input, state)
       3. Enters the executor loop (max_steps from config)
       4. Calls reflector.evaluate(state) after each step
       5. Returns the final AgentState
     
     Concurrency note: Each call to run() is independent. No shared mutable state 
     on the kernel instance itself.

5. cortex/agent/planner.py — Planner
   
   class Planner:
     - async decompose(user_input: str, state: AgentState) -> list[str]
       Calls the REASONING model with a structured prompt to break the task 
       into concrete, executable steps. Returns a list of step descriptions.
       
       System prompt pattern (write the actual prompt):
       "You are a precise task planner. Given a user request and available tools,
       produce a numbered list of concrete steps. Each step must be independently
       executable. Do not include steps that require human input. Maximum 8 steps."

6. cortex/agent/executor.py — Executor
   
   class Executor:
     - async step(state: AgentState, tool_registry: ToolRegistry, 
                  router: ModelRouter) -> AgentState
       Single step of the ReAct loop:
       1. Build context from state (messages + memory)
       2. Call FAST model with tools injected
       3. If model returns a tool call → execute it, append result to state
       4. If model returns final text → set state.final_answer and status=complete
       5. Increment steps_taken
       
   Implement exponential backoff retry for tool execution failures (max 3 retries).
   Instrument each step with an OTel span.

7. cortex/agent/reflector.py — Reflector
   
   class Reflector:
     - async evaluate(state: AgentState, router: ModelRouter) -> AgentState
       After each step, ask the FAST model whether:
       - The step made progress toward the goal (yes/no)
       - Any correction is needed
       
       If progress=no for two consecutive steps, set status=failed and return.
       This prevents infinite loops without hard-coding a step count alone.
       
       System prompt: be terse. This is called on every step — token efficiency matters.

8. cortex/agent/loop.py — LATS loop (stub for Phase 4)
   
   Just define the class with a TODO for now:
   
   class LATSLoop:
     """Language Agent Tree Search — beam search over the action space.
     Used for complex multi-branch tasks where ReAct alone fails.
     Implemented in Phase 4."""
     pass

9. tests/unit/test_provider.py:
   - Mock httpx responses, test complete() returns ModelResponse
   - Test health_check() returns False on connection error (no raise)
   
10. tests/unit/test_kernel.py:
    - Mock router and tool_registry
    - Test run() with a trivial task completes with status=complete
    - Test run() with a task that exhausts max_steps returns status=failed

STYLE RULES (same as Phase 0, restate for clarity):
- No global mutable state
- Async all the way down — no sync I/O in async code (no time.sleep, no open())
- dataclasses over dicts for internal structures, Pydantic for I/O boundaries
- CortexError base exception, specific subclasses: CortexModelError, CortexToolError, 
  CortexPlannerError
- Spans on every external I/O operation
```

---

## PHASE 2 — Memory Architecture

**Goal:** Four-tier memory system. Working, episodic, semantic, procedural — each a distinct subsystem.
**Estimated time:** 3–4 hours
**Deliverable:** Agent can recall past conversations and retrieve semantically relevant context.

---

### Cursor Prompt — Phase 2

```
CORTEX — Phase 2: Multi-Tier Memory Architecture

Context: Phases 0 and 1 are complete. We have a working agent kernel with a 
single-turn ReAct loop. Now we add memory — the feature that separates agents 
from chatbots. Memory is not a chat history list. It is a layered retrieval 
architecture.

Design philosophy:
- Working memory = what's in the current context window (transient, no persistence)
- Episodic memory = conversation history (SQLite, chronological, session-scoped)
- Semantic memory = long-term knowledge (ChromaDB vector store, cross-session)
- Procedural memory = how to use tools effectively (patterns learned from past runs)

REQUIREMENTS:

1. cortex/memory/base.py — Abstractions
   
   class MemoryEntry(BaseModel):
     id: str
     content: str
     metadata: dict
     timestamp: datetime
     memory_type: Literal["working", "episodic", "semantic", "procedural"]
   
   class MemoryQuery(BaseModel):
     text: str
     top_k: int = 5
     memory_types: list[str] = ["semantic", "episodic"]
     session_id: str | None = None
   
   class BaseMemory(ABC):
     @abstractmethod
     async def store(self, entry: MemoryEntry) -> str: ...   # returns entry id
     
     @abstractmethod
     async def retrieve(self, query: MemoryQuery) -> list[MemoryEntry]: ...
     
     @abstractmethod
     async def delete(self, entry_id: str) -> None: ...

2. cortex/memory/working.py — WorkingMemory
   
   In-memory store, no persistence. Holds the current step's scratchpad.
   
   class WorkingMemory(BaseMemory):
     - _store: dict[str, MemoryEntry] — keyed by id
     - store(), retrieve() (filter by text similarity using simple substring, 
       no vectors needed here), clear() — clears all entries
     - Max capacity: configurable, default 50 entries. On overflow, drop oldest.

3. cortex/memory/episodic.py — EpisodicMemory
   
   Persists conversation turns to SQLite. Schema:
   
   CREATE TABLE IF NOT EXISTS episodes (
     id TEXT PRIMARY KEY,
     session_id TEXT NOT NULL,
     role TEXT NOT NULL,
     content TEXT NOT NULL,
     metadata JSON,
     created_at TEXT NOT NULL
   );
   CREATE INDEX IF NOT EXISTS idx_session ON episodes(session_id);
   
   class EpisodicMemory(BaseMemory):
     - __init__(db_path: str | Path)
     - async initialize() → creates table if not exists (called on startup)
     - store(): INSERT into episodes
     - retrieve(): SELECT WHERE session_id = ? ORDER BY created_at DESC LIMIT ?
                   Apply recency weighting: more recent entries ranked higher
     - get_session_history(session_id: str) -> list[Message]: returns ordered 
       conversation for context injection
     - async prune_old_sessions(days: int = 30): DELETE entries older than N days

4. cortex/memory/semantic.py — SemanticMemory
   
   Long-term vector store using ChromaDB. Cross-session, persistent.
   
   class SemanticMemory(BaseMemory):
     - __init__(chroma_path: str, embed_model: str, provider: OllamaProvider)
     - async initialize(): create/connect to Chroma collection named "cortex_semantic"
     - store(): embed content using OllamaProvider.embed(), upsert to Chroma
     - retrieve(): embed query, Chroma similarity search, return top_k results
       with metadata: source_session, timestamp, relevance_score
     - async consolidate(session_id: str): after a session ends, extract key 
       facts from episodic history and store them as semantic memories. 
       This is how short-term becomes long-term.
       
   The consolidate() method is the most important part. Prompt pattern:
   "Given this conversation, extract 3-5 atomic facts that would be useful 
   to remember in future sessions. Format as JSON array of strings."

5. cortex/memory/procedural.py — ProceduralMemory
   
   Stores tool-use patterns: what sequence of tools worked for what class of tasks.
   Also stored in ChromaDB, separate collection: "cortex_procedural"
   
   class ToolPattern(BaseModel):
     task_description: str
     tool_sequence: list[str]
     success: bool
     avg_steps: int
   
   class ProceduralMemory(BaseMemory):
     - store_pattern(pattern: ToolPattern) -> None
     - retrieve_patterns(task: str, top_k: int = 3) -> list[ToolPattern]
       Returns patterns for similar past tasks. Agent uses these to skip planning 
       for well-known task types.

6. cortex/memory/manager.py — MemoryManager
   
   Single interface the AgentKernel uses. Does not expose individual stores.
   
   class MemoryManager:
     - __init__(working, episodic, semantic, procedural)
     - async store_turn(session_id: str, role: str, content: str) -> None
       Stores to working + episodic
     - async retrieve_context(query: str, session_id: str) -> str
       Queries episodic + semantic, deduplicates, formats as a context block:
       "--- Relevant Memory ---\n{entries}\n---"
     - async end_session(session_id: str) -> None
       Triggers semantic.consolidate(), clears working memory
     - async store_tool_pattern(...) -> None

7. Update cortex/agent/kernel.py to use MemoryManager:
   - Before planning: retrieve_context() and inject into first message
   - After each tool execution: store_turn() with the step content
   - After final answer: end_session()

8. tests/unit/test_memory.py:
   - Test EpisodicMemory stores and retrieves correctly
   - Test SemanticMemory consolidate() extracts facts (mock Ollama embed)
   - Test MemoryManager.retrieve_context() deduplicates overlapping entries

9. tests/integration/test_memory_integration.py:
   - Spin up real ChromaDB in-memory mode
   - Store 10 entries, retrieve top 3, assert relevance ordering

IMPLEMENTATION NOTES:
- Every async db call uses aiosqlite — no sync sqlite3 in async context
- ChromaDB client: chromadb.AsyncHttpClient for production, 
  chromadb.EphemeralClient for tests
- Memory IDs: use uuid4().hex — no sequential IDs
- All timestamps in UTC, stored as ISO-8601 strings
```

---

## PHASE 3 — Tool Registry & Execution Engine

**Goal:** Auto-discovering tool registry, four built-in tools, sandboxed code execution, plugin loader.
**Estimated time:** 4–5 hours
**Deliverable:** Agent can read/write files, execute Python, search documents, fetch URLs.

---

### Cursor Prompt — Phase 3

```
CORTEX — Phase 3: Tool Registry, Built-in Tools, Sandboxed Execution

Context: We have a working agent kernel with multi-tier memory. Now we give it 
hands. The tool system must be: safe (sandboxed), discoverable (auto-registration), 
observable (every execution traced), and extensible (drop a file in plugins/).

REQUIREMENTS:

1. cortex/tools/registry.py — ToolRegistry
   
   Auto-discovers and registers tools.
   
   class ToolRegistry:
     - __init__()
     - register(tool: BaseTool) -> None: register a tool instance
     - auto_discover(plugins_dir: Path) -> None:
       Scans plugins_dir for .py files. For each file, imports it and registers 
       any class that is a subclass of BaseTool. Uses importlib.import_module.
       If a plugin fails to import, log a warning and continue — bad plugins 
       must not crash the registry.
     - get(name: str) -> BaseTool | None
     - list_tools() -> list[ToolSchema]
     - to_ollama_tools() -> list[dict]: formats all schemas for Ollama API
     - async execute(tool_name: str, **kwargs) -> ToolResult:
       Find tool, validate kwargs against schema (jsonschema.validate), execute.
       Wrap in OTel span "tool.execute" with tool_name attribute.
       On timeout (configurable, default 30s): return ToolResult(success=False, 
       error="Execution timed out")

2. cortex/tools/builtin/filesystem.py — FileSystemTool
   
   Provides read, write, list operations.
   
   Operations (exposed as sub-actions via a single tool):
   - read_file(path: str) -> str content
   - write_file(path: str, content: str) -> confirmation
   - list_directory(path: str) -> list of entries
   - file_exists(path: str) -> bool
   
   Safety constraints (enforced, not configurable):
   - Path must be under allowed_root (from config, default: current working dir)
   - Reject any path with ".." components
   - Max file read: 1MB
   - Max file write: 512KB
   - Allowed extensions for write: .txt, .md, .json, .csv, .py (configurable)
   
   ToolSchema:
     name: "filesystem"
     description: "Read, write, and list files on the local filesystem. 
                   All paths are relative to the workspace root."
     parameters: JSON Schema with action enum, path, content (optional)

3. cortex/tools/builtin/code_exec.py — CodeExecutionTool
   
   Execute Python code in a restricted sandbox.
   
   Use RestrictedPython (pip: restrictedpython) for code compilation.
   
   class CodeExecutionTool(BaseTool):
     - Compile with compile_restricted() — raises SyntaxError on policy violations
     - Execute in a restricted globals dict:
       safe_globals = {
           '__builtins__': safe_builtins,  # from RestrictedPython
           'print': captured_output_print,
           'len': len, 'range': range, 'enumerate': enumerate,
           'str': str, 'int': int, 'float': float, 'list': list, 
           'dict': dict, 'set': set, 'bool': bool,
           'json': json, 'math': math, 'datetime': datetime,
       }
     - Capture stdout via io.StringIO
     - Hard timeout: 10 seconds via asyncio.wait_for
     - Max output: 8192 chars — truncate with notice
     - Blocked: file I/O, os module, subprocess, socket, importlib
     
   Returns: stdout + return value (repr'd if not string)
   
   ToolSchema:
     name: "python_exec"
     description: "Execute Python code in a sandboxed environment. 
                   No file access, no network, no subprocess. 
                   Use for computation, data transformation, and analysis."

4. cortex/tools/builtin/web_fetch.py — WebFetchTool
   
   Fetch and parse web content. Offline-tolerant (returns clear error if no network).
   
   class WebFetchTool(BaseTool):
     - async fetch(url: str, format: Literal["text", "markdown"] = "markdown") -> str
     - Use httpx.AsyncClient with timeout=20s
     - Parse HTML with html2text (pip: html2text) for clean markdown output
     - Strip scripts, styles, nav elements before conversion
     - Max response: 50KB (truncate)
     - Respect robots.txt — check before fetching (use robotparser)
     - If network is unavailable: return ToolResult(success=False, 
       error="Network unavailable — CORTEX is running in offline mode")
     
   ToolSchema:
     name: "web_fetch"
     description: "Fetch and parse web pages as readable markdown text."

5. cortex/tools/builtin/doc_search.py — DocumentSearchTool
   
   Search through local documents using semantic similarity.
   
   class DocumentSearchTool(BaseTool):
     - __init__(semantic_memory: SemanticMemory)
     - async search(query: str, top_k: int = 5) -> list[dict]
     - async index_document(path: str, chunk_size: int = 512, 
                            overlap: int = 64) -> int (chunks indexed)
       Split document into overlapping chunks, embed each, store in SemanticMemory
       with metadata: source_path, chunk_index, total_chunks
     
   Chunking strategy: split on paragraph boundaries first, then by character 
   count if paragraph > chunk_size. Overlap ensures no context is lost at boundaries.
   
   ToolSchema:
     name: "doc_search"
     description: "Search through locally indexed documents using semantic similarity."

6. cortex/tools/plugins/ — Plugin directory
   
   Create a README.md in this directory explaining:
   - How to create a custom tool (inherit BaseTool, define schema ClassVar)
   - Example plugin: a calculator tool
   - Drop the .py file here and restart CORTEX

   Create cortex/tools/plugins/calculator.py as a reference plugin:
   - Simple arithmetic, expression evaluation using ast.literal_eval (not eval())
   - Shows the correct pattern for a minimal plugin

7. Update ToolRegistry initialization in cortex/agent/kernel.py:
   - On startup: register all builtin tools
   - Then: auto_discover(settings.plugins_dir)

8. tests/unit/test_tools.py:
   - FileSystemTool: test path traversal rejection, read/write round-trip
   - CodeExecutionTool: test basic computation, test blocked imports raise, 
     test timeout triggers
   - WebFetchTool: mock httpx, test HTML→markdown conversion
   - ToolRegistry: test auto_discover picks up plugins, bad plugin skipped

9. tests/integration/test_tool_execution.py:
   - End-to-end: ToolRegistry.execute("python_exec", code="2+2") returns "4"
   - End-to-end: ToolRegistry.execute("filesystem", action="write_file", 
     path="test.txt", content="hello") then read back

STYLE NOTES:
- FileSystemTool path validation must happen at the point of use, not at registration
- CodeExecutionTool must NEVER use exec() with user code in the main process 
  globals — always the restricted sandbox dict
- Timeouts are enforced via asyncio.wait_for, not threading.Timer
```

---

## PHASE 4 — API Layer & Streaming

**Goal:** FastAPI server with SSE streaming, WebSocket support, full REST API for all subsystems.
**Estimated time:** 3–4 hours
**Deliverable:** `POST /chat` streams agent responses token by token. Full OpenAPI spec.

---

### Cursor Prompt — Phase 4

```
CORTEX — Phase 4: FastAPI API Layer with Server-Sent Events Streaming

Context: Core agent is complete. Now we expose it cleanly over HTTP. The API 
design is as important as the implementation — it's what developers will integrate 
against and what will be showcased in the README.

Key design decision: the agent kernel is a singleton per process, instantiated 
at startup via FastAPI's lifespan. No per-request instantiation.

REQUIREMENTS:

1. cortex/api/server.py — Application Factory
   
   def create_app() -> FastAPI:
     - Create FastAPI with lifespan
     - lifespan:
       1. Initialize settings
       2. Setup tracing and metrics
       3. Configure structlog
       4. Create OllamaProvider, check health — if unhealthy, log CRITICAL warning 
          but do NOT abort startup (Ollama may start late)
       5. Create ModelRouter
       6. Create MemoryManager (init all stores)
       7. Create ToolRegistry (register builtins, discover plugins)
       8. Create AgentKernel
       9. Store all of the above in app.state
       10. On shutdown: end all open sessions, close Chroma connection
     - Include routers: chat, tasks, memory, tools, health
     - Add CORS middleware (allow all origins in dev, configurable in prod)
     - Add request ID middleware (inject X-Request-ID header)
     - Instrument with OTel FastAPI middleware

2. cortex/api/routes/chat.py — Chat endpoints
   
   POST /chat/stream
   Request:
   {
     "message": str,
     "session_id": str | null,   // null = new session
     "model_capability": "FAST" | "REASONING" | null
   }
   
   Response: text/event-stream (SSE)
   Events:
   - data: {"type": "session_id", "value": "<uuid>"}
   - data: {"type": "plan", "steps": ["step1", "step2"]}
   - data: {"type": "step_start", "step": 1, "description": "..."}
   - data: {"type": "tool_call", "tool": "...", "args": {...}}
   - data: {"type": "tool_result", "success": bool, "output": "..."}
   - data: {"type": "token", "value": "..."}   // streaming final answer
   - data: {"type": "done", "steps_taken": N}
   - data: {"type": "error", "message": "..."}
   
   Implementation:
   - Use FastAPI's StreamingResponse with media_type="text/event-stream"
   - Run agent in a separate task, communicate via asyncio.Queue
   - Agent steps send events to the queue; route reads from queue and yields
   - This requires the AgentKernel to accept an optional event_queue parameter
     on run() — if provided, publish step events as they happen
   
   POST /chat/message  (non-streaming, for clients that can't SSE)
   - Same request, returns complete AgentState as JSON when done
   
   GET /chat/sessions/{session_id}/history
   - Returns EpisodicMemory.get_session_history(session_id)

3. cortex/api/routes/tasks.py — Async task queue (fire and forget)
   
   POST /tasks
   {
     "task": str,
     "priority": "low" | "normal" | "high",
     "callback_url": str | null
   }
   Response: {"task_id": str, "status": "queued"}
   
   GET /tasks/{task_id}
   Response: AgentState (or 404 if unknown)
   
   Implementation: Use asyncio.Queue in app.state as the task queue.
   Background worker (spawned in lifespan) pulls from queue and runs agent.
   Store task results in a simple dict keyed by task_id (for this phase;
   replace with Redis in a future iteration).

4. cortex/api/routes/memory.py — Memory inspection endpoints
   
   GET /memory/search?q=<query>&types=semantic,episodic&top_k=10
   GET /memory/sessions              // list all session IDs
   DELETE /memory/sessions/{id}      // delete a session's episodic memory
   POST /memory/index                // index a document
   Body: {"path": str, "chunk_size": int = 512}

5. cortex/api/routes/tools.py — Tool introspection
   
   GET /tools                        // list all registered tools with schemas
   POST /tools/{name}/execute        // directly execute a tool (for debugging)
   Body: tool-specific kwargs

6. cortex/api/routes/health.py
   
   GET /health
   {
     "status": "healthy" | "degraded" | "unhealthy",
     "ollama": bool,
     "chromadb": bool,
     "uptime_seconds": float
   }
   
   Degraded = Ollama reachable but no models loaded.
   Unhealthy = Ollama unreachable.

7. cortex/api/middleware/telemetry.py
   - Inject request_id into structlog context for the duration of the request
   - Log request method, path, status_code, duration_ms on completion

8. Modify AgentKernel.run() to accept event_queue: asyncio.Queue | None = None
   and publish structured events at each step transition.

9. tests/integration/test_api.py:
   - Use httpx.AsyncClient with ASGI transport (no real server needed)
   - Test POST /chat/message returns 200 with valid AgentState
   - Test GET /health returns correct structure
   - Test GET /tools returns list of tool schemas
   - Test SSE stream yields expected event types in order

DESIGN NOTE:
The SSE streaming format is a deliberate design choice. Native WebSockets are 
more complex and require bidirectional framing. SSE is HTTP, cacheable, 
reconnectable, and perfectly suited to the one-way agent→client event stream. 
Include a comment in the code explaining this.
```

---

## PHASE 5 — Next.js UI

**Goal:** Clean, production-quality terminal-inspired UI for the agent. Dark theme, streaming output, tool trace sidebar.
**Estimated time:** 4–5 hours
**Deliverable:** Full chat interface with real-time streaming, memory panel, tool execution trace.

---

### Cursor Prompt — Phase 5

```
CORTEX — Phase 5: Next.js Frontend

Context: Backend is complete with SSE streaming. Build a frontend that matches 
the quality and professionalism of the backend. This is what will be in the 
GitHub README screenshots and LinkedIn posts.

Design direction: Terminal-meets-IDE. Not a chatbot bubble UI. 
Think Cursor, Warp, or Linear — focused, dense, keyboard-first.

Tech: Next.js 14 (App Router), TypeScript strict mode, Tailwind CSS, 
shadcn/ui components, Framer Motion for transitions.

Color palette: Background #0a0a0f, Surface #111118, Border #1e1e2e, 
Accent #7c3aed (violet), Text primary #e2e8f0, Text muted #64748b.
Font: JetBrains Mono for code/output, Inter for UI chrome.

REQUIREMENTS:

1. ui/src/app/layout.tsx — Root layout
   - Dark background, no white flash
   - JetBrains Mono + Inter via next/font
   - Global CSS variables for the color palette above

2. ui/src/app/page.tsx — Main shell
   Three-panel layout (like an IDE):
   - Left sidebar (240px): Session list, memory indicator, tool status
   - Center (flex-1): Chat/agent output area
   - Right panel (300px, collapsible): Execution trace (plan steps, tool calls, timings)

3. ui/src/components/ChatInput.tsx
   - Multi-line textarea that submits on Enter (shift+Enter for newline)
   - Model selector dropdown: Fast / Reasoning
   - Send button with keyboard shortcut hint (⌘↵)
   - Disabled + spinner state while agent is running

4. ui/src/components/AgentOutput.tsx
   - Renders the SSE event stream in real-time
   - Plan steps: shown as a numbered checklist, each step gets a checkmark 
     when complete
   - Tool calls: collapsible cards showing tool name, args (syntax highlighted), 
     result
   - Final answer: rendered as markdown (use react-markdown + remark-gfm + 
     rehype-highlight)
   - Typing cursor animation while streaming tokens

5. ui/src/components/ExecutionTrace.tsx (right panel)
   - Timeline view of every step in the current run
   - Each entry: step number, timestamp, duration_ms, tool name or "llm call"
   - Color-coded: tool calls in violet, LLM calls in blue, errors in red
   - Total elapsed time at the bottom
   - Collapsible — right panel can be hidden with a keyboard shortcut

6. ui/src/components/MemoryPanel.tsx (in left sidebar)
   - Shows count of semantic memories and recent episodes
   - Button: "Search memory..." opens a command palette
   - CommandPalette: searches GET /memory/search as user types (debounced 300ms)

7. ui/src/lib/streaming.ts
   - useAgentStream(message: string, sessionId: string | null) hook
   - Connects to POST /chat/stream
   - Parses SSE events, returns: { events, isStreaming, error, sessionId }
   - Handles reconnection on network drop

8. ui/src/lib/api.ts
   - Typed API client for all non-streaming endpoints
   - Uses fetch with proper error handling
   - CORTEX_API_URL from NEXT_PUBLIC_CORTEX_API_URL env var

9. ui/src/app/api/proxy/route.ts (optional)
   - Next.js route handler to proxy to the FastAPI backend
   - Avoids CORS issues in development

10. Update infra/docker-compose.yml to add:
    - ui service: node:20-alpine, port 3000
    - Build from ui/ directory

VISUAL DETAILS THAT MATTER:
- Agent "thinking" state: subtle pulsing dot next to the current step
- Tool execution: brief flash animation when a tool result arrives
- Error states: red left border on the message card, never just text color
- Session switching: smooth fade transition, not a jump
- Keyboard shortcut: Ctrl+K / Cmd+K opens command palette from anywhere
- All transitions: 150ms ease-out — fast enough to feel instant, slow enough 
  to feel polished

The UI should look like something a Principal Engineer built for their own use, 
not a hackathon demo.
```

---

## PHASE 6 — Multi-Agent Orchestration (LATS)

**Goal:** Implement LATS loop for complex tasks. Add a Supervisor agent that can spawn sub-agents.
**Estimated time:** 4–5 hours
**Deliverable:** `supervisor.run("research X and write a report")` spawns multiple sub-agents and aggregates.

---

### Cursor Prompt — Phase 6

```
CORTEX — Phase 6: Multi-Agent Orchestration with LATS

Context: We have a production-grade single agent. Now we implement the two 
advanced orchestration patterns that will define CORTEX as a serious framework:

1. LATS (Language Agent Tree Search) — for complex tasks where a single linear 
   plan fails. Explored multiple solution paths, backtrack when stuck.
2. Supervisor-Worker pattern — decompose large tasks into parallel sub-tasks, 
   each handled by an independent agent instance.

LATS Reference: Yao et al. "Language Agent Tree Search Unifies Reasoning, 
Acting, and Planning in Language Models" (arXiv:2310.04406)

REQUIREMENTS:

1. cortex/agent/loop.py — LATSLoop (full implementation)
   
   class SearchNode(BaseModel):
     id: str
     state: AgentState
     parent_id: str | None
     children: list[str]
     value: float           // estimated value of this node
     visit_count: int
     depth: int
   
   class LATSLoop:
     """
     LATS combines Monte Carlo Tree Search with LLM-based expansion.
     At each node: expand → simulate → backpropagate.
     Terminates when a solution node is found or budget is exhausted.
     """
     
     - __init__(kernel: AgentKernel, router: ModelRouter, 
                max_depth: int = 5, n_branches: int = 3, 
                simulation_budget: int = 10)
     
     - async run(task: str, session_id: str) -> AgentState:
       Main loop:
       1. Root node = initial AgentState
       2. While budget not exhausted:
          a. Select: UCB1-select most promising unexplored node
          b. Expand: LLM generates n_branches alternative next steps
          c. Simulate: run each branch to completion (or max_depth)
          d. Backpropagate: update value scores up the tree
       3. Return best terminal state
     
     - async _evaluate_state(state: AgentState) -> float:
       Ask the REASONING model to score the state 0.0–1.0 on:
       - Goal completion
       - Factual correctness
       - Conciseness
       Returns weighted average.
     
     - _ucb1_select(nodes: dict[str, SearchNode]) -> SearchNode:
       UCB1 formula: value + C * sqrt(ln(parent_visits) / node_visits)
       C = 1.414 (sqrt(2), standard exploration constant)
   
   LATSLoop is invoked by AgentKernel when settings.use_lats=true OR when 
   the Reflector detects two consecutive non-progress steps (falls back 
   from ReAct to LATS automatically).

2. cortex/agent/supervisor.py — SupervisorAgent
   
   Orchestrates multiple AgentKernel instances working in parallel.
   
   class SubTask(BaseModel):
     id: str
     description: str
     assigned_to: str    // worker_id
     result: AgentState | None = None
     status: Literal["pending", "running", "done", "failed"]
   
   class SupervisorAgent:
     - __init__(kernel_factory: Callable[[], AgentKernel], 
                max_workers: int = 3)
       kernel_factory creates a fresh AgentKernel instance (with shared 
       MemoryManager but independent working memory)
     
     - async run(task: str, session_id: str) -> SupervisorResult:
       1. Decompose task into sub-tasks using REASONING model
       2. Spawn up to max_workers worker kernels
       3. Run sub-tasks in parallel via asyncio.gather()
       4. Aggregate results: call REASONING model to synthesize sub-task 
          outputs into a coherent final answer
       5. Return SupervisorResult with all sub-states + final_answer
     
     - async _decompose(task: str) -> list[SubTask]:
       System prompt: "Decompose this task into independent sub-tasks that 
       can be worked on in parallel. Each sub-task must be self-contained. 
       Output JSON array of sub-task descriptions. Maximum 4 sub-tasks."
     
     - async _aggregate(sub_results: list[AgentState], 
                        original_task: str) -> str:
       System prompt: "You are synthesizing the results of parallel 
       research tasks. Combine the following results into a coherent, 
       well-structured answer to the original task."

3. cortex/api/routes/tasks.py — Update to support supervisor mode
   
   POST /tasks
   Add field: "orchestration": "single" | "supervisor" | "lats"
   Default: "single"
   
   When "supervisor": use SupervisorAgent instead of AgentKernel
   When "lats": use AgentKernel with use_lats=True
   When "single": current behavior

4. Add cortex.yaml settings:
   lats:
     enabled: false            # opt-in
     max_depth: 5
     n_branches: 3
     budget: 10
   supervisor:
     max_workers: 3

5. tests/unit/test_lats.py:
   - Test UCB1 selection returns highest-value unexplored node
   - Test _evaluate_state returns float in [0, 1]
   - Test run() terminates when budget exhausted

6. tests/unit/test_supervisor.py:
   - Mock kernel_factory, test sub-tasks run in parallel
   - Test aggregation called with all sub-results

IMPORTANT: The LATS implementation should have a clear docstring referencing 
the original paper. This is what will attract academic and research attention 
on GitHub. The codebase should feel like it was written by someone who read 
the paper carefully, not someone who guessed at the algorithm.
```

---

## PHASE 7 — Observability, Hardening & Release

**Goal:** Full OTel traces visible in Jaeger. Rate limiting. Graceful shutdown. README. 
**Estimated time:** 3–4 hours
**Deliverable:** Production-ready release. GitHub README with architecture diagram, badges, demo GIF.

---

### Cursor Prompt — Phase 7

```
CORTEX — Phase 7: Observability, Hardening, Documentation, and Release

This is the polishing phase. The agent works. Now we make it trustworthy, 
observable, and presentable. This phase is what separates a project that gets 
1,000 GitHub stars from one that gets 10.

REQUIREMENTS:

1. Full OpenTelemetry trace coverage audit:
   - Every external I/O: Ollama calls, ChromaDB queries, SQLite reads, HTTP fetches
   - Every agent step: planning, execution, reflection, memory retrieval
   - Span names must follow OpenTelemetry semantic conventions where applicable
   - Add span attributes: session_id, model_name, tool_name, step_count
   - Ensure traces are visible end-to-end in Jaeger for a single /chat request
   
   Test this by: running the stack, sending one message, opening Jaeger at 
   localhost:16686, confirming the full trace tree is visible.

2. Rate limiting in cortex/api/middleware/rate_limit.py:
   - Token bucket algorithm, pure Python (no Redis dependency for this)
   - Per-IP limiting: 60 requests/minute for /chat, 200/minute for others
   - Return 429 with Retry-After header when exceeded
   - Configurable via cortex.yaml: rate_limit.enabled, rate_limit.requests_per_minute

3. Graceful shutdown:
   - Handle SIGTERM and SIGINT
   - On signal: stop accepting new requests (503 on /chat), 
     wait for in-flight agent runs to complete (max 30s), 
     flush OTel spans, close DB connections
   - Log "CORTEX shutting down gracefully" at INFO level

4. Input validation hardening:
   - All API request bodies: Pydantic validation with clear error messages
   - Message length: max 8192 chars
   - session_id: validate UUID format if provided
   - tool kwargs: validated against tool schema before execution (already done 
     in Phase 3, confirm it's in place)

5. README.md — this is a first-class deliverable:

   Structure:
   ```
   # CORTEX — Private Intelligence Framework
   
   > "Intelligence that stays yours."
   
   [badges: Python 3.12, License MIT, Docker, Stars]
   
   ## What is CORTEX?
   [2 paragraphs. Lead with the problem: cloud AI is powerful but everything 
   you send is logged. CORTEX is the answer.]
   
   ## Architecture
   [The full ASCII architecture diagram from this plan]
   
   ## Features
   [Bulleted, specific. "Multi-tier memory: working → episodic → semantic → 
   procedural" not "powerful memory system"]
   
   ## Quick Start
   [4 commands: clone, make up, make pull-models, open browser]
   
   ## Configuration
   [cortex.yaml reference table]
   
   ## Tool System
   [How to write a plugin — 20 lines of code example]
   
   ## Memory Architecture
   [Diagram of the four memory tiers]
   
   ## Observability
   [Screenshot placeholder for Jaeger trace]
   
   ## Roadmap
   [ ] Voice I/O (Whisper + TTS, fully local)
   [ ] Vision tool (LLaVA integration)
   [ ] Multi-modal document processing
   [ ] Agent-to-agent networking (libp2p)
   
   ## Contributing
   [Clean, welcoming, specific about what PRs are wanted]
   ```

6. CONTRIBUTING.md:
   - Code style: ruff + mypy, no exceptions
   - Testing: every new feature needs a test
   - Tool plugins: how to contribute a new tool
   - Architecture decisions: where to find ADRs

7. Create docs/architecture-decisions/ with ADRs for:
   - ADR-001: Why not LangChain
   - ADR-002: Why ChromaDB over Qdrant/Weaviate
   - ADR-003: Why SSE over WebSockets for streaming
   - ADR-004: Why LATS over pure ReAct for complex tasks
   
   Each ADR: Context → Decision → Consequences. 1 page max.
   These are what serious engineers look for when evaluating a project.

8. GitHub Actions workflow (.github/workflows/ci.yml):
   - Trigger on push to main and PRs
   - Jobs: lint (ruff, mypy), test (pytest with --cov)
   - Upload coverage to Codecov
   - Docker build check (build only, no push)

9. Final check — run through this list before tagging v0.1.0:
   - [ ] make up && make pull-models works from a clean clone
   - [ ] POST /chat/message returns a valid response
   - [ ] SSE stream delivers events in order
   - [ ] Jaeger shows the full trace for a chat request
   - [ ] All tests pass: pytest tests/ -v
   - [ ] mypy src/ returns 0 errors
   - [ ] ruff check src/ returns 0 violations
   - [ ] README renders correctly on GitHub

LAUNCH ASSETS (create these):

10. Create .github/ISSUE_TEMPLATE/bug_report.md and feature_request.md

11. Create a DEMO.md with example conversations showing:
    - Simple Q&A
    - File write + read back
    - Code execution with computation
    - Multi-step research task
    - Memory recall across sessions

    Format these as copy-pasteable API calls with expected outputs. 
    Developers should be able to reproduce these exactly.
```

---

## Posting Strategy for Maximum Reach

### GitHub
- Tag: `ai-agent`, `llm`, `ollama`, `privacy`, `local-ai`, `autonomous-agent`, `rag`, `python`
- Pin the architecture diagram in the README header
- ADRs are your differentiator — link to them in the README introduction
- First release: v0.1.0 with a proper GitHub Release and CHANGELOG

### LinkedIn Post Template
```
I spent the last 3 weeks building CORTEX — a fully local AI agent that runs 
entirely on your hardware.

No API keys. No cloud. No data leaving your machine.

What makes it different from the 100 other "local AI" projects:

→ 4-tier memory: working → episodic → semantic → procedural
→ LATS (Language Agent Tree Search) for complex tasks
→ Hot-swappable models at runtime
→ Sandboxed code execution
→ Full OpenTelemetry observability

One command to run: docker compose up

GitHub: [link]

[architecture diagram image]
```

### X/Twitter Thread
```
Tweet 1: Built a fully private AI agent framework from scratch. 
         No cloud. No API keys. Runs entirely on your hardware. 🧵

Tweet 2: The memory architecture is what I'm most proud of. 
         [diagram]

Tweet 3: LATS loop for complex tasks — when ReAct fails, it tree-searches 
         the solution space. Reference implementation of arXiv:2310.04406.

Tweet 4: Every tool runs sandboxed. No eval(), no unrestricted exec(). 
         Drop a .py file into plugins/ and it auto-registers.

Tweet 5: One command: docker compose up
         [GitHub link]
```

---

## Execution Checklist

```
Phase 0 ── [ ] Infrastructure & config
Phase 1 ── [ ] Model provider + Agent kernel  
Phase 2 ── [ ] Memory architecture (4 tiers)
Phase 3 ── [ ] Tool registry + 4 built-in tools
Phase 4 ── [ ] FastAPI + SSE streaming
Phase 5 ── [ ] Next.js UI
Phase 6 ── [ ] LATS + Supervisor multi-agent
Phase 7 ── [ ] Observability, hardening, docs, release
```

**Total estimated build time:** 26–34 hours of focused development.

Each Cursor session = one phase. Start a fresh context for each phase and paste that phase's full prompt. The prompts are self-contained — they include enough context from prior phases that Cursor won't need the prior sessions in context.

---

*CORTEX. Intelligence that stays yours.*
