# CORTEX — Private Intelligence Framework

> "Intelligence that stays yours."

![Python](https://img.shields.io/badge/Python-3.12-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Docker](https://img.shields.io/badge/Docker-Compose-blue)

## What is CORTEX?

Cloud AI is powerful — but everything you send is logged, retained, and used to train future models. Every query, every document, every private thought you share with a cloud assistant leaves a trace you do not control.

CORTEX is the answer. A fully local, privacy-first autonomous agent framework that runs entirely on your own hardware. No API keys. No cloud calls. No data ever leaving your machine. Built with production-grade Python, not a Jupyter notebook or a LangChain wrapper.

## Architecture

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
│                        │  ┌──────────┐  ┌─────────┐  │                      │
│                        │  │  Planner │  │Reflector│  │                      │
│                        │  └────┬─────┘  └────┬────┘  │                      │
│                        │  ┌────▼──────────────▼────┐  │                      │
│                        │  │    LATS / ReAct Loop   │  │                      │
│                        │  └────────────┬───────────┘  │                      │
│                        └──────────────┼──────────────┘                      │
│           ┌───────────────────────────┼───────────────────────────┐         │
│   ┌───────▼──────┐         ┌──────────▼─────────┐      ┌─────────▼──────┐  │
│   │ Memory Stack │         │   Tool Registry     │      │  Model Router  │  │
│   │ • Working    │         │ • Filesystem        │      │ • Ollama       │  │
│   │ • Episodic   │         │ • Code Exec (jail)  │      │ • llama3.1:8b  │  │
│   │ • Semantic   │         │ • Web Scraper       │      │ • qwen2.5:14b  │  │
│   │ • Procedural │         │ • Doc Search        │      │ • deepseek-r1  │  │
│   └──────────────┘         │ • Plugin Loader     │      └────────────────┘  │
│   ┌──────────────┐         └─────────────────────┘                          │
│   │  ChromaDB    │                                                            │
│   │  (local)     │                                                            │
│   └──────────────┘                                                            │
│   ┌───────────────────────────────────────────────────────────────────────┐  │
│   │   OpenTelemetry Collector → Jaeger UI (Port 16686)                    │  │
│   └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Features

- **Multi-tier memory**: working → episodic (SQLite) → semantic (ChromaDB) → procedural
- **LATS-augmented ReAct loop**: Language Agent Tree Search for non-trivial tasks
- **Hot-swappable models**: swap Ollama models at runtime, no restart
- **Sandboxed code execution**: RestrictedPython jail — no unrestricted `exec()`
- **Plugin-first tool registry**: drop a `.py` file in `tools/plugins/`, it auto-registers
- **Full OpenTelemetry observability**: traces visible in Jaeger from day one
- **Production-grade Python**: typed, tested, structured — not a script, not a notebook
- **Docker Compose stack**: one command to spin up the entire infrastructure

## Quick Start

```bash
git clone https://github.com/yourusername/CORTEX-Private-Intelligence-Framework.git
cd CORTEX-Private-Intelligence-Framework
make up
make pull-models
# Open http://localhost:8000/docs for the API, http://localhost:16686 for traces
```

## Stack

| Component | Choice |
|---|---|
| LLM Runtime | Ollama |
| Primary Model | llama3.1:8b / qwen2.5:14b |
| Reasoning Model | deepseek-r1:8b |
| Embeddings | nomic-embed-text |
| Vector Store | ChromaDB |
| Session DB | SQLite (aiosqlite) |
| Backend | FastAPI + uvicorn |
| Observability | OpenTelemetry + Jaeger |
| Frontend | Next.js 14 (Phase 5) |

## Configuration

All configuration lives in `cortex.yaml`:

| Key | Default | Description |
|---|---|---|
| `ollama_base_url` | `http://localhost:11434` | Ollama server URL |
| `ollama_model` | `llama3.1:8b` | Default FAST model |
| `embed_model` | `nomic-embed-text` | Embedding model |
| `chroma_path` | `./.cortex/chroma` | ChromaDB storage path |
| `db_path` | `./.cortex/cortex.db` | SQLite session DB path |
| `api_port` | `8000` | FastAPI port |
| `log_level` | `INFO` | Logging level |
| `otel_endpoint` | `http://localhost:4317` | OTLP gRPC endpoint |
| `max_agent_steps` | `20` | Max ReAct loop iterations |

Override any value with an env var: `CORTEX_API_PORT=9000`.

## Writing a Plugin Tool

Drop a `.py` file in `src/cortex/tools/plugins/` and restart CORTEX:

```python
from cortex.tools.base import BaseTool, ToolResult, ToolSchema
from typing import ClassVar

class MyTool(BaseTool):
    schema: ClassVar[ToolSchema] = ToolSchema(
        name="my_tool",
        description="Does something useful.",
        parameters={
            "type": "object",
            "properties": {"input": {"type": "string"}},
            "required": ["input"],
        },
    )

    async def execute(self, input: str, **kwargs) -> ToolResult:
        return ToolResult(
            tool_name="my_tool",
            success=True,
            output=f"Processed: {input}",
            execution_time_ms=1.0,
        )
```

## Implementation Status

| Phase | Description | Status |
|---|---|---|
| Phase 0 | Foundation, config, Docker, telemetry | ✅ Complete |
| Phase 1 | Model provider, agent kernel, ReAct loop | ✅ Complete |
| Phase 2 | Multi-tier memory architecture | 🔜 Planned |
| Phase 3 | Tool registry + built-in tools | 🔜 Planned |
| Phase 4 | FastAPI + SSE streaming | 🔜 Planned |
| Phase 5 | Next.js UI | 🔜 Planned |
| Phase 6 | LATS + multi-agent supervisor | 🔜 Planned |
| Phase 7 | Observability, hardening, release | 🔜 Planned |

## Roadmap

- [ ] Voice I/O (Whisper + TTS, fully local)
- [ ] Vision tool (LLaVA integration)
- [ ] Multi-modal document processing
- [ ] Agent-to-agent networking (libp2p)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). PRs welcome for new tool plugins, memory backends, and model provider adapters.

## License

MIT
