# Contributing to CORTEX

Thanks for helping improve CORTEX. This project aims to be a reference implementation for local, private AI agents, so contributions should be typed, tested, observable, and easy to review.

## Development Setup

```bash
git clone https://github.com/roydonsequeira/CORTEX-Private-Intelligence-Framework.git
cd CORTEX-Private-Intelligence-Framework
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pip install pre-commit && pre-commit install   # optional: ruff and mypy on commit

ollama pull qwen2.5:7b && ollama pull nomic-embed-text
cortex doctor                        # verify the setup
cortex serve                         # API on http://localhost:8000
cd ui && npm install && npm run dev  # UI on http://localhost:3000
```

Unit and integration tests mock the model, so they run without Ollama. Live
tests against a real model are opt-in: `pytest -m ollama`.

## Code Style

- Python code must pass `ruff` and `mypy` with no exceptions.
- TypeScript code must pass `npm run lint`, `npm run typecheck`, and `npm run build`.
- Keep modules focused. Prefer existing local abstractions over new framework layers.
- Avoid cloud dependencies and API-key assumptions unless the feature is explicitly optional.

## Tests

Every new feature needs tests:

- Unit tests for pure logic, routing, and validation.
- Integration tests for API behavior, tool execution, memory retrieval, and streaming contracts.
- Regression tests for bugs.

Run before opening a PR:

```bash
python -m pytest tests/ -v
python -m ruff check src tests
python -m mypy src tests
cd ui && npm run lint && npm run typecheck && npm run build
```

## Tool Plugins

Tool plugins live under `src/cortex/tools/plugins/`. A plugin should:

- Inherit from `BaseTool`.
- Define a `ToolSchema` with a JSON Schema `parameters` object.
- Return `ToolResult` from `execute()`.
- Avoid unsafe file, network, subprocess, or eval behavior unless intentionally sandboxed.

See `src/cortex/tools/plugins/README.md` and `calculator.py` for examples.

## Architecture Decisions

Architecture Decision Records live in `docs/architecture-decisions/`.

Before proposing major changes, read:

- ADR-001: Why not LangChain
- ADR-002: Why ChromaDB
- ADR-003: Why SSE
- ADR-004: Why LATS

New architectural changes should include a short ADR using Context, Decision, and Consequences.

## Pull Requests

Good PRs are small and reviewable:

- One feature or fix per PR.
- Include tests.
- Update docs if behavior or setup changes.
- Keep generated/runtime data out of the diff.
