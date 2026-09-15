"""Opt-in live smoke tests against a real Ollama server.

These are deselected from the default suite and CI (see the `ollama` marker in
pyproject). Run them explicitly once Ollama is up with a small model pulled::

    ollama pull llama3.2:1b
    pytest -m ollama

Override the model/host with CORTEX_TEST_MODEL, CORTEX_TEST_EMBED_MODEL, and
CORTEX_OLLAMA_BASE_URL. Each test skips (rather than fails) when Ollama is
unreachable or the requested model is not present, so the suite stays honest.
"""

import os
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from cortex.agent.kernel import AgentKernel
from cortex.config.settings import Settings
from cortex.memory.episodic import EpisodicMemory
from cortex.memory.manager import MemoryManager
from cortex.memory.procedural import ProceduralMemory
from cortex.memory.semantic import SemanticMemory
from cortex.memory.working import WorkingMemory
from cortex.models.provider import Message, OllamaProvider
from cortex.models.router import ModelCapability, ModelRouter
from cortex.tools.builtin import register_builtin_tools
from cortex.tools.registry import ToolRegistry

pytestmark = pytest.mark.ollama

_BASE_URL = os.environ.get("CORTEX_OLLAMA_BASE_URL", "http://localhost:11434")
_MODEL = os.environ.get("CORTEX_TEST_MODEL", "llama3.2:1b")
_EMBED_MODEL = os.environ.get("CORTEX_TEST_EMBED_MODEL", "nomic-embed-text")


@pytest.fixture()
async def provider() -> AsyncIterator[OllamaProvider]:
    """A real provider, skipping the test if Ollama or the model is unavailable."""
    prov = OllamaProvider(_BASE_URL)
    if not await prov.health_check():
        await prov.aclose()
        pytest.skip(f"Ollama not reachable at {_BASE_URL}")
    models = await prov.list_models()
    if not any(name.startswith(_MODEL.split(":")[0]) for name in models):
        await prov.aclose()
        pytest.skip(f"Test model {_MODEL} not pulled (have: {models})")
    yield prov
    await prov.aclose()


def _test_settings(tmp_path: Path) -> Settings:
    return Settings(
        ollama_base_url=_BASE_URL,
        ollama_model=_MODEL,
        embed_model=_EMBED_MODEL,
        chroma_path=tmp_path / "chroma",
        db_path=tmp_path / "cortex.db",
        max_agent_steps=6,
    )


def _router(provider: OllamaProvider, settings: Settings) -> ModelRouter:
    """Route every capability at the small test model to keep live runs cheap."""
    return ModelRouter(
        provider,
        settings,
        routing={
            ModelCapability.REASONING: _MODEL,
            ModelCapability.FAST: _MODEL,
            ModelCapability.CODE: _MODEL,
            ModelCapability.EMBEDDING: _EMBED_MODEL,
        },
    )


async def _build_kernel(provider: OllamaProvider, settings: Settings) -> AgentKernel:
    episodic = EpisodicMemory(settings.db_path)
    semantic = SemanticMemory(
        settings.chroma_path,
        settings.embed_model,
        provider,
        episodic_memory=episodic,
        consolidation_model=settings.ollama_model,
    )
    procedural = ProceduralMemory(settings.chroma_path, settings.embed_model, provider)
    memory_manager = MemoryManager(WorkingMemory(), episodic, semantic, procedural)
    await memory_manager.initialize()
    router = _router(provider, settings)
    registry = ToolRegistry(settings)
    register_builtin_tools(registry, settings, semantic)
    return AgentKernel(router, registry, memory_manager, settings)


@pytest.mark.asyncio
async def test_live_complete_round_trip(provider: OllamaProvider) -> None:
    """A direct completion returns non-empty content and token counts."""
    response = await provider.complete(
        _MODEL, [Message(role="user", content="Reply with the single word: pong")]
    )
    assert response.content.strip() != ""
    assert response.output_tokens > 0


@pytest.mark.asyncio
async def test_live_streaming_yields_multiple_chunks(provider: OllamaProvider) -> None:
    """Streaming yields incremental content chunks before the done marker."""
    chunks = [
        chunk
        async for chunk in provider.stream_complete(
            _MODEL, [Message(role="user", content="Count from 1 to 10 in words.")]
        )
    ]
    with_content = [c for c in chunks if c.content]
    assert len(with_content) > 1
    assert chunks[-1].done is True


@pytest.mark.asyncio
async def test_live_kernel_tool_call_and_memory(
    provider: OllamaProvider, tmp_path: Path
) -> None:
    """A full kernel run completes and persists episodic history."""
    settings = _test_settings(tmp_path)
    kernel = await _build_kernel(provider, settings)

    state = await kernel.run("Use a tool to compute 12 * 12 and report the number.")

    assert state.final_answer
    history = await kernel._memory_manager._episodic.get_session_history(state.session_id)
    assert len(history) > 0


@pytest.mark.asyncio
async def test_live_lats_end_to_end(provider: OllamaProvider, tmp_path: Path) -> None:
    """A LATS run against the live model returns a final answer within budget."""
    settings = _test_settings(tmp_path)
    settings.lats.budget = 2
    settings.lats.max_depth = 2
    kernel = await _build_kernel(provider, settings)

    state = await kernel.run("What is the capital of France?", use_lats=True)

    assert state.final_answer is not None
