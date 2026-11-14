"""Opt-in live tests of OpenAICompatibleProvider against a real model server.

Deselected from the default suite and CI, like test_live_ollama.py (the `ollama`
marker). With no other server running they use Ollama's own OpenAI-compatible
API::

    pytest -m ollama tests/live/test_live_openai_compat.py

Point them at LM Studio, llama-server or vLLM with QUIETPOST_TEST_OPENAI_URL, and
choose models with QUIETPOST_TEST_MODEL and QUIETPOST_TEST_EMBED_MODEL. Each test
skips (rather than fails) when the server or a model is not there.
"""

import json
import os
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from quietpost.agent.executor import _parse_tool_call
from quietpost.agent.kernel import AgentKernel
from quietpost.config.settings import Settings
from quietpost.memory.episodic import EpisodicMemory
from quietpost.memory.manager import MemoryManager
from quietpost.memory.procedural import ProceduralMemory
from quietpost.memory.semantic import SemanticMemory
from quietpost.memory.working import WorkingMemory
from quietpost.models.openai_compat import OpenAICompatibleProvider
from quietpost.models.provider import GenerationConfig, Message, ToolCall
from quietpost.models.router import ModelRouter
from quietpost.tools.builtin import register_builtin_tools
from quietpost.tools.registry import ToolRegistry

pytestmark = pytest.mark.ollama

_BASE_URL = os.environ.get("QUIETPOST_TEST_OPENAI_URL", "http://localhost:11434/v1")
_MODEL = os.environ.get("QUIETPOST_TEST_MODEL", "llama3.2:1b")
_EMBED_MODEL = os.environ.get("QUIETPOST_TEST_EMBED_MODEL", "nomic-embed-text")
_CALCULATOR = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression.",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
}


@pytest.fixture()
async def provider() -> AsyncIterator[OpenAICompatibleProvider]:
    """A real provider, skipping the test if the server or the model is unavailable."""
    prov = OpenAICompatibleProvider(_BASE_URL)
    if not await prov.health_check():
        await prov.aclose()
        pytest.skip(f"No OpenAI-compatible server at {_BASE_URL}")
    models = await prov.list_models()
    if not any(name.startswith(_MODEL.split(":")[0]) for name in models):
        await prov.aclose()
        pytest.skip(f"Test model {_MODEL} not on the server (have: {models})")
    yield prov
    await prov.aclose()


@pytest.mark.asyncio
async def test_live_complete_round_trip(provider: OpenAICompatibleProvider) -> None:
    response = await provider.complete(
        _MODEL, [Message(role="user", content="Reply with the single word: pong")]
    )
    assert response.content.strip() != ""
    assert response.output_tokens > 0


@pytest.mark.asyncio
async def test_live_streaming_yields_multiple_chunks(provider: OpenAICompatibleProvider) -> None:
    chunks = [
        chunk
        async for chunk in provider.stream_complete(
            _MODEL, [Message(role="user", content="Count from 1 to 10 in words.")]
        )
    ]
    assert len([c for c in chunks if c.content]) > 1
    assert chunks[-1].done is True


@pytest.mark.asyncio
async def test_live_tool_call_round_trip(provider: OpenAICompatibleProvider) -> None:
    """The call comes back parseable, and the reply to it is accepted on the next turn."""
    messages = [Message(role="user", content="Use the calculator tool to compute 12 * 12.")]
    first = await provider.complete(_MODEL, messages, tools=[_CALCULATOR])
    assert first.tool_calls, f"no tool call; the model said: {first.content!r}"
    name, arguments, error = _parse_tool_call(first.tool_calls[0])
    assert (name, error) == ("calculator", None)

    messages += [
        Message(
            role="assistant",
            content=first.content,
            tool_calls=[ToolCall(name=name, arguments=arguments)],
        ),
        Message(role="tool", tool_name=name, content="144"),
    ]
    second = await provider.complete(_MODEL, messages, tools=[_CALCULATOR])
    assert "144" in second.content


@pytest.mark.asyncio
async def test_live_structured_reply_matches_the_schema(provider: OpenAICompatibleProvider) -> None:
    schema = {
        "type": "object",
        "properties": {"merchant": {"type": "string"}, "amount": {"type": "number"}},
        "required": ["merchant", "amount"],
    }
    response = await provider.complete(
        _MODEL,
        [Message(role="user", content="Receipt: you paid 499.00 to Netflix. Extract it.")],
        GenerationConfig(temperature=0, json_schema=schema),
    )
    data = json.loads(response.content)
    assert set(data) >= {"merchant", "amount"}
    assert isinstance(data["amount"], int | float)


@pytest.mark.asyncio
async def test_live_embeddings(provider: OpenAICompatibleProvider) -> None:
    vectors = await provider.embed(_EMBED_MODEL, ["a receipt from Netflix", "a note from Ada"])
    assert len(vectors) == 2
    assert len(vectors[0]) == len(vectors[1]) > 0


@pytest.mark.asyncio
async def test_live_kernel_run_with_tools(
    provider: OpenAICompatibleProvider, tmp_path: Path
) -> None:
    """A whole agent run, tool calls included, through the OpenAI-compatible API."""
    settings = Settings(
        model_server="openai-compatible",
        model_server_url=_BASE_URL,
        chat_model=_MODEL,
        embed_model=_EMBED_MODEL,
        chroma_path=tmp_path / "chroma",
        db_path=tmp_path / "quietpost.db",
        allowed_root=tmp_path / "workspace",
        max_agent_steps=6,
    )
    episodic = EpisodicMemory(settings.db_path)
    semantic = SemanticMemory(
        settings.chroma_path,
        settings.embed_model,
        provider,
        episodic_memory=episodic,
        consolidation_model=settings.chat_model,
    )
    procedural = ProceduralMemory(settings.chroma_path, settings.embed_model, provider)
    memory = MemoryManager(WorkingMemory(), episodic, semantic, procedural)
    await memory.initialize()
    registry = ToolRegistry(settings)
    register_builtin_tools(registry, settings, semantic)
    kernel = AgentKernel(ModelRouter(provider, settings), registry, memory, settings)

    state = await kernel.run("Use python to compute 12 * 12 and tell me the number.")

    assert state.final_answer and "144" in state.final_answer
    await memory.drain()
