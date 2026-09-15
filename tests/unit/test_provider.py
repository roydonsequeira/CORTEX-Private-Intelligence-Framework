"""Unit tests for OllamaProvider — all network calls are mocked."""

import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from cortex.exceptions import CortexModelError
from cortex.models.provider import Message, ModelResponse, OllamaProvider


class _FakeStreamResponse:
    """Minimal stand-in for an httpx streaming Response."""

    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    def raise_for_status(self) -> None:
        return None

    async def aiter_lines(self) -> AsyncIterator[str]:
        for line in self._lines:
            yield line


class _FakeStreamContext:
    """Async context manager returned by a mocked client.stream()."""

    def __init__(self, lines: list[str]) -> None:
        self._response = _FakeStreamResponse(lines)

    async def __aenter__(self) -> _FakeStreamResponse:
        return self._response

    async def __aexit__(self, *args: object) -> bool:
        return False


def _make_chat_response(content: str, model: str = "llama3.1:8b") -> dict[str, Any]:
    return {
        "model": model,
        "message": {"role": "assistant", "content": content},
        "prompt_eval_count": 10,
        "eval_count": 20,
    }


@pytest.fixture()
def provider() -> OllamaProvider:
    return OllamaProvider(base_url="http://fake-ollama:11434")


@pytest.mark.asyncio
async def test_complete_returns_model_response(provider: OllamaProvider) -> None:
    """complete() parses the Ollama response into a ModelResponse."""
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = _make_chat_response("Hello, world!")

    with patch.object(provider._client, "post", new=AsyncMock(return_value=mock_resp)):
        result = await provider.complete(
            model="llama3.1:8b",
            messages=[Message(role="user", content="Hi")],
        )

    assert isinstance(result, ModelResponse)
    assert result.content == "Hello, world!"
    assert result.model == "llama3.1:8b"
    assert result.input_tokens == 10
    assert result.output_tokens == 20
    assert result.latency_ms >= 0


@pytest.mark.asyncio
async def test_complete_raises_cortex_model_error_on_http_failure(
    provider: OllamaProvider,
) -> None:
    """complete() wraps httpx.HTTPError in CortexModelError."""
    with patch.object(
        provider._client,
        "post",
        new=AsyncMock(side_effect=httpx.ConnectError("connection refused")),
    ), pytest.raises(CortexModelError):
        await provider.complete(
            model="llama3.1:8b",
            messages=[Message(role="user", content="Hi")],
        )


@pytest.mark.asyncio
async def test_health_check_returns_true_on_200(provider: OllamaProvider) -> None:
    """health_check() returns True when the Ollama /api/tags endpoint responds 200."""
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200

    with patch.object(provider._client, "get", new=AsyncMock(return_value=mock_resp)):
        result = await provider.health_check()

    assert result is True


@pytest.mark.asyncio
async def test_health_check_returns_false_on_connection_error(
    provider: OllamaProvider,
) -> None:
    """health_check() returns False (not raises) when Ollama is unreachable."""
    with patch.object(
        provider._client,
        "get",
        new=AsyncMock(side_effect=httpx.ConnectError("refused")),
    ):
        result = await provider.health_check()

    assert result is False


@pytest.mark.asyncio
async def test_stream_complete_yields_deltas_then_done(provider: OllamaProvider) -> None:
    """stream_complete() yields a StreamChunk per line and marks the final chunk done."""
    lines = [
        json.dumps({"model": "llama3.1:8b", "message": {"content": "Hel"}, "done": False}),
        json.dumps({"model": "llama3.1:8b", "message": {"content": "lo"}, "done": False}),
        "",  # blank keep-alive line is skipped
        json.dumps(
            {
                "model": "llama3.1:8b",
                "message": {"content": ""},
                "done": True,
                "prompt_eval_count": 5,
                "eval_count": 2,
            }
        ),
    ]
    with patch.object(
        provider._client, "stream", MagicMock(return_value=_FakeStreamContext(lines))
    ):
        chunks = [
            chunk
            async for chunk in provider.stream_complete(
                model="llama3.1:8b",
                messages=[Message(role="user", content="Hi")],
            )
        ]

    assert [c.content for c in chunks] == ["Hel", "lo", ""]
    assert chunks[-1].done is True
    assert chunks[-1].output_tokens == 2
    assert "".join(c.content for c in chunks) == "Hello"


@pytest.mark.asyncio
async def test_stream_complete_surfaces_tool_calls(provider: OllamaProvider) -> None:
    """stream_complete() surfaces tool_calls emitted on the stream."""
    tool_call = {"function": {"name": "calculator", "arguments": {"expression": "2+2"}}}
    lines = [
        json.dumps(
            {
                "model": "llama3.1:8b",
                "message": {"content": "", "tool_calls": [tool_call]},
                "done": True,
            }
        ),
    ]
    with patch.object(
        provider._client, "stream", MagicMock(return_value=_FakeStreamContext(lines))
    ):
        chunks = [
            chunk
            async for chunk in provider.stream_complete(
                model="llama3.1:8b",
                messages=[Message(role="user", content="2+2?")],
            )
        ]

    assert chunks[-1].tool_calls == [tool_call]


@pytest.mark.asyncio
async def test_embed_uses_batch_endpoint(provider: OllamaProvider) -> None:
    """embed() sends all inputs in one /api/embed request when available."""
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"embeddings": [[0.1, 0.2], [0.3, 0.4]]}

    post_mock = AsyncMock(return_value=mock_resp)
    with patch.object(provider._client, "post", new=post_mock):
        result = await provider.embed(model="nomic-embed-text", text=["a", "b"])

    assert result == [[0.1, 0.2], [0.3, 0.4]]
    assert post_mock.await_count == 1  # one batched request, not one per input
    assert post_mock.await_args is not None
    assert post_mock.await_args.args[0] == "/api/embed"


@pytest.mark.asyncio
async def test_embed_returns_list_of_vectors(provider: OllamaProvider) -> None:
    """embed() returns a list of float vectors, one per input text."""
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"embedding": [0.1, 0.2, 0.3]}

    with patch.object(provider._client, "post", new=AsyncMock(return_value=mock_resp)):
        result = await provider.embed(model="nomic-embed-text", text="hello")

    assert result == [[0.1, 0.2, 0.3]]


@pytest.mark.asyncio
async def test_list_models_returns_names(provider: OllamaProvider) -> None:
    """list_models() extracts model names from the Ollama /api/tags response."""
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "models": [{"name": "llama3.1:8b"}, {"name": "nomic-embed-text"}]
    }

    with patch.object(provider._client, "get", new=AsyncMock(return_value=mock_resp)):
        names = await provider.list_models()

    assert names == ["llama3.1:8b", "nomic-embed-text"]
