"""Unit tests for OllamaProvider — all network calls are mocked."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from cortex.exceptions import CortexModelError
from cortex.models.provider import Message, ModelResponse, OllamaProvider


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
