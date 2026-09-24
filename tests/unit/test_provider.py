"""Unit tests for OllamaProvider — all network calls are mocked."""

import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from cortex.exceptions import CortexModelError
from cortex.models.provider import Message, ModelResponse, OllamaProvider, ToolCall


class _FakeStreamResponse:
    """Minimal stand-in for an httpx streaming Response."""

    is_error = False

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


def _no_tools_error() -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://fake-ollama:11434/api/chat")
    response = httpx.Response(
        400,
        json={"error": "registry.ollama.ai/library/gemma3:4b does not support tools"},
        request=request,
    )
    return httpx.HTTPStatusError("400", request=request, response=response)


@pytest.mark.asyncio
async def test_model_without_tool_support_falls_back_to_no_tools(
    provider: OllamaProvider,
) -> None:
    """A 'does not support tools' 400 is retried without tools, and remembered."""
    rejected = MagicMock(spec=httpx.Response)
    rejected.raise_for_status = MagicMock(side_effect=_no_tools_error())
    accepted = MagicMock(spec=httpx.Response)
    accepted.raise_for_status = MagicMock()
    accepted.json.return_value = _make_chat_response("Hello without tools")
    post_mock = AsyncMock(side_effect=[rejected, accepted, accepted])
    tools = [{"type": "function", "function": {"name": "calculator"}}]

    with patch.object(provider._client, "post", new=post_mock):
        first = await provider.complete("gemma3:4b", [Message(role="user", content="Hi")], tools=tools)
        await provider.complete("gemma3:4b", [Message(role="user", content="Hi")], tools=tools)

    assert first.content == "Hello without tools"
    payloads = [call.kwargs["json"] for call in post_mock.await_args_list]
    assert "tools" in payloads[0]
    assert "tools" not in payloads[1]
    assert "tools" not in payloads[2]  # remembered: no second rejected attempt


class _RejectingStreamContext:
    """A streamed request Ollama rejects before sending any chunk."""

    async def __aenter__(self) -> Any:
        response = MagicMock()
        response.is_error = True
        response.aread = AsyncMock()
        response.raise_for_status = MagicMock(side_effect=_no_tools_error())
        return response

    async def __aexit__(self, *args: object) -> bool:
        return False


@pytest.mark.asyncio
async def test_streaming_model_without_tool_support_falls_back(
    provider: OllamaProvider,
) -> None:
    """The streaming path also retries without tools after a 'does not support tools' 400."""
    ok_lines = [json.dumps({"model": "gemma3:4b", "message": {"content": "hi"}, "done": True})]
    stream_mock = MagicMock(side_effect=[_RejectingStreamContext(), _FakeStreamContext(ok_lines)])
    tools = [{"type": "function", "function": {"name": "calculator"}}]

    with patch.object(provider._client, "stream", stream_mock):
        chunks = [
            chunk
            async for chunk in provider.stream_complete(
                "gemma3:4b", [Message(role="user", content="Hi")], tools=tools
            )
        ]

    assert [c.content for c in chunks] == ["hi"]
    assert "tools" not in stream_mock.call_args_list[1].kwargs["json"]


def test_localhost_base_url_uses_ipv4_loopback() -> None:
    """localhost is rewritten to 127.0.0.1 (Windows' ::1-first lookup costs ~2s/connect)."""
    assert OllamaProvider("http://localhost:11434")._base_url == "http://127.0.0.1:11434"
    assert OllamaProvider("http://ollama:11434")._base_url == "http://ollama:11434"


@pytest.mark.asyncio
async def test_payload_pins_context_window_and_keep_alive() -> None:
    """Every chat call carries num_ctx and keep_alive so Ollama never reloads mid-demo."""
    provider = OllamaProvider("http://fake-ollama:11434", num_ctx=8192, keep_alive="30m")
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = _make_chat_response("ok")
    post_mock = AsyncMock(return_value=mock_resp)

    with patch.object(provider._client, "post", new=post_mock):
        await provider.complete("qwen2.5:7b", [Message(role="user", content="Hi")])

    assert post_mock.await_args is not None
    payload = post_mock.await_args.kwargs["json"]
    assert payload["options"]["num_ctx"] == 8192
    assert payload["keep_alive"] == "30m"


def test_message_serialises_tool_calls_for_ollama() -> None:
    """Assistant tool calls and tool replies use Ollama's message shape."""
    call = Message(
        role="assistant",
        content="",
        tool_calls=[ToolCall(name="python_exec", arguments={"code": "print(1)"})],
    )
    reply = Message(role="tool", content="1", tool_name="python_exec")

    assert call.to_ollama() == {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": "python_exec", "arguments": {"code": "print(1)"}}}],
    }
    assert reply.to_ollama() == {"role": "tool", "content": "1", "tool_name": "python_exec"}


@pytest.mark.asyncio
async def test_missing_model_error_says_how_to_fix(provider: OllamaProvider) -> None:
    """A 404 from Ollama becomes an actionable 'ollama pull' message."""
    request = httpx.Request("POST", "http://fake-ollama:11434/api/chat")
    response = httpx.Response(404, json={"error": "model not found"}, request=request)
    error = httpx.HTTPStatusError("404", request=request, response=response)
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.raise_for_status = MagicMock(side_effect=error)

    with patch.object(provider._client, "post", new=AsyncMock(return_value=mock_resp)),             pytest.raises(CortexModelError, match="ollama pull qwen2.5:7b"):
        await provider.complete("qwen2.5:7b", [Message(role="user", content="Hi")])


@pytest.mark.asyncio
async def test_unreachable_ollama_error_says_how_to_fix(provider: OllamaProvider) -> None:
    """A connection failure names the URL and how to start Ollama."""
    with patch.object(
        provider._client, "post", new=AsyncMock(side_effect=httpx.ConnectError("refused"))
    ), pytest.raises(CortexModelError, match="ollama serve"):
        await provider.complete("qwen2.5:7b", [Message(role="user", content="Hi")])


@pytest.mark.asyncio
async def test_stream_error_line_raises(provider: OllamaProvider) -> None:
    """An {"error": ...} object mid-stream is raised, not silently ignored."""
    lines = [json.dumps({"error": "out of memory"})]
    with patch.object(
        provider._client, "stream", MagicMock(return_value=_FakeStreamContext(lines))
    ), pytest.raises(CortexModelError, match="out of memory"):
        async for _ in provider.stream_complete(
            model="qwen2.5:7b", messages=[Message(role="user", content="Hi")]
        ):
            pass
