"""Unit tests for OpenAICompatibleProvider: every request goes to an httpx MockTransport."""

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from quietpost.agent.executor import _parse_tool_call
from quietpost.exceptions import QuietpostModelError
from quietpost.models.openai_compat import OpenAICompatibleProvider, openai_messages
from quietpost.models.provider import GenerationConfig, Message, ToolCall

Handler = Callable[[httpx.Request], httpx.Response]
_URL = "http://lmstudio.test:1234/v1"


def _provider(handler: Handler, **kwargs: Any) -> OpenAICompatibleProvider:
    """A provider whose requests go to ``handler`` instead of the network."""
    provider = OpenAICompatibleProvider(_URL, **kwargs)
    provider._client = httpx.AsyncClient(
        base_url=provider.base_url,
        headers=provider._client.headers,
        transport=httpx.MockTransport(handler),
    )
    return provider


def _recording(response: httpx.Response, seen: list[httpx.Request]) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return response

    return handler


def _sse(*events: dict[str, Any] | str) -> httpx.Response:
    """A server-sent-events reply, one ``data:`` line per event."""
    body = "".join(
        f"data: {event if isinstance(event, str) else json.dumps(event)}\n\n" for event in events
    )
    return httpx.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"})


def _delta(**delta: Any) -> dict[str, Any]:
    return {"choices": [{"index": 0, "delta": delta}]}


@pytest.mark.asyncio
async def test_complete_sends_chat_completions_and_reads_the_reply() -> None:
    seen: list[httpx.Request] = []
    reply = {
        "model": "qwen2.5-7b-instruct",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello!"}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 3},
    }
    provider = _provider(_recording(httpx.Response(200, json=reply), seen))

    result = await provider.complete(
        "qwen2.5-7b-instruct",
        [Message(role="system", content="Be brief."), Message(role="user", content="Hi")],
        GenerationConfig(temperature=0.2, max_tokens=50, stop=["\n\n"]),
    )

    assert result.content == "Hello!"
    assert result.model == "qwen2.5-7b-instruct"
    assert (result.input_tokens, result.output_tokens) == (12, 3)
    assert result.tool_calls is None
    assert str(seen[0].url) == f"{_URL}/chat/completions"
    body = json.loads(seen[0].content)
    assert body["messages"] == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hi"},
    ]
    assert (body["temperature"], body["max_tokens"], body["stop"]) == (0.2, 50, ["\n\n"])
    assert body["stream"] is False
    assert "tools" not in body
    assert "response_format" not in body


@pytest.mark.asyncio
async def test_complete_returns_tool_calls_the_executor_can_parse() -> None:
    seen: list[httpx.Request] = []
    reply = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc",
                            "type": "function",
                            "function": {"name": "calculator", "arguments": '{"expression": "2+2"}'},
                        }
                    ],
                }
            }
        ]
    }
    provider = _provider(_recording(httpx.Response(200, json=reply), seen))
    tools = [{"type": "function", "function": {"name": "calculator", "parameters": {"type": "object"}}}]

    result = await provider.complete("m", [Message(role="user", content="2+2?")], tools=tools)

    assert result.content == ""
    assert result.tool_calls == [
        {"function": {"name": "calculator", "arguments": '{"expression": "2+2"}'}}
    ]
    assert _parse_tool_call(result.tool_calls[0]) == ("calculator", {"expression": "2+2"}, None)
    assert json.loads(seen[0].content)["tools"] == tools


@pytest.mark.asyncio
async def test_json_schema_is_sent_as_response_format() -> None:
    seen: list[httpx.Request] = []
    reply = {"choices": [{"message": {"role": "assistant", "content": '{"amount": 499}'}}]}
    provider = _provider(_recording(httpx.Response(200, json=reply), seen))
    schema = {"type": "object", "properties": {"amount": {"type": "number"}}, "required": ["amount"]}

    await provider.complete("m", [Message(role="user", content="x")], GenerationConfig(json_schema=schema))

    assert json.loads(seen[0].content)["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "reply", "schema": schema},
    }


def test_tool_replies_are_paired_with_their_calls_by_id() -> None:
    converted = openai_messages(
        [
            Message(role="user", content="Add, then read a.txt"),
            Message(
                role="assistant",
                content="",
                tool_calls=[
                    ToolCall(name="calculator", arguments={"expression": "1+1"}),
                    ToolCall(name="filesystem", arguments={"action": "read_file", "path": "a.txt"}),
                ],
            ),
            Message(role="tool", tool_name="calculator", content="2"),
            Message(role="tool", tool_name="filesystem", content="hello"),
            Message(role="assistant", content="2, and the file says hello."),
        ]
    )

    calls = converted[1]["tool_calls"]
    assert [call["function"]["name"] for call in calls] == ["calculator", "filesystem"]
    assert json.loads(calls[0]["function"]["arguments"]) == {"expression": "1+1"}
    assert converted[2] == {"role": "tool", "tool_call_id": calls[0]["id"], "content": "2"}
    assert converted[3] == {"role": "tool", "tool_call_id": calls[1]["id"], "content": "hello"}
    assert converted[4] == {"role": "assistant", "content": "2, and the file says hello."}
    # Distinct, and nine letters and digits (Mistral's chat template rejects other ids).
    assert calls[0]["id"] != calls[1]["id"]
    assert all(len(call["id"]) == 9 and call["id"].isalnum() for call in calls)


def test_tool_replies_pair_by_tool_name_then_in_order() -> None:
    converted = openai_messages(
        [
            Message(
                role="assistant",
                content="",
                tool_calls=[
                    ToolCall(name="web_fetch", arguments={"url": "https://a.example"}),
                    ToolCall(name="calculator", arguments={"expression": "3*3"}),
                    ToolCall(name="web_fetch", arguments={"url": "https://b.example"}),
                ],
            ),
            Message(role="tool", tool_name="calculator", content="9"),
            Message(role="tool", tool_name="web_fetch", content="page a"),
            Message(role="tool", tool_name="web_fetch", content="page b"),
        ]
    )

    ids = [call["id"] for call in converted[0]["tool_calls"]]
    assert [message["tool_call_id"] for message in converted[1:]] == [ids[1], ids[0], ids[2]]


def test_unanswered_calls_get_a_reply_and_stray_replies_get_a_call() -> None:
    """Strict servers reject a call without a reply, and a reply without a call."""
    converted = openai_messages(
        [
            Message(
                role="assistant",
                content="",
                tool_calls=[
                    ToolCall(name="python_exec", arguments={"code": "import os"}),
                    ToolCall(name="filesystem", arguments={"action": "list_directory"}),
                ],
            ),
            Message(role="tool", tool_name="python_exec", content="refused"),
            Message(role="user", content="ok"),
            Message(role="tool", tool_name="calculator", content="4"),
        ]
    )

    ids = [call["id"] for call in converted[0]["tool_calls"]]
    assert converted[1] == {"role": "tool", "tool_call_id": ids[0], "content": "refused"}
    assert converted[2] == {"role": "tool", "tool_call_id": ids[1], "content": "Not run."}
    assert converted[3] == {"role": "user", "content": "ok"}
    stray = converted[4]["tool_calls"][0]
    assert stray["function"]["name"] == "calculator"
    assert converted[5] == {"role": "tool", "tool_call_id": stray["id"], "content": "4"}


@pytest.mark.asyncio
async def test_stream_yields_text_then_joins_tool_call_pieces() -> None:
    response = _sse(
        {"model": "m", **_delta(role="assistant", content="Let me ")},
        _delta(content="check."),
        _delta(
            tool_calls=[
                {
                    "index": 0,
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "calculator", "arguments": ""},
                }
            ]
        ),
        _delta(tool_calls=[{"index": 0, "function": {"arguments": '{"expres'}}]),
        _delta(tool_calls=[{"index": 0, "function": {"arguments": 'sion": "6*7"}'}}]),
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
        {"choices": [], "usage": {"prompt_tokens": 30, "completion_tokens": 9}},
        "[DONE]",
    )
    provider = _provider(lambda request: response)

    chunks = [c async for c in provider.stream_complete("m", [Message(role="user", content="6*7?")])]

    assert [chunk.content for chunk in chunks[:-1]] == ["Let me ", "check."]
    final = chunks[-1]
    assert final.done
    assert final.tool_calls == [
        {"function": {"name": "calculator", "arguments": '{"expression": "6*7"}'}}
    ]
    assert (final.input_tokens, final.output_tokens) == (30, 9)


@pytest.mark.asyncio
async def test_stream_keeps_parallel_tool_calls_apart() -> None:
    response = _sse(
        _delta(tool_calls=[{"index": 0, "id": "a", "function": {"name": "web_fetch", "arguments": '{"url": '}}]),
        _delta(tool_calls=[{"index": 1, "id": "b", "function": {"name": "calculator", "arguments": "{}"}}]),
        _delta(tool_calls=[{"index": 0, "function": {"arguments": '"https://example.com"}'}}]),
        "[DONE]",
    )
    provider = _provider(lambda request: response)

    chunks = [c async for c in provider.stream_complete("m", [Message(role="user", content="x")])]

    assert chunks[-1].tool_calls == [
        {"function": {"name": "web_fetch", "arguments": '{"url": "https://example.com"}'}},
        {"function": {"name": "calculator", "arguments": "{}"}},
    ]


@pytest.mark.asyncio
async def test_stream_reads_whole_tool_calls_without_an_index() -> None:
    """Ollama's /v1 sends each call whole, in one piece."""
    response = _sse(
        _delta(
            tool_calls=[
                {"id": "x", "function": {"name": "calculator", "arguments": '{"expression": "1+1"}'}},
                {"id": "y", "function": {"name": "web_fetch", "arguments": '{"url": "https://a.example"}'}},
            ]
        ),
        "[DONE]",
    )
    provider = _provider(lambda request: response)

    chunks = [c async for c in provider.stream_complete("m", [Message(role="user", content="x")])]

    assert [call["function"]["name"] for call in chunks[-1].tool_calls or []] == [
        "calculator",
        "web_fetch",
    ]


@pytest.mark.asyncio
async def test_stream_error_event_raises() -> None:
    response = _sse({"error": {"message": "model crashed", "type": "server_error"}})
    provider = _provider(lambda request: response)

    with pytest.raises(QuietpostModelError, match="model crashed"):
        async for _ in provider.stream_complete("m", [Message(role="user", content="x")]):
            pass


@pytest.mark.asyncio
async def test_embed_sends_one_request_and_orders_by_index() -> None:
    seen: list[httpx.Request] = []
    reply = {"data": [{"index": 1, "embedding": [0.3, 0.4]}, {"index": 0, "embedding": [0.1, 0.2]}]}
    provider = _provider(_recording(httpx.Response(200, json=reply), seen))

    assert await provider.embed("embed-m", ["a", "b"]) == [[0.1, 0.2], [0.3, 0.4]]
    assert str(seen[0].url) == f"{_URL}/embeddings"
    assert json.loads(seen[0].content) == {"model": "embed-m", "input": ["a", "b"]}


@pytest.mark.asyncio
async def test_list_models_and_health_check_use_the_model_list() -> None:
    seen: list[httpx.Request] = []
    reply = {"data": [{"id": "qwen2.5-7b-instruct"}, {"id": "text-embedding-nomic-embed-text-v1.5"}]}
    provider = _provider(_recording(httpx.Response(200, json=reply), seen))

    assert await provider.list_models() == [
        "qwen2.5-7b-instruct",
        "text-embedding-nomic-embed-text-v1.5",
    ]
    assert await provider.health_check() is True
    assert {str(request.url) for request in seen} == {f"{_URL}/models"}


@pytest.mark.asyncio
async def test_health_check_is_false_when_the_server_is_down() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    assert await _provider(refuse).health_check() is False


@pytest.mark.asyncio
async def test_warmup_asks_for_a_single_token() -> None:
    seen: list[httpx.Request] = []
    reply = {"choices": [{"message": {"role": "assistant", "content": "H"}}]}
    provider = _provider(_recording(httpx.Response(200, json=reply), seen))

    await provider.warmup("qwen2.5-7b-instruct")

    body = json.loads(seen[0].content)
    assert (body["model"], body["max_tokens"]) == ("qwen2.5-7b-instruct", 1)


@pytest.mark.asyncio
async def test_api_key_is_sent_as_a_bearer_token() -> None:
    seen: list[httpx.Request] = []
    provider = _provider(_recording(httpx.Response(200, json={"data": []}), seen), api_key="test-key")

    await provider.list_models()

    assert seen[0].headers["authorization"] == "Bearer test-key"


def test_base_url_gets_the_api_path_and_ipv4_loopback() -> None:
    assert OpenAICompatibleProvider("http://localhost:1234").base_url == "http://127.0.0.1:1234/v1"
    assert OpenAICompatibleProvider("http://localhost:8080/v1/").base_url == "http://127.0.0.1:8080/v1"
    assert OpenAICompatibleProvider("http://gpu-box.lan:8000/v1").base_url == "http://gpu-box.lan:8000/v1"


def test_model_requests_ignore_proxy_environment_variables() -> None:
    assert OpenAICompatibleProvider("http://127.0.0.1:1234/v1")._client.trust_env is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "message"),
    [
        (httpx.Response(404, json={"error": {"message": "model not found"}}), "has no model 'qwen-x'"),
        (httpx.Response(401, json={"error": "unauthorized"}), "model_server_api_key"),
        (httpx.Response(500, json={"error": {"message": "out of memory"}}), "HTTP 500 .*out of memory"),
        (httpx.Response(200, text="<html>Welcome</html>"), "did not reply with JSON"),
    ],
)
async def test_errors_say_what_to_fix(response: httpx.Response, message: str) -> None:
    provider = _provider(lambda request: response)

    with pytest.raises(QuietpostModelError, match=message):
        await provider.complete("qwen-x", [Message(role="user", content="Hi")])


@pytest.mark.asyncio
async def test_unreachable_server_error_names_the_address() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(QuietpostModelError, match=f"Cannot reach the model server at {_URL}"):
        await _provider(refuse).complete("m", [Message(role="user", content="Hi")])
