"""OpenAI-compatible model servers: LM Studio, llama.cpp's llama-server, vLLM.

They serve the OpenAI Chat Completions API (/v1/chat/completions, /v1/embeddings
and /v1/models), and so does Ollama under /v1. Only the parts Quietpost uses
are implemented.
"""

import json
import time
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urlsplit

import httpx
import structlog

from quietpost.exceptions import QuietpostModelError
from quietpost.models.provider import (
    GenerationConfig,
    Message,
    ModelResponse,
    StreamChunk,
    prefer_ipv4_loopback,
)
from quietpost.observability.metrics import record_llm_latency
from quietpost.observability.tracing import get_tracer

logger = structlog.get_logger(__name__)
_tracer = get_tracer(__name__)


class OpenAICompatibleProvider:
    """Async client for a local server that speaks the OpenAI Chat Completions API.

    ``base_url`` is the API root, such as LM Studio's http://localhost:1234/v1; a
    URL with no path gets /v1 added. ``api_key`` is sent as a bearer token, for a
    server started with one (vLLM's ``--api-key``). ``timeout`` bounds a whole
    non-streaming reply and the gap between streamed chunks.
    """

    def __init__(self, base_url: str, timeout: float = 300.0, api_key: str | None = None) -> None:
        url = prefer_ipv4_loopback(base_url.rstrip("/"))
        self._base_url = url if urlsplit(url).path else f"{url}/v1"
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(timeout, connect=10.0),
            headers={"Authorization": f"Bearer {api_key}"} if api_key else None,
            # Never through an HTTP(S)_PROXY from the environment: prompts and
            # email go to the model server directly, not via another host.
            trust_env=False,
        )

    @property
    def base_url(self) -> str:
        """The API root requests go to."""
        return self._base_url

    def _payload(
        self,
        model: str,
        messages: list[Message],
        cfg: GenerationConfig,
        tools: list[dict[str, Any]] | None,
        stream: bool,
    ) -> dict[str, Any]:
        """Build a /chat/completions request body."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": openai_messages(messages),
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "max_tokens": cfg.max_tokens,
            "stream": stream,
        }
        if cfg.stop:
            payload["stop"] = cfg.stop
        if tools:
            payload["tools"] = tools
        if cfg.json_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "reply", "schema": cfg.json_schema},
            }
        return payload

    def _describe_error(self, exc: httpx.HTTPError, model: str) -> str:
        """Turn an httpx failure into an actionable, human-readable message."""
        if isinstance(exc, httpx.ConnectError):
            return (
                f"Cannot reach the model server at {self._base_url}. Start it (LM Studio's "
                "local server, llama-server or vLLM, for example) and retry."
            )
        if isinstance(exc, httpx.TimeoutException):
            return (
                f"The model server did not respond in time for model '{model}'. It may still "
                "be loading the model: retry in a moment, or raise model_timeout_seconds."
            )
        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
            detail = _error_detail(exc.response)
            if status == 404:
                return (
                    f"The model server at {self._base_url} has no model '{model}', or this is "
                    f"not its OpenAI-compatible API (HTTP 404: {detail}). Load the model there, "
                    "or check model_server_url, which usually ends in /v1."
                )
            if status in (401, 403):
                return (
                    f"The model server at {self._base_url} refused the request (HTTP {status}). "
                    "If it was started with an API key, set model_server_api_key to that key."
                )
            return f"The model server at {self._base_url} returned HTTP {status} for model '{model}': {detail}"
        return f"Model server request failed: {exc}"

    async def complete(
        self,
        model: str,
        messages: list[Message],
        config: GenerationConfig | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelResponse:
        """Call /chat/completions and return a normalised ModelResponse."""
        cfg = config or GenerationConfig()
        payload = self._payload(model, messages, cfg, tools, stream=False)

        start = time.monotonic()
        with _tracer.start_as_current_span("openai_compat.complete") as span:
            span.set_attribute("model", model)
            span.set_attribute("model_name", model)
            try:
                response = await self._client.post("/chat/completions", json=payload)
                response.raise_for_status()
                data = response.json()
            except httpx.HTTPError as exc:
                raise QuietpostModelError(self._describe_error(exc, model)) from exc
            except ValueError as exc:
                raise QuietpostModelError(
                    f"The model server at {self._base_url} did not reply with JSON. "
                    "Check model_server_url."
                ) from exc

            latency = (time.monotonic() - start) * 1000
            usage = data.get("usage") or {}
            record_llm_latency(latency / 1000, model=model)
            span.set_attribute("latency_ms", latency)
            span.set_attribute("llm.input_tokens", usage.get("prompt_tokens") or 0)
            span.set_attribute("llm.output_tokens", usage.get("completion_tokens") or 0)

        message = _first_choice(data).get("message") or {}
        return ModelResponse(
            content=message.get("content") or "",
            model=data.get("model") or model,
            input_tokens=usage.get("prompt_tokens") or 0,
            output_tokens=usage.get("completion_tokens") or 0,
            latency_ms=latency,
            raw=data,
            tool_calls=_tool_calls(message.get("tool_calls")),
        )

    async def stream_complete(
        self,
        model: str,
        messages: list[Message],
        config: GenerationConfig | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream /chat/completions (server-sent events), yielding StreamChunk deltas.

        A tool call arrives in pieces spread over many events. The pieces are
        joined, and the final chunk carries the calls in the same shape as
        complete()'s.
        """
        cfg = config or GenerationConfig()
        payload = self._payload(model, messages, cfg, tools, stream=True)
        calls: dict[int, dict[str, str]] = {}
        usage: dict[str, Any] = {}
        served_model = model

        start = time.monotonic()
        with _tracer.start_as_current_span("openai_compat.stream_complete") as span:
            span.set_attribute("model", model)
            span.set_attribute("model_name", model)
            try:
                async with self._client.stream(
                    "POST", "/chat/completions", json=payload
                ) as response:
                    if response.is_error:
                        await response.aread()
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue  # blank lines and SSE comments or fields
                        text = line[len("data:") :].strip()
                        if text == "[DONE]":
                            break
                        try:
                            data = json.loads(text)
                        except json.JSONDecodeError:
                            logger.warning("model_stream_bad_line", line=line[:200])
                            continue
                        if data.get("error"):
                            raise QuietpostModelError(
                                f"The model server returned an error for model '{model}': "
                                f"{_error_text(data['error'])}"
                            )
                        served_model = data.get("model") or served_model
                        usage = data.get("usage") or usage
                        delta = _first_choice(data).get("delta") or {}
                        for piece in delta.get("tool_calls") or []:
                            _add_tool_call_piece(calls, piece)
                        if delta.get("content"):
                            yield StreamChunk(content=delta["content"], model=served_model)
            except httpx.HTTPError as exc:
                raise QuietpostModelError(self._describe_error(exc, model)) from exc

            latency = (time.monotonic() - start) * 1000
            record_llm_latency(latency / 1000, model=model)
            span.set_attribute("latency_ms", latency)
            span.set_attribute("llm.output_tokens", usage.get("completion_tokens") or 0)

        joined = [
            {"function": {"name": call["name"], "arguments": call["arguments"]}}
            for _, call in sorted(calls.items())
        ]
        yield StreamChunk(
            done=True,
            tool_calls=joined or None,
            input_tokens=usage.get("prompt_tokens") or 0,
            output_tokens=usage.get("completion_tokens") or 0,
            model=served_model,
        )

    async def embed(self, model: str, text: str | list[str]) -> list[list[float]]:
        """Embed one or more texts with a single /embeddings request."""
        inputs = [text] if isinstance(text, str) else text
        with _tracer.start_as_current_span("openai_compat.embed") as span:
            span.set_attribute("model", model)
            span.set_attribute("model_name", model)
            span.set_attribute("embedding.input_count", len(inputs))
            try:
                response = await self._client.post(
                    "/embeddings", json={"model": model, "input": inputs}
                )
                response.raise_for_status()
                rows = response.json().get("data") or []
            except httpx.HTTPError as exc:
                raise QuietpostModelError(self._describe_error(exc, model)) from exc
            except ValueError as exc:
                raise QuietpostModelError(
                    f"The model server at {self._base_url} did not reply with JSON."
                ) from exc
        if len(rows) != len(inputs):
            raise QuietpostModelError(
                f"The model server returned {len(rows)} embeddings for {len(inputs)} texts."
            )
        return [list(row["embedding"]) for row in sorted(rows, key=lambda row: row.get("index", 0))]

    async def list_models(self) -> list[str]:
        """Return the ids of the models the server lists."""
        try:
            response = await self._client.get("/models")
            response.raise_for_status()
            return [str(entry["id"]) for entry in response.json().get("data", []) if "id" in entry]
        except (httpx.HTTPError, ValueError) as exc:
            raise QuietpostModelError(
                f"Failed to list the models at {self._base_url}: {exc}"
            ) from exc

    async def health_check(self) -> bool:
        """Return True if the server answers its model list."""
        try:
            response = await self._client.get("/models", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False

    async def warmup(self, model: str) -> None:
        """Load a chat model ahead of the first request by asking for one token.

        The API has no load-only call. A server that loads models on demand (LM
        Studio, Ollama) loads it now; one with the model already loaded answers at once.
        """
        payload = {"model": model, "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 1}
        try:
            response = await self._client.post("/chat/completions", json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise QuietpostModelError(self._describe_error(exc, model)) from exc

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()


def openai_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Quietpost messages in the Chat Completions shape.

    That API pairs each tool reply with the call it answers by an id. Quietpost's
    messages name the tool instead (Ollama's shape), so ids are made up here: a
    tool reply answers the earliest unanswered call to the same tool. A call left
    unanswered gets a "Not run." reply, because strict servers reject a call
    without one.
    """
    converted: list[dict[str, Any]] = []
    unanswered: list[tuple[str, str]] = []  # (call id, tool name), oldest first
    count = 0

    def new_id() -> str:
        nonlocal count
        count += 1
        # Nine letters and digits: Mistral's chat template accepts no other ids.
        return f"call{count:05d}"

    def close_unanswered() -> None:
        converted.extend(
            {"role": "tool", "tool_call_id": call_id, "content": "Not run."}
            for call_id, _ in unanswered
        )
        unanswered.clear()

    for message in messages:
        if message.role == "tool":
            name = message.tool_name or ""
            call_id = next((cid for cid, tool in unanswered if tool == name), None)
            if call_id is None:
                # A reply to no call: give it one, so it still reads as tool output.
                call_id = new_id()
                converted.append(
                    {"role": "assistant", "content": "", "tool_calls": [_call(call_id, name, {})]}
                )
            else:
                unanswered.remove((call_id, name))
            converted.append({"role": "tool", "tool_call_id": call_id, "content": message.content})
            continue
        close_unanswered()
        entry: dict[str, Any] = {"role": message.role, "content": message.content}
        if message.role == "assistant" and message.tool_calls:
            entry["tool_calls"] = []
            for tool_call in message.tool_calls:
                call_id = new_id()
                unanswered.append((call_id, tool_call.name))
                entry["tool_calls"].append(_call(call_id, tool_call.name, tool_call.arguments))
        converted.append(entry)
    close_unanswered()
    return converted


def _call(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """One tool call in the Chat Completions shape (arguments as a JSON string)."""
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _first_choice(data: dict[str, Any]) -> dict[str, Any]:
    """The first entry of a reply's ``choices`` ({} when there is none)."""
    choices = data.get("choices") or [{}]
    return choices[0] if isinstance(choices[0], dict) else {}


def _tool_calls(raw: Any) -> list[dict[str, Any]] | None:
    """A reply message's tool calls as ``[{"function": {"name", "arguments"}}]``."""
    if not isinstance(raw, list):
        return None
    calls = [
        {
            "function": {
                "name": call["function"].get("name") or "",
                "arguments": call["function"].get("arguments") or "",
            }
        }
        for call in raw
        if isinstance(call, dict) and isinstance(call.get("function"), dict)
    ]
    return calls or None


def _add_tool_call_piece(calls: dict[int, dict[str, str]], piece: dict[str, Any]) -> None:
    """Merge one streamed piece of a tool call into ``calls``, keyed by the call's index.

    The first piece of a call carries its name; the arguments arrive as pieces
    of a JSON string. Ollama sends each call whole, in one piece.
    """
    index = piece.get("index")
    if not isinstance(index, int):
        # No index: a piece with an id starts a new call, any other continues the last.
        index = len(calls) if piece.get("id") or not calls else max(calls)
    call = calls.setdefault(index, {"name": "", "arguments": ""})
    function = piece.get("function") or {}
    if function.get("name") and not call["name"]:
        call["name"] = str(function["name"])
    arguments = function.get("arguments")
    if isinstance(arguments, dict):
        arguments = json.dumps(arguments)
    if arguments:
        call["arguments"] += str(arguments)


def _error_text(error: Any) -> str:
    """The message of an error value: OpenAI's {"message": ...} object, or plain text."""
    if isinstance(error, dict):
        error = error.get("message", error)
    return str(error)[:300]


def _error_detail(response: httpx.Response) -> str:
    """The error message in a failed response's body."""
    try:
        return _error_text(response.json().get("error", response.text))
    except Exception:
        try:
            return response.text[:300]
        except Exception:
            return "no detail"
