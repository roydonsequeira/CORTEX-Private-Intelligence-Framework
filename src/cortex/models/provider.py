"""Ollama model provider — async HTTP client wrapping the Ollama REST API."""

import json
import time
from collections.abc import AsyncIterator
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

import httpx
import structlog
from pydantic import BaseModel

from cortex.exceptions import CortexModelError
from cortex.observability.metrics import record_llm_latency
from cortex.observability.tracing import get_tracer

logger = structlog.get_logger(__name__)
_tracer = get_tracer(__name__)


class ToolCall(BaseModel):
    """A single tool invocation requested by the model."""

    name: str
    arguments: dict[str, Any]


class ToolResult(BaseModel):
    """Result from executing a tool call."""

    tool_name: str
    success: bool
    output: str
    error: str | None = None
    execution_time_ms: float


class Message(BaseModel):
    """A single message in the conversation history.

    An assistant message that requested tools carries ``tool_calls``; the tool
    replies that follow carry ``tool_name``. Both are sent to Ollama so the model
    sees its own tool calls paired with their results — without that pairing a
    small model re-issues the same call because it never "remembers" making it.
    """

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_calls: list[ToolCall] | None = None
    tool_results: list[ToolResult] | None = None
    tool_name: str | None = None

    def to_ollama(self) -> dict[str, Any]:
        """Serialise to the Ollama /api/chat message shape."""
        payload: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            payload["tool_calls"] = [
                {"function": {"name": call.name, "arguments": call.arguments}}
                for call in self.tool_calls
            ]
        if self.tool_name:
            payload["tool_name"] = self.tool_name
        return payload


class GenerationConfig(BaseModel):
    """Controls sampling behaviour for a generation call."""

    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 4096
    stop: list[str] | None = None


class ModelResponse(BaseModel):
    """Normalised response from any model backend."""

    content: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    raw: dict[str, Any]


class StreamChunk(BaseModel):
    """One incremental chunk from a streaming generation call.

    ``content`` is the delta for this chunk. The terminal chunk (``done=True``)
    also carries any ``tool_calls`` the model emitted and the final token counts,
    so a streaming caller can both render tokens live and still detect tool use.
    """

    content: str = ""
    done: bool = False
    tool_calls: list[dict[str, Any]] | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""


class OllamaProvider:
    """Async client for the Ollama REST API. Reuses a single httpx.AsyncClient.

    ``timeout`` bounds a whole non-streaming generation (and the gap between
    streamed chunks), so it must cover a cold model load plus generation on
    modest hardware. ``num_ctx`` pins the context window for every call — Ollama
    reloads the model whenever it changes, and its small default silently
    truncates long tool transcripts. ``keep_alive`` keeps the model resident
    between requests so a pause in a demo does not trigger a cold reload.
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 300.0,
        num_ctx: int | None = None,
        keep_alive: str | None = None,
    ) -> None:
        self._base_url = _prefer_ipv4_loopback(base_url.rstrip("/"))
        self._num_ctx = num_ctx
        self._keep_alive = keep_alive
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(timeout, connect=10.0),
        )

    def _payload(
        self,
        model: str,
        messages: list[Message],
        cfg: GenerationConfig,
        tools: list[dict[str, Any]] | None,
        stream: bool,
    ) -> dict[str, Any]:
        """Build an /api/chat request body."""
        options: dict[str, Any] = {
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "num_predict": cfg.max_tokens,
        }
        if self._num_ctx:
            options["num_ctx"] = self._num_ctx
        if cfg.stop:
            options["stop"] = cfg.stop
        payload: dict[str, Any] = {
            "model": model,
            "messages": [m.to_ollama() for m in messages],
            "stream": stream,
            "options": options,
        }
        if self._keep_alive:
            payload["keep_alive"] = self._keep_alive
        if tools:
            payload["tools"] = tools
        return payload

    def _describe_error(self, exc: httpx.HTTPError, model: str) -> str:
        """Turn an httpx failure into an actionable, human-readable message."""
        if isinstance(exc, httpx.ConnectError):
            return (
                f"Cannot reach Ollama at {self._base_url}. "
                "Start it with `ollama serve` (or open the Ollama app) and retry."
            )
        if isinstance(exc, httpx.TimeoutException):
            return (
                f"Ollama did not respond in time for model '{model}'. The model may "
                "still be loading — retry in a moment, or raise ollama_timeout_seconds."
            )
        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
            detail = _error_detail(exc.response)
            if status == 404:
                return f"Model '{model}' is not available in Ollama. Run: ollama pull {model}"
            return f"Ollama returned HTTP {status} for model '{model}': {detail}"
        return f"Ollama request failed: {exc}"

    async def complete(
        self,
        model: str,
        messages: list[Message],
        config: GenerationConfig | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelResponse:
        """Call the Ollama /api/chat endpoint and return a normalised ModelResponse."""
        cfg = config or GenerationConfig()
        payload = self._payload(model, messages, cfg, tools, stream=False)

        start = time.monotonic()
        with _tracer.start_as_current_span("ollama.complete") as span:
            span.set_attribute("model", model)
            span.set_attribute("model_name", model)
            try:
                response = await self._client.post("/api/chat", json=payload)
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise CortexModelError(self._describe_error(exc, model)) from exc

            latency = (time.monotonic() - start) * 1000
            data = response.json()
            record_llm_latency(latency / 1000, model=model)
            span.set_attribute("latency_ms", latency)
            span.set_attribute("llm.input_tokens", data.get("prompt_eval_count", 0))
            span.set_attribute("llm.output_tokens", data.get("eval_count", 0))

        msg = data.get("message", {})
        return ModelResponse(
            content=msg.get("content", ""),
            model=data.get("model", model),
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
            latency_ms=latency,
            raw=data,
        )

    async def stream_complete(
        self,
        model: str,
        messages: list[Message],
        config: GenerationConfig | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream the Ollama /api/chat endpoint, yielding StreamChunk deltas.

        Sets ``stream: true`` so tokens are yielded as the model generates them
        rather than buffered. Tool calls, when the model emits them, arrive on the
        chunks and are surfaced verbatim so the caller can still route tool use.
        """
        cfg = config or GenerationConfig()
        payload = self._payload(model, messages, cfg, tools, stream=True)

        start = time.monotonic()
        with _tracer.start_as_current_span("ollama.stream_complete") as span:
            span.set_attribute("model", model)
            span.set_attribute("model_name", model)
            try:
                async with self._client.stream(
                    "POST", "/api/chat", json=payload
                ) as response:
                    if response.is_error:
                        await response.aread()
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            logger.warning("ollama_stream_bad_line", line=line[:200])
                            continue
                        if data.get("error"):
                            raise CortexModelError(
                                f"Ollama error from model '{model}': {data['error']}"
                            )
                        message = data.get("message", {})
                        done = bool(data.get("done"))
                        if done:
                            latency = (time.monotonic() - start) * 1000
                            record_llm_latency(latency / 1000, model=model)
                            span.set_attribute("latency_ms", latency)
                            span.set_attribute(
                                "llm.output_tokens", data.get("eval_count", 0)
                            )
                        yield StreamChunk(
                            content=message.get("content", ""),
                            done=done,
                            tool_calls=message.get("tool_calls"),
                            input_tokens=data.get("prompt_eval_count", 0),
                            output_tokens=data.get("eval_count", 0),
                            model=data.get("model", model),
                        )
            except httpx.HTTPError as exc:
                raise CortexModelError(self._describe_error(exc, model)) from exc

    async def embed(self, model: str, text: str | list[str]) -> list[list[float]]:
        """Embed one or more texts using the specified model.

        Sends the whole batch to Ollama's ``/api/embed`` (plural) endpoint in a
        single request; falls back to the per-item ``/api/embeddings`` loop when
        the batch endpoint is unavailable on this Ollama version.
        """
        inputs = [text] if isinstance(text, str) else text
        with _tracer.start_as_current_span("ollama.embed") as span:
            span.set_attribute("model", model)
            span.set_attribute("model_name", model)
            span.set_attribute("embedding.input_count", len(inputs))
            try:
                response = await self._client.post(
                    "/api/embed", json={"model": model, "input": inputs}
                )
                response.raise_for_status()
                embeddings = response.json().get("embeddings")
                if embeddings is not None:
                    return [list(vector) for vector in embeddings]
            except httpx.HTTPError:
                pass  # Older Ollama without /api/embed: fall back to per-item.
            return await self._embed_per_item(model, inputs)

    async def _embed_per_item(
        self, model: str, inputs: list[str]
    ) -> list[list[float]]:
        """Embed each text with a separate /api/embeddings request (legacy path)."""
        embeddings: list[list[float]] = []
        for chunk in inputs:
            try:
                response = await self._client.post(
                    "/api/embeddings", json={"model": model, "prompt": chunk}
                )
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise CortexModelError(self._describe_error(exc, model)) from exc
            embeddings.append(response.json()["embedding"])
        return embeddings

    async def list_models(self) -> list[str]:
        """Return names of all models available in this Ollama instance."""
        try:
            response = await self._client.get("/api/tags")
            response.raise_for_status()
            return [m["name"] for m in response.json().get("models", [])]
        except httpx.HTTPError as exc:
            raise CortexModelError(f"Failed to list Ollama models: {exc}") from exc

    async def health_check(self) -> bool:
        """Return True if Ollama is reachable and responsive."""
        try:
            response = await self._client.get("/api/tags", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False

    async def warmup(self, model: str) -> None:
        """Load a chat model into memory ahead of the first request.

        An empty-prompt /api/generate loads the weights without generating. The
        same ``num_ctx`` as real calls is sent so the first chat does not reload.
        """
        payload: dict[str, Any] = {"model": model}
        if self._num_ctx:
            payload["options"] = {"num_ctx": self._num_ctx}
        if self._keep_alive:
            payload["keep_alive"] = self._keep_alive
        try:
            response = await self._client.post("/api/generate", json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise CortexModelError(self._describe_error(exc, model)) from exc

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()


def _prefer_ipv4_loopback(url: str) -> str:
    """Rewrite a ``localhost`` Ollama URL to ``127.0.0.1``.

    Ollama binds IPv4 loopback only. On Windows, resolving ``localhost`` tries
    ``::1`` first and the refused IPv6 connect costs ~2 s before falling back,
    on every new connection (measured: 2062 ms vs 23 ms).
    """
    parts = urlsplit(url)
    if parts.hostname != "localhost":
        return url
    netloc = "127.0.0.1" + (f":{parts.port}" if parts.port else "")
    return urlunsplit(parts._replace(netloc=netloc))


def _error_detail(response: httpx.Response) -> str:
    """Extract Ollama's ``{"error": ...}`` message from a failed response."""
    try:
        return str(response.json().get("error", response.text))[:300]
    except Exception:
        try:
            return response.text[:300]
        except Exception:
            return "no detail"
