"""Ollama model provider — async HTTP client wrapping the Ollama REST API."""

import time
from typing import Any, Literal

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
    """A single message in the conversation history."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_calls: list[ToolCall] | None = None
    tool_results: list[ToolResult] | None = None


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


class OllamaProvider:
    """Async client for the Ollama REST API. Reuses a single httpx.AsyncClient."""

    def __init__(self, base_url: str, timeout: float = 120.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=timeout)

    async def complete(
        self,
        model: str,
        messages: list[Message],
        config: GenerationConfig | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelResponse:
        """Call the Ollama /api/chat endpoint and return a normalised ModelResponse."""
        cfg = config or GenerationConfig()
        payload: dict[str, Any] = {
            "model": model,
            "messages": [m.model_dump(exclude_none=True) for m in messages],
            "stream": False,
            "options": {
                "temperature": cfg.temperature,
                "top_p": cfg.top_p,
                "num_predict": cfg.max_tokens,
            },
        }
        if cfg.stop:
            payload["options"]["stop"] = cfg.stop
        if tools:
            payload["tools"] = tools

        start = time.monotonic()
        with _tracer.start_as_current_span("ollama.complete") as span:
            span.set_attribute("model", model)
            span.set_attribute("model_name", model)
            try:
                response = await self._client.post("/api/chat", json=payload)
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise CortexModelError(f"Ollama request failed: {exc}") from exc

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

    async def embed(self, model: str, text: str | list[str]) -> list[list[float]]:
        """Embed one or more texts using the specified model."""
        inputs = [text] if isinstance(text, str) else text
        embeddings: list[list[float]] = []
        with _tracer.start_as_current_span("ollama.embed") as span:
            span.set_attribute("model", model)
            span.set_attribute("model_name", model)
            span.set_attribute("embedding.input_count", len(inputs))
            for chunk in inputs:
                try:
                    response = await self._client.post(
                        "/api/embeddings", json={"model": model, "prompt": chunk}
                    )
                    response.raise_for_status()
                except httpx.HTTPError as exc:
                    raise CortexModelError(f"Ollama embed failed: {exc}") from exc
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

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()
