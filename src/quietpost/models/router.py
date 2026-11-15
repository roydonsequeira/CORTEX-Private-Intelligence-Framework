"""Model router — selects the right model for a given capability."""

from collections.abc import AsyncIterator
from enum import StrEnum
from typing import Any

from quietpost.config.settings import Settings
from quietpost.models.provider import (
    GenerationConfig,
    Message,
    ModelProvider,
    ModelResponse,
    StreamChunk,
)


class ModelCapability(StrEnum):
    """High-level capability buckets used to route requests to the right model.

    EXTRACT is for reading one email at a time, without tools, into structured
    fields: work a smaller, faster model can do.
    """

    REASONING = "REASONING"
    FAST = "FAST"
    EXTRACT = "EXTRACT"
    EMBEDDING = "EMBEDDING"
    CODE = "CODE"


class ModelRouter:
    """Routes model requests to the appropriate model by capability."""

    def __init__(
        self,
        provider: ModelProvider,
        settings: Settings,
        routing: dict[ModelCapability, str] | None = None,
    ) -> None:
        self._provider = provider
        self._settings = settings
        models = settings.configured_models
        self._routing = routing or {
            ModelCapability.REASONING: models["reasoning_model"],
            ModelCapability.FAST: models["chat_model"],
            ModelCapability.EXTRACT: models["extract_model"],
            ModelCapability.EMBEDDING: models["embed_model"],
            ModelCapability.CODE: models["code_model"],
        }

    def route(self, capability: ModelCapability) -> str:
        """Return the model name mapped to the requested capability."""
        return self._routing[capability]

    async def complete(
        self,
        capability: ModelCapability,
        messages: list[Message],
        config: GenerationConfig | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelResponse:
        """Delegate a completion call to the provider using the routed model."""
        model = self.route(capability)
        return await self._provider.complete(model, messages, config, tools)

    async def stream_complete(
        self,
        capability: ModelCapability,
        messages: list[Message],
        config: GenerationConfig | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a completion from the routed model, yielding StreamChunk deltas."""
        model = self.route(capability)
        async for chunk in self._provider.stream_complete(model, messages, config, tools):
            yield chunk

    async def embed(self, text: str | list[str]) -> list[list[float]]:
        """Embed text using the configured embedding model."""
        model = self.route(ModelCapability.EMBEDDING)
        return await self._provider.embed(model, text)
