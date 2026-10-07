"""Builds the model provider the settings ask for."""

from quietpost.config.settings import Settings
from quietpost.models.endpoints import ensure_local_model_server
from quietpost.models.openai_compat import OpenAICompatibleProvider
from quietpost.models.provider import ModelProvider, OllamaProvider


async def create_provider(settings: Settings) -> ModelProvider:
    """A client for the configured model server.

    Raises QuietpostConfigError, before any request is made, when the server is
    on a public address: Quietpost only uses models on this machine or the
    local network.
    """
    await ensure_local_model_server(settings.model_url)
    if settings.model_server == "openai-compatible":
        return OpenAICompatibleProvider(
            settings.model_url,
            timeout=settings.model_timeout_seconds,
            api_key=settings.model_server_api_key,
        )
    return OllamaProvider(
        settings.model_url,
        timeout=settings.model_timeout_seconds,
        num_ctx=settings.ollama_num_ctx,
        keep_alive=settings.ollama_keep_alive,
    )
