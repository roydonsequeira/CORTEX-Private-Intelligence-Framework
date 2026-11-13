"""Builds the model provider the settings ask for, and checks where its models run."""

from quietpost.config.settings import Settings
from quietpost.exceptions import QuietpostConfigError, QuietpostModelError
from quietpost.models.endpoints import (
    ensure_local_model_server,
    ensure_no_cloud_models,
    model_refusal,
)
from quietpost.models.openai_compat import OpenAICompatibleProvider
from quietpost.models.provider import ModelProvider, OllamaProvider


async def create_provider(settings: Settings) -> ModelProvider:
    """A client for the configured model server.

    Raises QuietpostConfigError, before any request is made, when the server is
    on a public address or a setting names one of Ollama's cloud models:
    Quietpost only uses models on this machine or the local network. The Ollama
    client also refuses, before a model's first request, a model Ollama passes
    on to another computer.
    """
    await ensure_local_model_server(settings.model_url)
    ensure_no_cloud_models(settings.configured_models)
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
        think=settings.ollama_think,
        local_only=True,
    )


async def ensure_models_run_here(provider: ModelProvider, settings: Settings) -> None:
    """Raise QuietpostConfigError if Ollama passes a configured model on to another computer.

    This is for refusing to start: the Ollama client checks each model again
    before its first request. A model Ollama doesn't have, or an Ollama that
    can't be asked, is left to that check. Other model servers don't say where a
    model runs, so only the names were checked (create_provider).
    """
    if not isinstance(provider, OllamaProvider):
        return
    asked: set[str] = set()
    for setting, model in settings.configured_models.items():
        if model in asked:
            continue
        asked.add(model)
        try:
            host = await provider.remote_host(model)
        except QuietpostModelError:
            continue
        if host:
            raise QuietpostConfigError(model_refusal(setting, model, host))
