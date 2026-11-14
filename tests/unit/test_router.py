"""Unit tests for ModelRouter capability routing."""

from unittest.mock import MagicMock

from quietpost.config.settings import Settings
from quietpost.models.provider import OllamaProvider
from quietpost.models.router import ModelCapability, ModelRouter


def test_router_maps_every_capability_from_settings() -> None:
    """Each capability routes to the model configured in settings."""
    settings = Settings(
        chat_model="fast-m",
        reasoning_model="reason-m",
        code_model="code-m",
        embed_model="embed-m",
    )
    router = ModelRouter(MagicMock(spec=OllamaProvider), settings)

    assert router.route(ModelCapability.FAST) == "fast-m"
    assert router.route(ModelCapability.REASONING) == "reason-m"
    assert router.route(ModelCapability.CODE) == "code-m"
    assert router.route(ModelCapability.EMBEDDING) == "embed-m"


def test_extract_routes_to_the_extract_model() -> None:
    settings = Settings(chat_model="chat-m", extract_model="small-m")
    router = ModelRouter(MagicMock(spec=OllamaProvider), settings)

    assert router.route(ModelCapability.EXTRACT) == "small-m"
    assert router.route(ModelCapability.FAST) == "chat-m"


def test_roles_without_their_own_model_use_the_chat_model() -> None:
    """One model setting is enough, which matters on servers that name models their own way."""
    router = ModelRouter(MagicMock(spec=OllamaProvider), Settings(chat_model="qwen2.5-7b-instruct"))

    for capability in (
        ModelCapability.FAST,
        ModelCapability.REASONING,
        ModelCapability.CODE,
        ModelCapability.EXTRACT,
    ):
        assert router.route(capability) == "qwen2.5-7b-instruct"
    assert router.route(ModelCapability.EMBEDDING) == "nomic-embed-text"


def test_router_low_resource_profile_points_all_at_one_model() -> None:
    """A demo profile can route reasoning/fast/code at a single tiny model."""
    settings = Settings(
        chat_model="llama3.2:1b",
        reasoning_model="llama3.2:1b",
        code_model="llama3.2:1b",
        embed_model="nomic-embed-text",
    )
    router = ModelRouter(MagicMock(spec=OllamaProvider), settings)

    assert router.route(ModelCapability.REASONING) == "llama3.2:1b"
    assert router.route(ModelCapability.FAST) == "llama3.2:1b"
    assert router.route(ModelCapability.CODE) == "llama3.2:1b"


def test_router_explicit_routing_overrides_settings() -> None:
    """An explicit routing dict takes precedence over settings defaults."""
    settings = Settings(chat_model="fast-m")
    router = ModelRouter(
        MagicMock(spec=OllamaProvider),
        settings,
        routing={
            ModelCapability.FAST: "override-m",
            ModelCapability.REASONING: "override-m",
            ModelCapability.CODE: "override-m",
            ModelCapability.EMBEDDING: "override-m",
        },
    )

    assert router.route(ModelCapability.FAST) == "override-m"
