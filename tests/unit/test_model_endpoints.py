"""Models run on this computer or the local network only (models/endpoints.py).

That rules out a model server on a public address, and a model that a local
Ollama passes on to another computer, such as one of Ollama's cloud models.
"""

import ipaddress
from typing import Any
from unittest.mock import AsyncMock

import pytest

from quietpost.api import server
from quietpost.config.settings import Settings
from quietpost.exceptions import QuietpostConfigError, QuietpostModelError
from quietpost.models import endpoints, factory
from quietpost.models.endpoints import ensure_local_model_server, is_cloud_model_name
from quietpost.models.factory import create_provider, ensure_models_run_here
from quietpost.models.openai_compat import OpenAICompatibleProvider
from quietpost.models.provider import Message, OllamaProvider
from tests.unit._fake_ollama import OLLAMA_COM, FakeOllama, route_ollama


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:11434",
        "http://127.0.0.1:11434",
        "http://[::1]:8080/v1",
        "http://192.168.1.20:1234/v1",
        "http://10.0.0.5:8000/v1",
        "http://172.17.0.2:11434",  # a Docker network
        "http://100.101.102.103:11434",  # Tailscale
        "http://[fd7a:115c:a1e0::1]:11434",  # Tailscale over IPv6
        "127.0.0.1:11434",  # no scheme
    ],
)
async def test_this_computer_and_private_networks_are_allowed(url: str) -> None:
    await ensure_local_model_server(url)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "http://8.8.8.8:11434",
        "https://1.1.1.1/v1",
        "http://[2606:4700:4700::1111]:8080/v1",
        "http://[::ffff:8.8.8.8]:11434",  # a public IPv4 address written as IPv6
    ],
)
async def test_public_addresses_are_refused(url: str) -> None:
    with pytest.raises(QuietpostConfigError, match="public internet address"):
        await ensure_local_model_server(url)


@pytest.mark.asyncio
async def test_a_name_that_resolves_to_a_public_address_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolve = AsyncMock(return_value=[ipaddress.ip_address("93.184.215.14")])
    monkeypatch.setattr(endpoints, "resolve_addresses", resolve)

    with pytest.raises(QuietpostConfigError, match=r"api\.example\.com is a public"):
        await ensure_local_model_server("https://api.example.com/v1")
    resolve.assert_awaited_once_with("api.example.com")


@pytest.mark.asyncio
async def test_one_public_address_among_private_ones_is_enough_to_refuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    addresses = [ipaddress.ip_address("10.0.0.2"), ipaddress.ip_address("93.184.215.14")]
    monkeypatch.setattr(endpoints, "resolve_addresses", AsyncMock(return_value=addresses))

    with pytest.raises(QuietpostConfigError):
        await ensure_local_model_server("http://split-horizon.example:11434")


@pytest.mark.asyncio
async def test_a_name_that_does_not_resolve_is_left_to_fail_when_connecting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(endpoints, "resolve_addresses", AsyncMock(return_value=None))

    await ensure_local_model_server("http://gpu-box.lan:8000/v1")


@pytest.mark.asyncio
async def test_a_url_without_a_host_is_refused() -> None:
    with pytest.raises(QuietpostConfigError, match="no host"):
        await ensure_local_model_server("http://:11434")


@pytest.mark.asyncio
async def test_create_provider_builds_the_configured_client() -> None:
    ollama = await create_provider(Settings())
    lm_studio = await create_provider(Settings(model_server="openai-compatible"))
    lan = await create_provider(
        Settings(model_server="openai-compatible", model_server_url="http://192.168.1.20:8080")
    )
    try:
        assert isinstance(ollama, OllamaProvider)
        assert ollama.base_url == "http://127.0.0.1:11434"
        assert isinstance(lm_studio, OpenAICompatibleProvider)
        assert lm_studio.base_url == "http://127.0.0.1:1234/v1"
        assert lan.base_url == "http://192.168.1.20:8080/v1"
    finally:
        for provider in (ollama, lm_studio, lan):
            await provider.aclose()


@pytest.mark.asyncio
async def test_create_provider_refuses_a_public_server_before_building_a_client() -> None:
    with pytest.raises(QuietpostConfigError):
        await create_provider(
            Settings(model_server="openai-compatible", model_server_url="https://8.8.8.8/v1")
        )


@pytest.mark.parametrize(
    "model",
    [
        "gpt-oss:120b-cloud",
        "gemma4:cloud",
        "deepseek-v3.1:671b-cloud",
        "gpt-oss:20b:cloud",  # the tag is what follows the last colon
        "GPT-OSS:120B-CLOUD",
        " kimi-k2:1t-cloud ",
        "someone/model:cloud",
    ],
)
def test_ollama_cloud_model_names_are_recognised(model: str) -> None:
    assert is_cloud_model_name(model)


@pytest.mark.parametrize(
    "model",
    [
        "qwen2.5:7b",
        "nomic-embed-text",
        "gpt-oss-cloud",  # no tag
        "cloud",
        "registry.example.com:5000/model-cloud",  # a port, not a tag
        "hf.co/someone/model-cloud:Q4_K_M",
        "qwen2.5:7b:local",
        "model:x/y-cloud",
    ],
)
def test_other_model_names_are_not_cloud_models(model: str) -> None:
    assert not is_cloud_model_name(model)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("setting", "example"),
    [
        ("chat_model", "qwen2.5:7b"),
        ("reasoning_model", "qwen2.5:7b"),
        ("embed_model", "nomic-embed-text"),
    ],
)
async def test_create_provider_refuses_a_cloud_model_before_building_a_client(
    setting: str, example: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeOllama({})
    route_ollama(monkeypatch, factory, fake.transport())
    cloud_model: dict[str, Any] = {setting: "gpt-oss:120b-cloud"}

    with pytest.raises(QuietpostConfigError) as refused:
        await create_provider(Settings(**cloud_model))

    assert str(refused.value) == (
        f"{setting}: Ollama passes gpt-oss:120b-cloud on to ollama.com instead of running it "
        "on this computer, so everything Quietpost sent it would leave this computer. Set "
        f"{setting} to a model that runs on this computer, such as {example}."
    )
    assert fake.requests == []


@pytest.mark.asyncio
async def test_a_cloud_model_is_refused_with_any_model_server() -> None:
    """Ollama's OpenAI-compatible API serves its cloud models too."""
    settings = Settings(model_server="openai-compatible", chat_model="gpt-oss:120b-cloud")

    with pytest.raises(QuietpostConfigError, match="^chat_model: Ollama passes gpt-oss"):
        await create_provider(settings)


@pytest.mark.asyncio
async def test_the_ollama_client_refuses_a_model_ollama_passes_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeOllama({"my-assistant": OLLAMA_COM})
    route_ollama(monkeypatch, factory, fake.transport())
    provider = await create_provider(Settings())
    try:
        with pytest.raises(QuietpostModelError, match="Quietpost sent it nothing"):
            await provider.complete("my-assistant", [Message(role="user", content="my email")])
    finally:
        await provider.aclose()

    assert fake.paths() == ["/api/show"]


def _client(fake: FakeOllama) -> OllamaProvider:
    return OllamaProvider("http://127.0.0.1:11434", local_only=True, transport=fake.transport())


@pytest.mark.asyncio
async def test_the_server_does_not_start_with_a_model_ollama_passes_on() -> None:
    fake = FakeOllama({"qwen2.5:7b": None, "my-assistant": OLLAMA_COM})
    provider = _client(fake)

    with pytest.raises(QuietpostConfigError) as refused:
        await server._check_models(provider, Settings(code_model="my-assistant"))

    assert str(refused.value).startswith(
        f"code_model: Ollama passes my-assistant on to {OLLAMA_COM} instead of running it"
    )
    assert "such as qwen2.5:7b." in str(refused.value)
    assert set(fake.paths()) == {"/api/tags", "/api/show"}


@pytest.mark.asyncio
async def test_the_server_starts_when_each_model_runs_here_or_is_missing() -> None:
    """A missing model is logged with the command that downloads it, as before."""
    fake = FakeOllama({"qwen2.5:7b": None})

    await server._check_models(_client(fake), Settings())

    assert [request for request in fake.requests if request[0] == "/api/show"] == [
        ("/api/show", "qwen2.5:7b"),
        ("/api/show", "nomic-embed-text"),
    ]


@pytest.mark.asyncio
async def test_a_model_ollama_lacks_does_not_end_the_check_of_the_others() -> None:
    fake = FakeOllama({"nomic-embed-text": None, "my-assistant": OLLAMA_COM})
    settings = Settings(chat_model="not-downloaded-yet", code_model="my-assistant")

    with pytest.raises(QuietpostConfigError, match="^code_model: Ollama passes my-assistant"):
        await ensure_models_run_here(_client(fake), settings)


@pytest.mark.asyncio
async def test_only_ollama_is_asked_where_a_model_runs() -> None:
    """Other model servers don't say; their models' names were checked already."""
    provider = OpenAICompatibleProvider("http://127.0.0.1:1234/v1")
    try:
        await ensure_models_run_here(provider, Settings(model_server="openai-compatible"))
    finally:
        await provider.aclose()
