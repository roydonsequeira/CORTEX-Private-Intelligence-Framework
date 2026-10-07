"""The model server must be on this computer or the local network (models/endpoints.py)."""

import ipaddress
from unittest.mock import AsyncMock

import pytest

from quietpost.config.settings import Settings
from quietpost.exceptions import QuietpostConfigError
from quietpost.models import endpoints
from quietpost.models.endpoints import ensure_local_model_server
from quietpost.models.factory import create_provider
from quietpost.models.openai_compat import OpenAICompatibleProvider
from quietpost.models.provider import OllamaProvider


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
