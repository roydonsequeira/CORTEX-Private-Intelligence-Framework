"""Quietpost uses models on this machine or the local network, never a public server.

Email is the most private data most people have. A model server URL that
reaches the internet (a pasted cloud endpoint, a typo, a host that moved)
would send it there, so it is refused before any request is made. So is a
model that a local Ollama passes on to another computer, such as one of
Ollama's cloud models: the address is local, but the model isn't.
"""

from collections.abc import Mapping
from urllib.parse import urlsplit

from quietpost.exceptions import QuietpostConfigError
from quietpost.models.profiles import EMBED_MODEL, profile_for
from quietpost.network import is_public_address, resolve_addresses

# Where Ollama runs a model whose name marks it as a cloud model.
OLLAMA_CLOUD = "ollama.com"


async def ensure_local_model_server(url: str) -> None:
    """Raise QuietpostConfigError if ``url``'s host is, or resolves to, a public address.

    Loopback, private networks and Tailscale-style CGNAT addresses are allowed.
    A host that does not resolve is allowed here: connecting to it fails with
    its own clear error.
    """
    host = urlsplit(url if "://" in url else f"http://{url}").hostname
    if not host:
        raise QuietpostConfigError(f"The model server URL has no host: {url!r}.")
    addresses = await resolve_addresses(host)
    public = [address for address in addresses or [] if is_public_address(address)]
    if public:
        raise QuietpostConfigError(
            f"Quietpost only uses models on this computer or your local network, and "
            f"{host} is a public internet address ({public[0]}). Set model_server_url to "
            "a server on this machine (such as http://localhost:11434 for Ollama) or on "
            "your local network, using its local IPv4 address."
        )


def is_cloud_model_name(model: str) -> bool:
    """Whether ``model`` names one of Ollama's cloud models, which run on ollama.com.

    This is Ollama's own rule (its internal/modelref package): the tag, the text
    after the last colon, is "cloud" or ends in "-cloud", as in gemma4:cloud or
    gpt-oss:120b-cloud. Since Ollama 0.17.6 a request for such a name goes to
    ollama.com even when nothing was downloaded.
    """
    _, colon, tag = model.strip().rpartition(":")
    if not colon:
        return False
    tag = tag.strip().lower()
    return tag == "cloud" or ("/" not in tag and tag.endswith("-cloud"))


def runs_elsewhere(model: str, host: str) -> str:
    """Why a model is refused: Ollama passes it on to ``host``."""
    return (
        f"Ollama passes {model} on to {host} instead of running it on this computer, "
        "so everything Quietpost sent it would leave this computer"
    )


def model_refusal(setting: str, model: str, host: str) -> str:
    """The error for a setting that names a model Ollama passes on to ``host``."""
    example = EMBED_MODEL if setting == "embed_model" else profile_for(None).chat_model
    return (
        f"{setting}: {runs_elsewhere(model, host)}. Set {setting} to a model that runs "
        f"on this computer, such as {example}."
    )


def ensure_no_cloud_models(models: Mapping[str, str]) -> None:
    """Raise QuietpostConfigError if a setting names one of Ollama's cloud models.

    ``models`` maps each setting to its model, as Settings.configured_models does.
    Only the name is needed, so nothing is asked of the model server.
    """
    for setting, model in models.items():
        if is_cloud_model_name(model):
            raise QuietpostConfigError(model_refusal(setting, model, OLLAMA_CLOUD))
