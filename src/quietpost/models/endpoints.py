"""Quietpost uses models on this machine or the local network, never a public server.

Email is the most private data most people have. A model server URL that
reaches the internet (a pasted cloud endpoint, a typo, a host that moved)
would send it there, so it is refused before any request is made.
"""

from urllib.parse import urlsplit

from quietpost.exceptions import QuietpostConfigError
from quietpost.network import is_public_address, resolve_addresses


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
