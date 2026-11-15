"""Where a host name points: shared by web_fetch (public only) and the model
server check (never public)."""

import asyncio
import ipaddress
import socket

IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address


async def resolve_addresses(host: str) -> list[IPAddress] | None:
    """The IP addresses ``host`` resolves to, or None if it does not resolve.

    An IP literal (with or without IPv6 brackets) is returned as is.
    """
    host = host.strip("[]")
    if not host:
        return None
    try:
        return [ipaddress.ip_address(host)]
    except ValueError:
        pass
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(host, None)
    except (socket.gaierror, UnicodeError):
        return None
    return [ipaddress.ip_address(str(info[4][0]).split("%")[0]) for info in infos]


def is_public_address(address: IPAddress) -> bool:
    """True for a globally routable address (an IPv4-mapped IPv6 is judged as IPv4).

    Loopback, private networks, link-local (cloud metadata), carrier-grade NAT
    (Tailscale's 100.64.0.0/10) and reserved ranges are not public.
    """
    mapped = getattr(address, "ipv4_mapped", None)
    return (mapped or address).is_global
