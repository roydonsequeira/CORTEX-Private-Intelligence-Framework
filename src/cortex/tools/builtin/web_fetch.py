"""Web fetch tool — offline-tolerant HTML/text retrieval with robots.txt checks."""

import asyncio
import ssl
import time
from typing import Any, Literal
from urllib import robotparser
from urllib.parse import urlparse

import html2text
import httpx

from cortex.observability.tracing import get_tracer
from cortex.tools.base import BaseTool, ToolResult, ToolSchema

# Bytes downloaded per page (a Wikipedia article is ~1 MB of HTML, most of it
# markup), and characters of converted text returned to the model.
_MAX_DOWNLOAD_BYTES = 2 * 1024 * 1024
_MAX_OUTPUT_CHARS = 12_000
# A descriptive agent string with a contact URL; several sites (Wikipedia among
# them) reject generic or bare agent strings with HTTP 403.
_USER_AGENT = (
    "CORTEX/1.0 (local research agent; "
    "+https://github.com/roydonsequeira/CORTEX-Private-Intelligence-Framework)"
)
_ROBOTS_AGENT = "CORTEX"
_tracer = get_tracer(__name__)


def _ssl_context() -> ssl.SSLContext | bool:
    """Verify TLS against the operating system trust store when available.

    Corporate proxies and antivirus products that inspect HTTPS install their
    root certificate in the OS store, not in certifi's bundle, so plain certifi
    verification fails on those machines. ``truststore`` delegates to the OS.
    """
    try:
        import truststore

        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except Exception:  # optional dependency or unsupported platform
        return True


class WebFetchTool(BaseTool):
    """Fetch and parse web pages as readable text or markdown."""

    schema = ToolSchema(
        name="web_fetch",
        description=(
            "Fetch a web page by full URL and return its readable text as markdown. "
            "Use only when the user gives a URL or asks for live web content."
        ),
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "format": "uri"},
                "format": {"type": "string", "enum": ["text", "markdown"]},
            },
            "required": ["url"],
            "additionalProperties": False,
        },
    )

    def __init__(self, timeout_seconds: float = 20.0, client: Any | None = None) -> None:
        self._client = client or httpx.AsyncClient(
            timeout=timeout_seconds,
            follow_redirects=True,
            verify=_ssl_context(),
            headers={"User-Agent": _USER_AGENT},
        )

    async def execute(self, **kwargs: object) -> ToolResult:
        """Fetch a URL and return readable content."""
        start = time.monotonic()
        url = _normalise_url(str(kwargs["url"]))
        output_format: Literal["text", "markdown"] = (
            "markdown" if kwargs.get("format", "markdown") == "markdown" else "text"
        )
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            return _result(False, "", start, f"Not a valid web URL: {url}")
        try:
            with _tracer.start_as_current_span("http.client.request") as span:
                span.set_attribute("tool_name", self.schema.name)
                span.set_attribute("url.full", url)
                span.set_attribute("server.address", parsed.netloc)
                span.set_attribute("http.request.method", "GET")
                if not await self._allowed_by_robots(url):
                    return _result(False, "", start, "Blocked by the site's robots.txt.")
                response = await self._client.get(url, headers={"User-Agent": _USER_AGENT})
                span.set_attribute("http.response.status_code", response.status_code)
            if response.status_code >= 400:
                return _result(
                    False, "", start, f"HTTP {response.status_code} fetching {url}."
                )
            body = response.content[:_MAX_DOWNLOAD_BYTES]
            raw = body.decode(response.encoding or "utf-8", errors="replace")
            content_type = response.headers.get("content-type", "")
            if "html" in content_type or raw.lstrip()[:1] == "<":
                # HTML conversion of a large page is CPU-bound: keep it off the event loop.
                convert = _to_markdown if output_format == "markdown" else _strip_html
                output = await asyncio.to_thread(convert, raw)
            else:
                output = raw
            if len(output) > _MAX_OUTPUT_CHARS:
                output = output[:_MAX_OUTPUT_CHARS] + "\n[content truncated]"
            return _result(True, output or "(the page has no readable text)", start)
        except httpx.TimeoutException:
            return _result(False, "", start, f"Timed out fetching {url}.")
        except httpx.ConnectError as exc:
            message = str(exc)
            if "CERTIFICATE_VERIFY_FAILED" in message:
                return _result(
                    False, "", start, f"TLS certificate verification failed for {parsed.netloc}."
                )
            return _result(
                False,
                "",
                start,
                f"Could not connect to {parsed.netloc} — the network may be offline.",
            )
        except httpx.HTTPError as exc:
            return _result(False, "", start, f"Request to {parsed.netloc} failed: {exc}")

    async def _allowed_by_robots(self, url: str) -> bool:
        """Return whether url is allowed for CORTEX according to robots.txt."""
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        parser = robotparser.RobotFileParser()
        try:
            response = await self._client.get(
                robots_url, headers={"User-Agent": _USER_AGENT}
            )
            if response.status_code >= 400:
                return True
            parser.parse(response.text.splitlines())
            return parser.can_fetch(_ROBOTS_AGENT, url)
        except httpx.HTTPError:
            return True

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()


def _normalise_url(url: str) -> str:
    """Trim quotes/whitespace and add https:// to scheme-less URLs ("example.com")."""
    cleaned = url.strip().strip("\"'<>")
    if cleaned and "://" not in cleaned:
        cleaned = "https://" + cleaned.lstrip("/")
    return cleaned


def _to_markdown(html: str) -> str:
    """Convert HTML to markdown, ignoring script/style/nav content."""
    converter = html2text.HTML2Text()
    converter.ignore_links = False
    converter.ignore_images = True
    converter.body_width = 0
    return converter.handle(_strip_unwanted(html)).strip()


def _strip_html(html: str) -> str:
    """Convert HTML to plain-ish text via html2text without markdown links."""
    converter = html2text.HTML2Text()
    converter.ignore_links = True
    converter.ignore_images = True
    converter.body_width = 0
    return converter.handle(_strip_unwanted(html)).strip()


def _strip_unwanted(html: str) -> str:
    """Remove common noisy HTML blocks before conversion."""
    import re

    patterns = [
        r"<script\b[^<]*(?:(?!</script>)<[^<]*)*</script>",
        r"<style\b[^<]*(?:(?!</style>)<[^<]*)*</style>",
        r"<nav\b[^<]*(?:(?!</nav>)<[^<]*)*</nav>",
        r"<header\b[^<]*(?:(?!</header>)<[^<]*)*</header>",
        r"<footer\b[^<]*(?:(?!</footer>)<[^<]*)*</footer>",
    ]
    cleaned = html
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    return cleaned


def _result(
    success: bool, output: str, start: float, error: str | None = None
) -> ToolResult:
    """Build a ToolResult for web_fetch."""
    return ToolResult(
        tool_name="web_fetch",
        success=success,
        output=output,
        error=error,
        execution_time_ms=(time.monotonic() - start) * 1000,
    )
