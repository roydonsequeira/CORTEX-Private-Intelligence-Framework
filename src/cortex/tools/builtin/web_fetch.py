"""Web fetch tool — offline-tolerant HTML/text retrieval with robots.txt checks."""

import time
from typing import Literal
from urllib import robotparser
from urllib.parse import urlparse

import html2text
import httpx

from cortex.observability.tracing import get_tracer
from cortex.tools.base import BaseTool, ToolResult, ToolSchema

_MAX_RESPONSE_BYTES = 50 * 1024
_USER_AGENT = "CORTEX/0.1"
_tracer = get_tracer(__name__)


class WebFetchTool(BaseTool):
    """Fetch and parse web pages as readable text or markdown."""

    schema = ToolSchema(
        name="web_fetch",
        description="Fetch and parse web pages as readable markdown text.",
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

    def __init__(self, timeout_seconds: float = 20.0) -> None:
        self._client = httpx.AsyncClient(timeout=timeout_seconds, follow_redirects=True)

    async def execute(self, **kwargs: object) -> ToolResult:
        """Fetch a URL and return readable content."""
        start = time.monotonic()
        url = str(kwargs["url"])
        output_format: Literal["text", "markdown"] = (
            "markdown" if kwargs.get("format", "markdown") == "markdown" else "text"
        )
        try:
            parsed = urlparse(url)
            with _tracer.start_as_current_span("http.client.request") as span:
                span.set_attribute("tool_name", self.schema.name)
                span.set_attribute("url.full", url)
                span.set_attribute("server.address", parsed.netloc)
                span.set_attribute("http.request.method", "GET")
                if not await self._allowed_by_robots(url):
                    return _result(False, "", start, "Blocked by robots.txt")
                response = await self._client.get(url, headers={"User-Agent": _USER_AGENT})
                response.raise_for_status()
                span.set_attribute("http.response.status_code", response.status_code)
            raw = response.content[:_MAX_RESPONSE_BYTES].decode(
                response.encoding or "utf-8", errors="replace"
            )
            output = _to_markdown(raw) if output_format == "markdown" else _strip_html(raw)
            if len(response.content) > _MAX_RESPONSE_BYTES:
                output += "\n[response truncated]"
            return _result(True, output, start)
        except httpx.HTTPError:
            return _result(
                False,
                "",
                start,
                "Network unavailable — CORTEX is running in offline mode",
            )

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
            return parser.can_fetch(_USER_AGENT, url)
        except httpx.HTTPError:
            return True

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()


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
