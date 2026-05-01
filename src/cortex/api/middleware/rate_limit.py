"""Pure-Python token bucket rate limiting middleware."""

import time
from dataclasses import dataclass
from typing import Any

from fastapi import Request, Response
from starlette.responses import JSONResponse


@dataclass
class TokenBucket:
    """Mutable token bucket state for one client and route class."""

    tokens: float
    updated_at: float


class RateLimitMiddleware:
    """Per-IP token bucket limiter with separate chat and general limits."""

    def __init__(
        self,
        app: Any,
        enabled: bool = True,
        requests_per_minute: int = 200,
        chat_requests_per_minute: int = 60,
    ) -> None:
        self._app = app
        self._enabled = enabled
        self._requests_per_minute = requests_per_minute
        self._chat_requests_per_minute = chat_requests_per_minute
        self._buckets: dict[tuple[str, str], TokenBucket] = {}

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        """Apply token bucket rate limiting to HTTP requests."""
        if scope["type"] != "http" or not self._enabled:
            await self._app(scope, receive, send)
            return
        request = Request(scope, receive=receive)
        route_class = "chat" if request.url.path.startswith("/chat") else "default"
        limit = (
            self._chat_requests_per_minute
            if route_class == "chat"
            else self._requests_per_minute
        )
        client = request.client.host if request.client else "unknown"
        retry_after = self._consume(client, route_class, limit)
        if retry_after is not None:
            response: Response = JSONResponse(
                {"detail": "Rate limit exceeded."},
                status_code=429,
                headers={"Retry-After": str(max(1, int(retry_after)))},
            )
            await response(scope, receive, send)
            return
        await self._app(scope, receive, send)

    def _consume(self, client: str, route_class: str, limit: int) -> float | None:
        """Consume a token or return retry-after seconds."""
        now = time.monotonic()
        key = (client, route_class)
        bucket = self._buckets.get(key)
        if bucket is None:
            self._buckets[key] = TokenBucket(tokens=float(limit - 1), updated_at=now)
            return None
        refill_rate = limit / 60.0
        elapsed = now - bucket.updated_at
        bucket.tokens = min(float(limit), bucket.tokens + elapsed * refill_rate)
        bucket.updated_at = now
        if bucket.tokens >= 1.0:
            bucket.tokens -= 1.0
            return None
        return (1.0 - bucket.tokens) / refill_rate
