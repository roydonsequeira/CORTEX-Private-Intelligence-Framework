"""Optional API-key authentication middleware.

When ``api_key`` is unset (the localhost default) the middleware is a no-op, so
the developer experience is unchanged. When set, every request except CORS
preflight and the exempt public paths must carry ``Authorization: Bearer <key>``.
"""

import hmac
from typing import Any

from fastapi import Request, Response
from starlette.responses import JSONResponse

_EXEMPT_PREFIXES = ("/health", "/docs", "/redoc", "/openapi.json")


class APIKeyMiddleware:
    """Require a bearer API key on all routes except health and API docs."""

    def __init__(self, app: Any, api_key: str | None = None) -> None:
        self._app = app
        self._api_key = api_key

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        """Reject requests lacking a valid bearer token when a key is configured."""
        if scope["type"] != "http" or not self._api_key:
            await self._app(scope, receive, send)
            return
        request = Request(scope, receive=receive)
        if request.method == "OPTIONS" or self._is_exempt(request.url.path):
            await self._app(scope, receive, send)
            return
        if not self._is_authorized(request):
            response: Response = JSONResponse(
                {"detail": "Invalid or missing API key."},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )
            await response(scope, receive, send)
            return
        await self._app(scope, receive, send)

    def _is_exempt(self, path: str) -> bool:
        """Return True for public paths that never require a key."""
        return any(path.startswith(prefix) for prefix in _EXEMPT_PREFIXES)

    def _is_authorized(self, request: Request) -> bool:
        """Constant-time compare the Authorization header against the configured key."""
        header = request.headers.get("authorization", "")
        expected = f"Bearer {self._api_key}"
        return hmac.compare_digest(header, expected)
