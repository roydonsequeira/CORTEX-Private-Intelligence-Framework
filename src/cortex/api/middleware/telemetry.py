"""Request telemetry middleware with request-id propagation."""

import re
import time
import uuid
from collections.abc import Awaitable, Callable

import structlog
from fastapi import Request, Response
from structlog.contextvars import bind_contextvars, clear_contextvars

logger = structlog.get_logger(__name__)

# A client-supplied X-Request-ID is echoed into logs, so accept only a short,
# plain token; anything else gets a fresh id (no log injection via headers).
_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


async def telemetry_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Bind request_id and log request completion with duration."""
    supplied = request.headers.get("X-Request-ID", "")
    request_id = supplied if _REQUEST_ID.match(supplied) else uuid.uuid4().hex
    bind_contextvars(request_id=request_id)
    start = time.monotonic()
    try:
        response = await call_next(request)
        duration_ms = (time.monotonic() - start) * 1000
        response.headers["X-Request-ID"] = request_id
        # Logged before the context is cleared so the line carries its request_id.
        logger.info(
            "http_request_complete",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=round(duration_ms, 2),
        )
        return response
    finally:
        clear_contextvars()
