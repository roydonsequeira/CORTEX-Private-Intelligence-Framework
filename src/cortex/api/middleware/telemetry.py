"""Request telemetry middleware with request-id propagation."""

import time
import uuid
from collections.abc import Awaitable, Callable

import structlog
from fastapi import Request, Response
from structlog.contextvars import bind_contextvars, clear_contextvars

logger = structlog.get_logger(__name__)


async def telemetry_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Bind request_id and log request completion with duration."""
    request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex)
    bind_contextvars(request_id=request_id)
    start = time.monotonic()
    try:
        response = await call_next(request)
    finally:
        clear_contextvars()
    duration_ms = (time.monotonic() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "http_request_complete",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=round(duration_ms, 2),
    )
    return response
