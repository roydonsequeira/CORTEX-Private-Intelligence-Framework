"""Serves the bundled UI: the static build of ui/ that ships inside the package.

`npm run build:bundle` in ui/ puts it in quietpost/web/. When it is there, the
API serves the UI on its own address, so an installed Quietpost needs no
Node.js. A source checkout without it serves only the API; the UI then runs
with `npm run dev`.
"""

import os
from pathlib import Path, PurePath
from typing import Any

from fastapi import FastAPI
from starlette.responses import Response
from starlette.staticfiles import StaticFiles

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

# The same policy the UI's own server sends (ui/next.config.mjs), with the API on
# the page's own origin: the page may talk only to the server it came from. Even
# if text the model read got markup into an answer, the browser would refuse to
# send anything elsewhere.
_CONTENT_SECURITY_POLICY = "; ".join(
    [
        "default-src 'self'",
        # Next.js inlines its bootstrap scripts.
        "script-src 'self' 'unsafe-inline'",
        "style-src 'self' 'unsafe-inline'",
        "img-src 'self' data: blob:",
        "font-src 'self' data:",
        "connect-src 'self'",
        "object-src 'none'",
        "base-uri 'self'",
        "form-action 'self'",
        "frame-ancestors 'none'",
    ]
)
SECURITY_HEADERS = {
    "Content-Security-Policy": _CONTENT_SECURITY_POLICY,
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
}
# Scripts and styles under _next/static/ are named after their content, so a
# cached copy is never stale. Everything else is checked on each visit: after an
# upgrade, an old cached page would ask for files that no longer exist.
_CACHE_FOREVER = "public, max-age=31536000, immutable"
_CHECK_EACH_TIME = "no-cache"


class _UIFiles(StaticFiles):
    """The UI's files, each sent with the security headers."""

    async def get_response(self, path: str, scope: Any) -> Response:
        response = await super().get_response(path, scope)
        if response.status_code == 404 and not os.path.splitext(path)[1]:
            # Next.js exports the page /audit as audit.html, so a link to
            # /audit, or a refresh there, needs the name with .html added.
            response = await super().get_response(f"{path}.html", scope)
        response.headers.update(SECURITY_HEADERS)
        hashed = PurePath(path).parts[:2] == ("_next", "static")
        found = response.status_code in (200, 304)
        response.headers["Cache-Control"] = _CACHE_FOREVER if hashed and found else _CHECK_EACH_TIME
        return response


def has_bundled_ui(web_dir: Path | None = None) -> bool:
    """True when this installation includes the built UI."""
    return ((web_dir or WEB_DIR) / "index.html").is_file()


def mount_ui(app: FastAPI, web_dir: Path | None = None) -> bool:
    """Serve the bundled UI at / (after the API routes); False when there is none."""
    web_dir = web_dir or WEB_DIR
    if not has_bundled_ui(web_dir):
        return False
    app.mount("/", _UIFiles(directory=web_dir, html=True), name="ui")
    return True
