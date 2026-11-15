"""The bundled UI is served by the API itself, with its security headers (api/web.py)."""

from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from quietpost.api import web
from quietpost.api.web import SECURITY_HEADERS, has_bundled_ui, mount_ui


def _bundle(tmp_path: Path) -> Path:
    """A small stand-in for the static export in quietpost/web/, laid out the same way."""
    web_dir = tmp_path / "web"
    (web_dir / "_next" / "static").mkdir(parents=True)
    (web_dir / "audit").mkdir()
    files = {
        "index.html": "<!doctype html><title>Quietpost</title>",
        "index.txt": "the page's data for the client router",
        "404.html": "<!doctype html><title>Not found</title>",
        # Next.js writes a page as audit.html, next to a folder of its prefetch data.
        "audit.html": "<!doctype html><title>Audit</title>",
        "audit/__next.audit.__PAGE__.txt": "prefetch data",
        "_next/static/app.js": "console.log(1)",
    }
    for name, text in files.items():
        (web_dir / name).write_text(text, encoding="utf-8")
    return web_dir


def _app_with_ui(web_dir: Path) -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "healthy"}

    assert mount_ui(app, web_dir)
    return app


def _client(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost")


@pytest.mark.asyncio
async def test_the_ui_is_served_with_its_security_headers(tmp_path: Path) -> None:
    async with _client(_app_with_ui(_bundle(tmp_path))) as client:
        page = await client.get("/")
        script = await client.get("/_next/static/app.js")
        missing = await client.get("/no-such-page")

    assert page.status_code == 200
    assert "<title>Quietpost</title>" in page.text
    assert script.status_code == 200
    for response in (page, script, missing):
        for name, value in SECURITY_HEADERS.items():
            assert response.headers[name] == value
    # The API is on the page's own origin, so nothing else may be contacted.
    assert "connect-src 'self';" in page.headers["content-security-policy"]


@pytest.mark.asyncio
async def test_a_link_to_a_page_or_a_refresh_on_it_finds_the_page(tmp_path: Path) -> None:
    async with _client(_app_with_ui(_bundle(tmp_path))) as client:
        responses = [await client.get(path) for path in ("/audit", "/audit/", "/audit.html")]
        prefetch = await client.get("/audit/__next.audit.__PAGE__.txt")

    for response in responses:
        assert response.status_code == 200
        assert "<title>Audit</title>" in response.text
    assert prefetch.text == "prefetch data"


@pytest.mark.asyncio
async def test_only_files_named_after_their_content_are_cached_for_good(tmp_path: Path) -> None:
    """After an upgrade, a cached old page would ask for scripts that are gone."""
    async with _client(_app_with_ui(_bundle(tmp_path))) as client:
        script = await client.get("/_next/static/app.js")
        unchanged = await client.get(
            "/_next/static/app.js", headers={"If-None-Match": script.headers["etag"]}
        )
        pages = [await client.get(path) for path in ("/", "/audit", "/index.txt")]
        missing_script = await client.get("/_next/static/gone.js")

    assert unchanged.status_code == 304
    for response in (script, unchanged):
        assert response.headers["cache-control"] == "public, max-age=31536000, immutable"
    assert missing_script.status_code == 404
    for response in (*pages, missing_script):
        assert response.headers["cache-control"] == "no-cache"


@pytest.mark.asyncio
async def test_api_routes_still_answer_and_unknown_paths_get_the_404_page(
    tmp_path: Path,
) -> None:
    async with _client(_app_with_ui(_bundle(tmp_path))) as client:
        health = await client.get("/health")
        missing = await client.get("/no-such-page")
        posted = await client.post("/no-such-page")

    assert health.json() == {"status": "healthy"}
    assert missing.status_code == 404
    assert "Not found" in missing.text
    assert posted.status_code == 405


def test_without_a_bundle_only_the_api_is_served(tmp_path: Path) -> None:
    app = FastAPI()

    assert not has_bundled_ui(tmp_path / "web")
    assert mount_ui(app, tmp_path / "web") is False
    assert not any(getattr(route, "name", "") == "ui" for route in app.routes)


def test_create_app_mounts_the_ui_after_every_api_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from quietpost.api import server

    # conftest hides any UI built into this checkout; this test gives the app one.
    assert not has_bundled_ui()
    monkeypatch.setattr(web, "WEB_DIR", _bundle(tmp_path))

    app = server.create_app()

    assert getattr(app.routes[-1], "name", "") == "ui"
