"""`quietpost` command line: start the API and diagnose a local setup.

    quietpost serve          start the API (uvicorn) with the settings from quietpost.yaml
    quietpost doctor         check Python, config, the model server, models, ports, and HTTPS
    quietpost reset-memory   wipe stored memory (asks for --yes)

Because this runs inside the interpreter Quietpost is installed in, it cannot pick
up a different Python's `uvicorn` from PATH — the most common reason a bare
`uvicorn quietpost.api.server:create_app` fails with "No module named 'quietpost'".
"""

import argparse
import asyncio
import os
import socket
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quietpost.config.settings import Settings

_OK = "[ ok ]"
_WARN = "[warn]"
_FAIL = "[FAIL]"


def main(argv: list[str] | None = None) -> int:
    """Entry point for the `quietpost` console script."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(errors="replace")
    parser = argparse.ArgumentParser(prog="quietpost", description="Quietpost local AI agent")
    sub = parser.add_subparsers(dest="command")

    serve = sub.add_parser("serve", help="start the Quietpost API server")
    serve.add_argument("--host", default=None, help="bind address (default: from config)")
    serve.add_argument("--port", type=int, default=None, help="port (default: from config)")
    serve.add_argument("--reload", action="store_true", help="auto-reload on code changes")

    sub.add_parser("doctor", help="diagnose the local setup")

    reset = sub.add_parser(
        "reset-memory", help="delete all stored memory (episodic, semantic, procedural)"
    )
    reset.add_argument("--yes", action="store_true", help="confirm the deletion")

    args = parser.parse_args(argv)
    if args.command == "serve":
        return _serve(args.host, args.port, args.reload)
    if args.command == "doctor":
        return doctor()
    if args.command == "reset-memory":
        return _reset_memory(args.yes)
    parser.print_help()
    return 0


def _serve(host: str | None, port: int | None, reload: bool) -> int:
    """Run uvicorn in this interpreter with the configured host and port."""
    import uvicorn

    from quietpost.config import get_settings
    from quietpost.exceptions import QuietpostConfigError
    from quietpost.models.endpoints import ensure_local_model_server

    settings = get_settings()
    bind_host = host or settings.api_host
    bind_port = port or settings.api_port
    try:
        asyncio.run(ensure_local_model_server(settings.model_url))
    except QuietpostConfigError as exc:
        print(f"{_FAIL} {exc}")
        return 1
    if not _port_free(bind_host, bind_port):
        print(
            f"{_FAIL} Port {bind_port} is already in use — another Quietpost (or other "
            f"server) is running. Stop it, or use: quietpost serve --port {bind_port + 1}"
        )
        return 1
    if bind_host not in ("127.0.0.1", "localhost", "::1") and not settings.api_key:
        print(
            f"{_WARN} Binding to {bind_host} without an api_key exposes an agent that can "
            "run code and read/write workspace files to your network. Set api_key "
            "(or QUIETPOST_API_KEY) and add this host name to allowed_hosts."
        )
    print(f"Quietpost API -> http://localhost:{bind_port}  (docs: http://localhost:{bind_port}/docs)")
    uvicorn.run(
        "quietpost.api.server:create_app",
        factory=True,
        host=bind_host,
        port=bind_port,
        reload=reload,
        log_level=settings.log_level.lower(),
    )
    return 0


def doctor() -> int:
    """Print a checklist of everything Quietpost needs; return non-zero on failures."""
    import httpx

    from quietpost.config.settings import Settings, _resolve_config_path, user_config_file
    from quietpost.exceptions import QuietpostConfigError, QuietpostModelError

    failures = 0
    print("Quietpost doctor\n")

    print(f"{_OK} Python {sys.version.split()[0]} at {sys.executable}")

    config_path = _resolve_config_path()
    if config_path:
        print(f"{_OK} Config: {config_path}")
    else:
        print(
            f"{_WARN} No quietpost.yaml found from {Path.cwd()} upward or in "
            f"{user_config_file().parent} — using defaults"
        )
    try:
        settings = Settings()
    except Exception as exc:
        print(f"{_FAIL} Config could not be loaded: {exc}")
        return 1

    from quietpost.api.server import missing_model_fix, missing_models, start_server_fix

    server = "Ollama" if settings.model_server == "ollama" else "Model server"
    try:
        available = asyncio.run(_server_models(settings))
        print(f"{_OK} {server} reachable at {settings.model_url} ({len(available)} models)")
    except QuietpostConfigError as exc:
        print(f"{_FAIL} {exc}")
        return failures + 1
    except QuietpostModelError as exc:
        print(f"{_FAIL} {server} not reachable at {settings.model_url}: {exc}")
        print(f"       Fix: {start_server_fix(settings)}.")
        return failures + 1

    missing = set(missing_models(settings, available))
    unset = [
        name
        for name in ("reasoning_model", "code_model", "extract_model")
        if getattr(settings, name) is None
    ]
    for setting, model in settings.configured_models.items():
        if setting in unset:
            continue
        if model in missing:
            print(f"{_FAIL} {setting}: '{model}' is missing.  Fix: {missing_model_fix(settings, model)}")
            failures += 1
        else:
            print(f"{_OK} {setting}: {model}")
    if unset:
        print(f"       {', '.join(unset)}: not set, so chat_model is used")
    if missing and available and settings.model_server != "ollama":
        # Other servers name models their own way ("qwen2.5-7b-instruct").
        print(f"       The server lists: {', '.join(sorted(available)[:12])}")

    for label, path in (
        ("episodic db", settings.db_path.parent),
        ("chroma", settings.chroma_path),
        ("workspace (the agent's files)", settings.allowed_root),
    ):
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".quietpost-write-test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            print(f"{_OK} {label} directory writable: {path}")
        except OSError as exc:
            print(f"{_FAIL} {label} directory not writable ({path}): {exc}")
            failures += 1

    if _port_free(settings.api_host, settings.api_port):
        print(f"{_OK} API port {settings.api_port} is free")
    else:
        print(f"{_WARN} API port {settings.api_port} is in use (Quietpost may already be running)")

    try:
        from quietpost.tools.builtin.web_fetch import _ssl_context

        httpx.get("https://example.com", timeout=8.0, verify=_ssl_context())
        print(f"{_OK} HTTPS works (web_fetch)")
    except Exception as exc:
        print(f"{_WARN} HTTPS check failed — web_fetch will be unavailable: {exc}")

    print()
    if failures:
        print(f"{failures} problem(s) found. Fix them and re-run `quietpost doctor`.")
    else:
        print("All checks passed. Start the API with: quietpost serve")
    return 1 if failures else 0


async def _server_models(settings: "Settings") -> set[str]:
    """The models the configured model server lists (QuietpostModelError if unreachable)."""
    from quietpost.models.factory import create_provider

    provider = await create_provider(settings)
    try:
        return set(await provider.list_models())
    finally:
        await provider.aclose()


def _reset_memory(confirmed: bool) -> int:
    """Delete the SQLite episodic store and the Chroma collections."""
    import shutil

    from quietpost.config import get_settings

    settings = get_settings()
    targets = [settings.db_path, settings.chroma_path]
    if not _port_free(settings.api_host, settings.api_port):
        print(f"{_FAIL} Stop the Quietpost API first (port {settings.api_port} is in use).")
        return 1
    if not confirmed:
        print("This permanently deletes:")
        for target in targets:
            print(f"  {target}")
        print("Re-run with --yes to confirm:  quietpost reset-memory --yes")
        return 1
    for target in targets:
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
        print(f"{_OK} deleted {target}")
    print("Memory reset. It will be recreated empty on the next `quietpost serve`.")
    return 0


def _port_free(host: str, port: int) -> bool:
    """Return True if nothing is listening on host:port."""
    probe_host = "127.0.0.1" if host in ("0.0.0.0", "", "::") else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((probe_host, port)) != 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    sys.exit(main())
