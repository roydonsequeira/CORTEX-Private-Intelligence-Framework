"""`cortex` command line: start the API and diagnose a local setup.

    cortex serve          start the API (uvicorn) with the settings from cortex.yaml
    cortex doctor         check Python, config, Ollama, models, ports, and HTTPS
    cortex reset-memory   wipe stored memory (asks for --yes)

Because this runs inside the interpreter CORTEX is installed in, it cannot pick
up a different Python's `uvicorn` from PATH — the most common reason a bare
`uvicorn cortex.api.server:create_app` fails with "No module named 'cortex'".
"""

import argparse
import os
import socket
import sys
from pathlib import Path

_OK = "[ ok ]"
_WARN = "[warn]"
_FAIL = "[FAIL]"


def main(argv: list[str] | None = None) -> int:
    """Entry point for the `cortex` console script."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(errors="replace")
    parser = argparse.ArgumentParser(prog="cortex", description="CORTEX local AI agent")
    sub = parser.add_subparsers(dest="command")

    serve = sub.add_parser("serve", help="start the CORTEX API server")
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

    from cortex.config import get_settings

    settings = get_settings()
    bind_host = host or settings.api_host
    bind_port = port or settings.api_port
    if not _port_free(bind_host, bind_port):
        print(
            f"{_FAIL} Port {bind_port} is already in use — another CORTEX (or other "
            f"server) is running. Stop it, or use: cortex serve --port {bind_port + 1}"
        )
        return 1
    print(f"CORTEX API -> http://localhost:{bind_port}  (docs: http://localhost:{bind_port}/docs)")
    uvicorn.run(
        "cortex.api.server:create_app",
        factory=True,
        host=bind_host,
        port=bind_port,
        reload=reload,
        log_level=settings.log_level.lower(),
    )
    return 0


def doctor() -> int:
    """Print a checklist of everything CORTEX needs; return non-zero on failures."""
    import httpx

    from cortex.config.settings import Settings, _resolve_config_path

    failures = 0
    print("CORTEX doctor\n")

    print(f"{_OK} Python {sys.version.split()[0]} at {sys.executable}")

    config_path = _resolve_config_path()
    if config_path:
        print(f"{_OK} Config: {config_path}")
    else:
        print(f"{_WARN} No cortex.yaml found from {Path.cwd()} upward — using defaults")
    try:
        settings = Settings()
    except Exception as exc:
        print(f"{_FAIL} Config could not be loaded: {exc}")
        return 1

    base = settings.ollama_base_url.rstrip("/")
    try:
        response = httpx.get(f"{base}/api/tags", timeout=5.0)
        response.raise_for_status()
        available = {m["name"] for m in response.json().get("models", [])}
        print(f"{_OK} Ollama reachable at {base} ({len(available)} models)")
    except Exception as exc:
        print(f"{_FAIL} Ollama not reachable at {base}: {exc}")
        print("       Fix: start the Ollama app or run `ollama serve`.")
        return failures + 1

    from cortex.api.server import missing_models

    configured = {
        "chat (ollama_model)": settings.ollama_model,
        "reasoning (reasoning_model)": settings.reasoning_model,
        "code (code_model)": settings.code_model,
        "embeddings (embed_model)": settings.embed_model,
    }
    missing = set(missing_models(settings, available))
    for role, model in configured.items():
        if model in missing:
            print(f"{_FAIL} {role}: '{model}' is not pulled.  Fix: ollama pull {model}")
            failures += 1
        else:
            print(f"{_OK} {role}: {model}")

    for label, path in (("episodic db", settings.db_path.parent), ("chroma", settings.chroma_path)):
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".cortex-write-test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            print(f"{_OK} {label} directory writable: {path}")
        except OSError as exc:
            print(f"{_FAIL} {label} directory not writable ({path}): {exc}")
            failures += 1

    if _port_free(settings.api_host, settings.api_port):
        print(f"{_OK} API port {settings.api_port} is free")
    else:
        print(f"{_WARN} API port {settings.api_port} is in use (CORTEX may already be running)")

    try:
        from cortex.tools.builtin.web_fetch import _ssl_context

        httpx.get("https://example.com", timeout=8.0, verify=_ssl_context())
        print(f"{_OK} HTTPS works (web_fetch)")
    except Exception as exc:
        print(f"{_WARN} HTTPS check failed — web_fetch will be unavailable: {exc}")

    print()
    if failures:
        print(f"{failures} problem(s) found. Fix them and re-run `cortex doctor`.")
    else:
        print("All checks passed. Start the API with: cortex serve")
    return 1 if failures else 0


def _reset_memory(confirmed: bool) -> int:
    """Delete the SQLite episodic store and the Chroma collections."""
    import shutil

    from cortex.config import get_settings

    settings = get_settings()
    targets = [settings.db_path, settings.chroma_path]
    if not _port_free(settings.api_host, settings.api_port):
        print(f"{_FAIL} Stop the CORTEX API first (port {settings.api_port} is in use).")
        return 1
    if not confirmed:
        print("This permanently deletes:")
        for target in targets:
            print(f"  {target}")
        print("Re-run with --yes to confirm:  cortex reset-memory --yes")
        return 1
    for target in targets:
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
        print(f"{_OK} deleted {target}")
    print("Memory reset. It will be recreated empty on the next `cortex serve`.")
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
