"""`quietpost` command line: start the API and diagnose a local setup.

    quietpost init           set up models, settings and folders (asks before downloading)
    quietpost serve          start the API (uvicorn) with the settings from quietpost.yaml
    quietpost doctor         check Python, config, the model server, models and ports
                             (--online: HTTPS too, which connects to example.com)
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quietpost.config.settings import Settings

_OK = "[ ok ]"
_WARN = "[warn]"
_FAIL = "[FAIL]"
_SKIP = "[skip]"
# The one check that leaves this computer, so doctor runs it only with --online.
_HTTPS_CHECK_URL = "https://example.com"
_CONFIG_VARIABLES = ("QUIETPOST_CONFIG", "CORTEX_CONFIG")


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

    init = sub.add_parser(
        "init", help="set Quietpost up: models, settings and folders, in one go"
    )
    init.add_argument(
        "--model-server",
        choices=["ollama", "openai-compatible"],
        default="ollama",
        help="where the models run (default: ollama)",
    )
    init.add_argument("--url", default=None, help="the model server's address, if not the usual one")
    init.add_argument("--chat-model", default=None, help="the chat model (default: chosen for your GPU)")
    init.add_argument("--embed-model", default=None, help="the embedding model")
    init.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="where memory and other data go (default: your user data folder)",
    )
    init.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="the only folder Quietpost's tools can read and write (default: Quietpost "
        "in your home folder)",
    )
    init.add_argument("--yes", action="store_true", help="accept every suggestion without asking")
    init.add_argument("--force", action="store_true", help="replace an existing config")

    doctor_parser = sub.add_parser("doctor", help="diagnose the local setup")
    doctor_parser.add_argument(
        "--online",
        action="store_true",
        help=f"also check HTTPS for web_fetch, by connecting to {_HTTPS_CHECK_URL}",
    )

    reset = sub.add_parser(
        "reset-memory", help="delete all stored memory (episodic, semantic, procedural)"
    )
    reset.add_argument("--yes", action="store_true", help="confirm the deletion")

    args = parser.parse_args(argv)
    if args.command == "serve":
        return _serve(args.host, args.port, args.reload)
    if args.command == "init":
        from quietpost import init_wizard

        return init_wizard.run(
            init_wizard.InitOptions(
                model_server=args.model_server,
                url=args.url,
                chat_model=args.chat_model,
                embed_model=args.embed_model,
                yes=args.yes,
                force=args.force,
                data_dir=args.data_dir,
                workspace=args.workspace,
            )
        )
    if args.command == "doctor":
        return doctor(online=args.online)
    if args.command == "reset-memory":
        return _reset_memory(args.yes)
    parser.print_help()
    return 0


def _serve(host: str | None, port: int | None, reload: bool) -> int:
    """Run uvicorn in this interpreter with the configured host and port."""
    import uvicorn

    from quietpost.config import get_settings
    from quietpost.exceptions import QuietpostConfigError

    bad_config_file = _missing_config_file()
    if bad_config_file:
        print(f"{_FAIL} {bad_config_file}")
        return 1
    settings = get_settings()
    no_config = _no_config(settings)
    if no_config:
        level, lines = no_config
        print(f"{level} " + "\n       ".join(lines))
        if level == _FAIL:
            return 1
    bind_host = host or settings.api_host
    bind_port = port or settings.api_port
    try:
        asyncio.run(_check_model_server(settings))
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
    from quietpost.api.web import has_bundled_ui

    if has_bundled_ui() and not settings.api_key:
        print(f"Quietpost -> http://localhost:{bind_port}  (open it in your browser)")
    else:
        print(f"Quietpost API -> http://localhost:{bind_port}  (docs: http://localhost:{bind_port}/docs)")
        if has_bundled_ui():
            print(f"{_WARN} The UI can't send an API key yet, so it doesn't work while api_key is set.")
    uvicorn.run(
        "quietpost.api.server:create_app",
        factory=True,
        host=bind_host,
        port=bind_port,
        reload=reload,
        log_level=settings.log_level.lower(),
    )
    return 0


def _missing_config_file() -> str | None:
    """Say so when QUIETPOST_CONFIG (or CORTEX_CONFIG) names a file that doesn't exist.

    Without this check the defaults apply silently, which hides a typo in the path.
    """
    for variable in _CONFIG_VARIABLES:
        value = os.environ.get(variable)
        if value:
            return None if Path(value).is_file() else f"{variable} names {value}, which doesn't exist."
    return None


def _no_config(settings: "Settings") -> tuple[str, list[str]] | None:
    """What to say when no config file is read: (_FAIL or _WARN, lines), or None.

    The defaults keep memory and the workspace under the folder Quietpost was
    started in, so a first start without a config would leave them there, and a
    later `quietpost init` would point at new, empty folders. A first start is
    therefore refused with how to set up. A setup that QUIETPOST_* (or CORTEX_*)
    variables configure, or that already has memory where the defaults point, as
    a 1.x install can, starts as before; nothing is moved.
    """
    from quietpost.config.settings import _resolve_config_path, user_config_file

    if _resolve_config_path() is not None:
        return None
    looked = f"no quietpost.yaml in {Path.cwd()} or above it, or in {user_config_file().parent}"
    if _configured_by_environment() or _has_memory(settings):
        return _WARN, [
            f"No config file ({looked}), so the defaults and any QUIETPOST_* settings apply.",
            f"Memory: {settings.db_path.parent.resolve()}",
            f"Workspace: {settings.allowed_root.resolve()}",
        ]
    return _FAIL, [
        f"No config found: {looked}.",
        "Run `quietpost init` first. It picks the models and keeps memory in your user data",
        "folder, whichever folder you start Quietpost in. To keep memory and the workspace",
        "in this folder instead, create an empty quietpost.yaml here.",
    ]


def _has_memory(settings: "Settings") -> bool:
    """Whether memory is stored where the settings point (empty folders don't count)."""
    chroma = settings.chroma_path
    return settings.db_path.is_file() or (chroma.is_dir() and any(chroma.iterdir()))


def _configured_by_environment() -> bool:
    """Whether a QUIETPOST_* variable, or a CORTEX_* one from 1.x, sets a setting."""
    from quietpost.config.settings import Settings

    names = {
        f"{prefix}{name}".upper()
        for prefix in ("QUIETPOST_", "CORTEX_")
        for name in Settings.model_fields
    }
    return any(variable.upper() in names for variable in os.environ)


async def _check_model_server(settings: "Settings") -> None:
    """Refuse a model server or a model that isn't on this computer or the local network.

    Done before the server starts, so the reason is printed plainly. The server
    checks again as it starts, and the Ollama client before each model's first
    request.
    """
    from quietpost.models.factory import create_provider, ensure_models_run_here

    provider = await create_provider(settings)
    try:
        if await provider.health_check():
            await ensure_models_run_here(provider, settings)
    finally:
        await provider.aclose()


def doctor(online: bool = False) -> int:
    """Print a checklist of everything Quietpost needs; return non-zero on failures.

    Every check stays on this computer and the local network unless ``online`` is
    set, which adds the HTTPS check against example.com.
    """
    import httpx

    from quietpost.config.settings import Settings, _resolve_config_path
    from quietpost.exceptions import QuietpostConfigError, QuietpostModelError

    failures = 0
    print("Quietpost doctor\n")

    print(f"{_OK} Python {sys.version.split()[0]} at {sys.executable}")

    bad_config_file = _missing_config_file()
    if bad_config_file:
        print(f"{_FAIL} {bad_config_file}")
        return 1
    config_path = _resolve_config_path()
    if config_path:
        print(f"{_OK} Config: {config_path}")
    try:
        settings = Settings()
    except Exception as exc:
        print(f"{_FAIL} Config could not be loaded: {exc}")
        return 1
    no_config = _no_config(settings)
    if no_config:
        level, lines = no_config
        print(f"{level} " + "\n       ".join(lines))
        if level == _FAIL:
            failures += 1

    from quietpost.api.server import missing_model_fix, missing_models, start_server_fix
    from quietpost.models.endpoints import model_refusal
    from quietpost.models.provider import error_reply

    server = "Ollama" if settings.model_server == "ollama" else "Model server"
    try:
        report = asyncio.run(_ask_model_server(settings))
        available = report.models
        print(f"{_OK} {server} reachable at {settings.model_url} ({len(available)} models)")
    except QuietpostConfigError as exc:
        print(f"{_FAIL} {exc}")
        return failures + 1
    except QuietpostModelError as exc:
        reply = error_reply(exc)
        if reply is None:
            print(f"{_FAIL} {server} not reachable at {settings.model_url}: {exc}")
            print(f"       Fix: {start_server_fix(settings)}.")
            return failures + 1
        status, detail = reply
        print(f"{_FAIL} {server} at {settings.model_url} answered HTTP {status}: {detail}")
        if status in (401, 403) and settings.model_server == "ollama":
            print(f"       Ollama itself doesn't ask for a key: is {settings.model_url} your Ollama?")
        elif status in (401, 403):
            print("       Fix: set QUIETPOST_MODEL_SERVER_API_KEY (or model_server_api_key) to its API key.")
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
        if model in report.elsewhere:
            print(f"{_FAIL} {model_refusal(setting, model, report.elsewhere[model])}")
            failures += 1
        elif model in missing:
            print(f"{_FAIL} {setting}: '{model}' is missing.  Fix: {missing_model_fix(settings, model)}")
            failures += 1
        elif model in report.unchecked:
            print(f"{_FAIL} {setting}: Ollama didn't say where '{model}' runs, so Quietpost won't use it.")
            print(f"       {report.unchecked[model]}")
            failures += 1
        else:
            print(f"{_OK} {setting}: {model}")
    if unset:
        print(f"       {', '.join(unset)}: not set, so chat_model is used")
    if missing and available and settings.model_server != "ollama":
        # Other servers name models their own way ("qwen2.5-7b-instruct").
        print(f"       The server lists: {', '.join(sorted(available)[:12])}")
    if settings.model_server == "ollama":
        _print_cloud_switch(report.cloud_off)

    folders: tuple[tuple[str, Path], ...] = (
        ("episodic db", settings.db_path.parent),
        ("chroma", settings.chroma_path),
        ("workspace (the agent's files)", settings.allowed_root),
    )
    if no_config and no_config[0] == _FAIL:
        # The check creates the folders, and then they would be in the start folder.
        print(f"{_SKIP} Folders: not checked until there's a config, so none is created here.")
        folders = ()
    for label, path in folders:
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

    if online:
        try:
            from quietpost.tools.builtin.web_fetch import _ssl_context

            httpx.get(_HTTPS_CHECK_URL, timeout=8.0, verify=_ssl_context())
            print(f"{_OK} HTTPS works for web_fetch (connected to {_HTTPS_CHECK_URL})")
        except Exception as exc:
            print(f"{_WARN} HTTPS check against {_HTTPS_CHECK_URL} failed, so web_fetch won't work: {exc}")
    else:
        print(f"{_SKIP} HTTPS for web_fetch: not checked, because the check connects to example.com.")
        print("       Run `quietpost doctor --online` to include it.")

    print()
    if failures:
        print(f"{failures} problem(s) found. Fix them and re-run `quietpost doctor`.")
    else:
        print("All checks passed. Start the API with: quietpost serve")
    return 1 if failures else 0


@dataclass
class _ServerReport:
    """What `doctor` learned from the model server."""

    models: set[str]
    # Configured models Ollama passes on to another computer, with that computer.
    elsewhere: dict[str, str] = field(default_factory=dict)
    # Configured models Ollama couldn't be asked about, with what went wrong.
    unchecked: dict[str, str] = field(default_factory=dict)
    # Whether Ollama's own switch for its cloud models is off; None when unknown.
    cloud_off: bool | None = None


async def _ask_model_server(settings: "Settings") -> _ServerReport:
    """The models the server lists and, from Ollama, where the configured ones run.

    QuietpostModelError if the server can't be asked; QuietpostConfigError for a
    public server or a cloud model's name, before anything is asked.
    """
    from quietpost.exceptions import QuietpostModelError
    from quietpost.models.factory import create_provider
    from quietpost.models.provider import OllamaProvider

    provider = await create_provider(settings)
    try:
        report = _ServerReport(set(await provider.list_models()))
        if isinstance(provider, OllamaProvider):
            for model in dict.fromkeys(settings.configured_models.values()):
                try:
                    host = await provider.remote_host(model)
                except QuietpostModelError as exc:
                    # Shown only for a model Ollama lists: a missing one is reported as missing.
                    report.unchecked[model] = str(exc)
                    continue
                if host:
                    report.elsewhere[model] = host
            report.cloud_off = await provider.cloud_switched_off()
        return report
    finally:
        await provider.aclose()


def _print_cloud_switch(cloud_off: bool | None) -> None:
    """Recommend Ollama's own switch for its cloud models, a second lock behind Quietpost's."""
    if cloud_off:
        print(f"{_OK} Ollama's cloud models are switched off")
    elif cloud_off is None:
        print(f"{_WARN} Ollama didn't say whether its cloud models are switched off. Ollama 0.16.2")
        print("       and later can switch them off with OLLAMA_NO_CLOUD=1. Quietpost won't use them.")
    else:
        print(f"{_WARN} Ollama can also run models on ollama.com. Quietpost won't use them; to switch")
        print("       them off in Ollama too, set OLLAMA_NO_CLOUD=1 for Ollama and restart it.")


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
