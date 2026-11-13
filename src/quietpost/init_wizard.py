"""`quietpost init`: set Quietpost up on this computer in one go.

It checks the model server, suggests models for the GPU, downloads the ones
Ollama is missing once the user agrees, and writes the quietpost.yaml that
Quietpost will read, with data and workspace folders that don't depend on where
a command is run from. A model that would run on another computer, such as one
of Ollama's cloud models, is refused before anything is downloaded.
"""

import asyncio
import json
import os
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal, TextIO

import httpx
import yaml
from platformdirs import user_data_path

from quietpost.config import settings as config_settings
from quietpost.exceptions import QuietpostConfigError, QuietpostModelError
from quietpost.models.endpoints import (
    OLLAMA_CLOUD,
    ensure_local_model_server,
    is_cloud_model_name,
    runs_elsewhere,
)
from quietpost.models.openai_compat import OpenAICompatibleProvider
from quietpost.models.profiles import (
    DOWNLOAD_SIZES_GB,
    EMBED_MODEL,
    detect_gpu_memory_gb,
    profile_for,
)
from quietpost.models.provider import (
    ModelProvider,
    OllamaProvider,
    error_reply,
    prefer_ipv4_loopback,
)

Ask = Callable[[str], str]
Say = Callable[[str], None]
Pull = Callable[[str, str, Say], None]
# One step of a download, as Ollama reports it: what it is doing, then the bytes
# done and the total when the step has them.
Progress = Callable[[str, int | None, int | None], None]
ModelServer = Literal["ollama", "openai-compatible"]

# Read for a model server started with an API key. Never written to the config.
API_KEY_ENV_VAR = "QUIETPOST_MODEL_SERVER_API_KEY"


@dataclass
class InitOptions:
    """What `quietpost init` was asked for on the command line."""

    model_server: ModelServer = "ollama"
    url: str | None = None
    chat_model: str | None = None
    embed_model: str | None = None
    yes: bool = False  # accept every suggestion without asking
    force: bool = False  # replace an existing config
    config_path: Path | None = None
    data_dir: Path | None = None
    workspace: Path | None = None


@dataclass
class Folders:
    """Where the config, the data and the workspace go."""

    config: Path
    data: Path
    workspace: Path


@dataclass
class _Setup:
    """What goes into the config, and what is left for the user to do."""

    chat_model: str
    embed_model: str
    missing: list[str]  # models still to download
    download_failed: bool = False


def folders_for(options: InitOptions) -> Folders:
    """The folders asked for, else the defaults.

    The config is the one Quietpost will read (see config_target). Data goes in the
    user's data folder. The workspace goes in the home folder, not in Documents,
    which OneDrive and iCloud often sync to the cloud.
    """
    return Folders(
        config=options.config_path or config_target(),
        data=(
            _absolute(options.data_dir)
            if options.data_dir
            else user_data_path("quietpost", appauthor=False)
        ),
        workspace=(
            _absolute(options.workspace) if options.workspace else home_folder() / "Quietpost"
        ),
    )


def config_target() -> Path:
    """The config to write: the file QUIETPOST_CONFIG (or CORTEX_CONFIG) names, since
    Quietpost reads no other while it is set; else the one in the user's config folder."""
    override = os.environ.get(config_settings._CONFIG_ENV_VAR) or os.environ.get(
        config_settings._LEGACY_CONFIG_ENV_VAR
    )
    return Path(override) if override else config_settings.user_config_file()


def home_folder() -> Path:
    """The user's home folder; a function, so tests can point it elsewhere."""
    return Path.home()


def _absolute(path: Path) -> Path:
    """A folder for the config. A relative path there is read from the config's own
    folder, not from where `quietpost init` ran, so it is made absolute."""
    return path.expanduser().resolve()


def run(
    options: InitOptions,
    ask: Ask = input,
    say: Say = print,
    folders: Folders | None = None,
    gpu_memory: Callable[[], float | None] = detect_gpu_memory_gb,
    pull: Pull | None = None,
) -> int:
    """Run the wizard; returns the exit code (130 when stopped with Ctrl+C)."""
    folders = folders or folders_for(options)
    try:
        setup = _choose_models(
            options, ask, say, folders, gpu_memory, pull or _pull_with_progress
        )
    except KeyboardInterrupt:
        say("\nStopped. No config was written.")
        return 130
    if isinstance(setup, int):
        return setup

    _write_config(folders, options.model_server, options.url, setup.chat_model, setup.embed_model)
    say(f"\n[ ok ] Wrote {folders.config}")
    say(f"       Your data stays in {folders.data}")
    say(f"       Your workspace, the only folder Quietpost's tools can read and write: {folders.workspace}")
    _warn_if_another_config_is_read(folders.config, say)
    if options.model_server == "openai-compatible" and _api_key():
        say(f"       Keep {API_KEY_ENV_VAR} set when you run `quietpost serve`.")
    from quietpost.api.web import has_bundled_ui

    first = "download the models above, then " if setup.missing else ""
    if has_bundled_ui():
        say(f"\nNext: {first}run `quietpost serve`, then open http://localhost:8000")
    else:
        say(f"\nNext: {first}run `quietpost serve`, then start the UI with `npm run dev` in ui/")
    return 1 if setup.download_failed else 0


def _choose_models(
    options: InitOptions,
    ask: Ask,
    say: Say,
    folders: Folders,
    gpu_memory: Callable[[], float | None],
    pull: Pull,
) -> _Setup | int:
    """Everything before the config is written; an exit code when it stops early."""
    say("Quietpost setup\n")
    if folders.config.exists() and not options.force:
        question = f"{folders.config} already exists. Replace it?"
        if options.yes or not _confirm(ask, say, question, False):
            say(f"Kept {folders.config}. To start over: quietpost init --force")
            return 0

    url = options.url or config_settings._DEFAULT_MODEL_URLS[options.model_server]
    server_name = "Ollama" if options.model_server == "ollama" else "The model server"
    try:
        available = asyncio.run(_list_models(options.model_server, url, _api_key()))
    except QuietpostConfigError as exc:
        say(f"[FAIL] {exc}")
        return 1
    except QuietpostModelError as exc:
        _explain_model_server_error(exc, options.model_server, server_name, url, say)
        return 1
    say(f"[ ok ] {server_name} is running at {url} ({len(available)} models)")

    if options.model_server != "ollama":
        chosen = _server_models(options, available, ask, say)
        if chosen is None or not _all_run_here(_by_option(*chosen), say):
            return 1
        return _Setup(*chosen, missing=[])

    chat_model, embed_model = _ollama_models(options, available, say, gpu_memory)
    by_option = _by_option(chat_model, embed_model)
    present = [m for m in by_option.values() if _has_model(available, m)]
    if not _all_run_here(by_option, say, url, present):
        return 1
    missing = [m for m in dict.fromkeys((chat_model, embed_model)) if not _has_model(available, m)]
    failed = False
    if missing:
        wanted = missing
        missing, failed = _download(missing, url, options, ask, say, pull)
        downloaded = [m for m in wanted if m not in missing]
        if downloaded and not _all_run_here(by_option, say, url, downloaded):
            return 1
    if missing:
        say("       Download them later with:")
        for model in missing:
            say(f"         ollama pull {model}")
    return _Setup(chat_model, embed_model, missing, failed)


def _explain_model_server_error(
    exc: QuietpostModelError, model_server: ModelServer, server_name: str, url: str, say: Say
) -> None:
    """Say why the models couldn't be listed: no answer at all, or the server's reply."""
    reply = error_reply(exc)
    if reply is not None:
        status, detail = reply
        say(f"[FAIL] {server_name} at {url} answered HTTP {status}: {detail}")
        if status in (401, 403) and model_server == "ollama":
            say(f"       Ollama itself doesn't ask for a key: is {url} your Ollama?")
        elif status in (401, 403):
            say(f"       If it needs an API key, set {API_KEY_ENV_VAR} to it and run `quietpost init` again.")
        return
    if not isinstance(exc.__cause__, httpx.ConnectError | httpx.TimeoutException):
        say(f"[FAIL] {exc}")
        return
    say(f"[FAIL] {server_name} isn't answering at {url}.")
    if model_server == "ollama":
        say("       Install Ollama from https://ollama.com/download, start it, and run")
        say("       `quietpost init` again. Using LM Studio, llama.cpp or vLLM instead?")
        say("       quietpost init --model-server openai-compatible --url http://localhost:1234/v1")
    else:
        say("       Start it, check --url, and run `quietpost init` again.")


def _ollama_models(
    options: InitOptions,
    available: list[str],
    say: Say,
    gpu_memory: Callable[[], float | None],
) -> tuple[str, str]:
    """The chat and embedding models to use with Ollama."""
    if options.chat_model:
        chat_model = options.chat_model
    else:
        gpu_gb = gpu_memory()
        profile = profile_for(gpu_gb)
        chat_model = profile.chat_model
        if gpu_gb is None:
            say(
                f"[warn] Couldn't tell how much GPU memory this computer has. Suggesting "
                f"{chat_model}: {profile.note}. Choose another with --chat-model."
            )
        else:
            say(f"[ ok ] GPU memory: {gpu_gb:g} GB, so suggesting {chat_model} ({profile.note})")
    return chat_model, options.embed_model or EMBED_MODEL


def _by_option(chat_model: str, embed_model: str) -> dict[str, str]:
    """The chosen models, by the command-line option that chooses each."""
    return {"--chat-model": chat_model, "--embed-model": embed_model}


def _all_run_here(
    by_option: dict[str, str],
    say: Say,
    ollama_url: str | None = None,
    ask_about: list[str] | None = None,
) -> bool:
    """False, after saying why, when a chosen model would run on another computer.

    A cloud model's name is enough to tell, before anything is downloaded. The
    models in ``ask_about``, ones Ollama has, are also looked up in the Ollama at
    ``ollama_url``, which knows when it passes a model on.
    """
    hosts = {model: OLLAMA_CLOUD for model in by_option.values() if is_cloud_model_name(model)}
    if ollama_url and ask_about and not hosts:
        try:
            hosts = asyncio.run(_remote_hosts(ollama_url, ask_about))
        except QuietpostModelError as exc:
            say(f"[FAIL] {exc}")
            return False
    for option, model in by_option.items():
        if model in hosts:
            say(f"[FAIL] {runs_elsewhere(model, hosts[model])}.")
            say(f"       Choose a model that runs on this computer with {option}.")
            return False
    return True


def _download(
    missing: list[str],
    url: str,
    options: InitOptions,
    ask: Ask,
    say: Say,
    pull: Pull,
) -> tuple[list[str], bool]:
    """Download the missing models into Ollama if the user agrees.

    Returns the models still missing, and whether a download failed. Only --yes or
    a "yes" counts as agreeing: with nobody to answer, nothing is downloaded.
    """
    question = f"Download {' and '.join(_with_size(m) for m in missing)} into Ollama now?"
    agreed = True if options.yes else _confirm(ask, say, question, True)
    if agreed is None:
        say("[warn] Nothing was downloaded: nobody answered (no terminal?).")
        return missing, False
    if not agreed:
        say(f"[warn] Not downloaded: {', '.join(missing)}")
        return missing, False
    for done, model in enumerate(missing):
        try:
            pull(url, model, say)
        except QuietpostModelError as exc:
            say(f"[FAIL] {exc}")
            return missing[done:], True
        say(f"[ ok ] Downloaded {model}")
    return [], False


def _with_size(model: str) -> str:
    size = DOWNLOAD_SIZES_GB.get(model)
    return f"{model} ({size:g} GB)" if size else model


def _server_models(
    options: InitOptions, available: list[str], ask: Ask, say: Say
) -> tuple[str, str] | None:
    """The chat and embedding models on an OpenAI-compatible server (None: give up).

    These servers can't download models, and they name them their own way, so
    the choice is made from what the server lists.
    """
    if not available:
        say("[FAIL] The model server lists no models. Load a chat model and an embedding")
        say("       model in it, then run `quietpost init` again.")
        return None
    chosen: list[str] = []
    for role, given, flag in (
        ("chat", options.chat_model, "--chat-model"),
        ("embedding", options.embed_model, "--embed-model"),
    ):
        if given:
            if given not in available:
                say(f"[FAIL] The server has no model '{given}'. It lists: {', '.join(available)}")
                return None
            chosen.append(given)
            continue
        if options.yes:
            say(f"[FAIL] Choose the {role} model with {flag}.")
            say(f"       The server lists: {', '.join(available)}")
            return None
        question = f"Which model is the {role} model?"
        choice = _choose(ask, say, question, available, _suggest(available, role))
        if choice is None:
            say("[FAIL] Nobody answered (no terminal?). Name the models with --chat-model")
            say("       and --embed-model.")
            return None
        chosen.append(choice)
    return chosen[0], chosen[1]


def _suggest(available: list[str], role: str) -> str:
    """Embedding models nearly always say so in their name ("nomic-embed-text").

    A cloud model, which Quietpost refuses, is never the suggestion.
    """
    local = [m for m in available if not is_cloud_model_name(m)] or available
    embedding = [m for m in local if "embed" in m.lower()]
    others = [m for m in local if m not in embedding]
    return ((embedding if role == "embedding" else others) or local)[0]


def _write_config(
    folders: Folders,
    model_server: ModelServer,
    url: str | None,
    chat_model: str,
    embed_model: str,
) -> None:
    """Write quietpost.yaml (keeping a copy of one it replaces) and create the folders."""
    for folder in (folders.data, folders.workspace):
        folder.mkdir(parents=True, exist_ok=True)
    if folders.config.exists():
        shutil.copy2(folders.config, folders.config.with_suffix(".yaml.bak"))
    folders.config.parent.mkdir(parents=True, exist_ok=True)
    folders.config.write_text(
        render_config(folders, model_server, url, chat_model, embed_model), encoding="utf-8"
    )


def render_config(
    folders: Folders,
    model_server: ModelServer,
    url: str | None,
    chat_model: str,
    embed_model: str,
) -> str:
    """The text of quietpost.yaml, with a comment for each group of settings."""

    def entries(values: dict[str, object]) -> str:
        return yaml.safe_dump(values, sort_keys=False, allow_unicode=True)

    models: dict[str, object] = {"model_server": model_server}
    if url:
        models["model_server_url"] = url
    models.update({"chat_model": chat_model, "embed_model": embed_model})
    return (
        f"# Quietpost settings, written by `quietpost init` on {date.today().isoformat()}.\n"
        "# Change them here or with QUIETPOST_<NAME> environment variables, then run\n"
        "# `quietpost doctor` to check. All settings: see the README.\n\n"
        "# Where the models run. Only this computer or your local network is allowed.\n"
        + entries(models)
        + "\n# Memory and other data, kept on this computer.\n"
        + entries(
            {
                "chroma_path": str(folders.data / "chroma"),
                "db_path": str(folders.data / "quietpost.db"),
                "task_db_path": str(folders.data / "tasks.db"),
            }
        )
        + "\n# The only folder Quietpost's tools can read and write.\n"
        + entries({"allowed_root": str(folders.workspace)})
        + "\n# Plugins of your own: a folder of .py files. Each one runs, unsandboxed, when\n"
        "# Quietpost starts, so name a folder that only you can write to.\n"
        + f"# plugins_dir: {folders.data / 'plugins'}\n"
        + "\n# Tracing needs a collector, which a normal install doesn't run.\n"
        + entries({"telemetry_enabled": False})
    )


def _warn_if_another_config_is_read(config: Path, say: Say) -> None:
    """Warn when Quietpost, started from the current folder, would read another config."""
    used = config_settings._resolve_config_path()
    if used is not None and _same_file(used, config):
        return
    if used is None:
        say(f"[warn] Quietpost won't find it by itself. Set QUIETPOST_CONFIG={config}")
        return
    say(f"[warn] Started from {Path.cwd()}, Quietpost reads {used} instead:")
    say("       a config in the current folder or a folder above it comes first. Start")
    say(f"       Quietpost from another folder, or set QUIETPOST_CONFIG={config}")


def _same_file(a: Path, b: Path) -> bool:
    try:
        return a.samefile(b)
    except OSError:
        return False


def _api_key() -> str | None:
    return os.environ.get(API_KEY_ENV_VAR) or None


async def _list_models(model_server: ModelServer, url: str, api_key: str | None = None) -> list[str]:
    """The models the server has (QuietpostConfigError for a public address)."""
    await ensure_local_model_server(url)
    provider: ModelProvider = (
        OllamaProvider(url)
        if model_server == "ollama"
        else OpenAICompatibleProvider(url, api_key=api_key)
    )
    try:
        return await provider.list_models()
    finally:
        await provider.aclose()


async def _remote_hosts(url: str, models: list[str]) -> dict[str, str]:
    """The ``models`` the Ollama at ``url`` passes on to another computer, with that computer."""
    provider = OllamaProvider(url)
    try:
        hosts: dict[str, str] = {}
        for model in dict.fromkeys(models):
            host = await provider.remote_host(model)
            if host:
                hosts[model] = host
        return hosts
    finally:
        await provider.aclose()


def _has_model(available: list[str], model: str) -> bool:
    """Ollama lists "nomic-embed-text:latest" for "nomic-embed-text"."""
    return model in available or f"{model}:latest" in available


def _pull_with_progress(url: str, model: str, say: Say) -> None:
    progress = DownloadProgress(model, sys.stdout)
    try:
        asyncio.run(pull_model(url, model, progress))
    finally:
        progress.close()


async def pull_model(
    url: str,
    model: str,
    progress: Progress,
    transport: httpx.AsyncBaseTransport | None = None,
) -> None:
    """Download ``model`` into the Ollama at ``url``, reporting each step.

    Any failure, including a lost connection, raises QuietpostModelError.
    ``transport`` replaces the network, for tests.
    """
    await ensure_local_model_server(url)
    try:
        async with (
            httpx.AsyncClient(
                base_url=prefer_ipv4_loopback(url.rstrip("/")),
                timeout=httpx.Timeout(None, connect=10.0),
                trust_env=False,
                transport=transport,
            ) as client,
            client.stream("POST", "/api/pull", json={"model": model}) as response,
        ):
            if response.is_error:
                await response.aread()
                raise QuietpostModelError(
                    f"Ollama could not download {model}: {response.text[:300]}"
                )
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                data = json.loads(line)
                if data.get("error"):
                    raise QuietpostModelError(f"Ollama could not download {model}: {data['error']}")
                progress(str(data.get("status", "")), data.get("completed"), data.get("total"))
    except (httpx.HTTPError, json.JSONDecodeError) as exc:
        raise QuietpostModelError(f"Ollama could not download {model}: {exc}") from exc


class DownloadProgress:
    """Shows a download's progress.

    On a terminal, one line is rewritten in place. Anywhere else (a pipe, a log
    file), each step and every 10% gets a line of its own, so a log stays short.
    """

    def __init__(self, model: str, stream: TextIO) -> None:
        self._model = model
        self._stream = stream
        self._terminal = stream.isatty()
        self._shown = ""
        self._step: tuple[str, int | None] | None = None

    def __call__(self, status: str, completed: int | None, total: int | None) -> None:
        text = f"{self._model}: {status}"
        tenth: int | None = None
        if total:
            done = completed or 0
            percent = done * 100 // total
            text += f" {done / 1e9:.1f} of {total / 1e9:.1f} GB ({percent}%)"
            tenth = percent // 10
        if self._terminal:
            if text != self._shown:
                self._stream.write("\r" + text.ljust(len(self._shown)))
                self._stream.flush()
                self._shown = text
        elif (status, tenth) != self._step:
            self._stream.write(text + "\n")
            self._stream.flush()
            self._step = (status, tenth)

    def close(self) -> None:
        """End the line being rewritten, so what follows starts on a new one."""
        if self._terminal and self._shown:
            self._stream.write("\n")
            self._stream.flush()


def _answer(ask: Ask, say: Say, prompt: str) -> str | None:
    """The user's answer; None when nobody can answer (no terminal, or input ended)."""
    try:
        return ask(prompt)
    except EOFError:
        say("")  # end the line the question was left on
        return None


def _confirm(ask: Ask, say: Say, question: str, default: bool) -> bool | None:
    """Yes or no, where Enter gives ``default``; None when nobody can answer."""
    answer = _answer(ask, say, f"{question} [{'Y/n' if default else 'y/N'}] ")
    if answer is None:
        return None
    answer = answer.strip().lower()
    return default if not answer else answer in ("y", "yes")


def _choose(ask: Ask, say: Say, question: str, options: list[str], suggested: str) -> str | None:
    """One of ``options``, where Enter gives ``suggested``; None when nobody can answer."""
    say(question)
    for number, option in enumerate(options, 1):
        say(f"  {number}. {option}")
    default = options.index(suggested) + 1
    while True:
        answer = _answer(ask, say, f"Number [{default}]: ")
        if answer is None:
            return None
        answer = answer.strip()
        if not answer:
            return suggested
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return options[int(answer) - 1]
        say(f"Type a number from 1 to {len(options)}.")
