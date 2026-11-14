"""`quietpost init`: models for the GPU, downloads with consent, and a config that works."""

import io
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

from quietpost import cli, init_wizard
from quietpost.config import settings as settings_module
from quietpost.config.settings import Settings
from quietpost.exceptions import QuietpostConfigError, QuietpostModelError
from quietpost.init_wizard import Folders, InitOptions
from tests.unit._fake_ollama import OLLAMA_COM, FakeOllama, route_ollama


class _Session:
    """Scripted answers and everything the wizard said.

    When the answers run out, asking raises EOFError, as input() does with no
    terminal. An exception among the answers is raised in its turn.
    """

    def __init__(self, *answers: str | BaseException) -> None:
        self.answers = list(answers)
        self.questions: list[str] = []
        self.lines: list[str] = []
        self.pulled: list[str] = []
        self.failing_download: str | None = None

    def ask(self, question: str) -> str:
        self.questions.append(question)
        if not self.answers:
            raise EOFError
        answer = self.answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    def say(self, line: str) -> None:
        self.lines.append(line)

    def pull(self, url: str, model: str, say: Any) -> None:
        if model == self.failing_download:
            raise QuietpostModelError(f"Ollama could not download {model}: no space left on device")
        self.pulled.append(model)

    @property
    def output(self) -> str:
        return "\n".join(self.lines)


def _folders(root: Path) -> Folders:
    """With root=tmp_path, the config is the user config that conftest sets up."""
    return Folders(
        config=root / "user-config" / "quietpost.yaml",
        data=root / "data",
        workspace=root / "home" / "Quietpost",
    )


def _server_lists(
    monkeypatch: pytest.MonkeyPatch,
    models: list[str] | Exception,
    elsewhere: dict[str, str] | None = None,
) -> list[str | None]:
    """Fake the model server's list of models; returns the API key of each call.

    Ollama passes each model in ``elsewhere`` on to the host given for it, and runs
    every other model itself.
    """
    keys: list[str | None] = []

    async def list_models(model_server: str, url: str, api_key: str | None = None) -> list[str]:
        keys.append(api_key)
        if isinstance(models, Exception):
            raise models
        return models

    async def remote_hosts(url: str, asked: list[str]) -> dict[str, str]:
        hosts = elsewhere or {}
        return {model: hosts[model] for model in asked if model in hosts}

    monkeypatch.setattr(init_wizard, "_list_models", list_models)
    monkeypatch.setattr(init_wizard, "_remote_hosts", remote_hosts)
    return keys


def _server_error(cause: Exception) -> QuietpostModelError:
    """A failed model list as the providers raise it, wrapping what went wrong."""
    error = QuietpostModelError(f"Failed to list the models: {cause}")
    error.__cause__ = cause
    return error


def _http_error(status: int, body: str) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "http://localhost:1234/v1/models")
    response = httpx.Response(status, text=body, request=request)
    return httpx.HTTPStatusError(f"HTTP {status}", request=request, response=response)


def _run(root: Path, session: _Session, gpu_gb: float | None = 6.0, **options: Any) -> int:
    return init_wizard.run(
        InitOptions(**options),
        ask=session.ask,
        say=session.say,
        folders=_folders(root),
        gpu_memory=lambda: gpu_gb,
        pull=session.pull,
    )


def test_ollama_setup_suggests_a_model_downloads_what_is_missing_and_writes_the_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _server_lists(monkeypatch, ["qwen2.5:7b"])
    session = _Session("")  # Enter: yes, download

    assert _run(tmp_path, session) == 0

    assert "suggesting qwen2.5:7b" in session.output
    assert session.questions == ["Download nomic-embed-text (0.3 GB) into Ollama now? [Y/n] "]
    assert session.pulled == ["nomic-embed-text"]
    folders = _folders(tmp_path)
    config = yaml.safe_load(folders.config.read_text(encoding="utf-8"))
    assert config["model_server"] == "ollama"
    assert config["chat_model"] == "qwen2.5:7b"
    assert config["embed_model"] == "nomic-embed-text"
    assert config["telemetry_enabled"] is False
    assert Path(config["db_path"]) == folders.data / "quietpost.db"
    assert Path(config["allowed_root"]) == folders.workspace
    assert folders.workspace.is_dir()
    # Every .py file in a plugins folder runs at start, so init names none.
    assert "plugins_dir" not in config
    assert not (folders.data / "plugins").exists()
    assert not [line for line in session.lines if line.startswith("[warn]")]


def test_the_written_config_loads_as_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _server_lists(monkeypatch, ["qwen2.5:7b", "nomic-embed-text:latest"])
    assert _run(tmp_path, _Session(), yes=True) == 0
    folders = _folders(tmp_path)
    for name in ("QUIETPOST_DB_PATH", "QUIETPOST_CHROMA_PATH"):
        monkeypatch.delenv(name)
    monkeypatch.setenv("QUIETPOST_CONFIG", str(folders.config))

    settings = Settings()

    assert settings.chat_model == "qwen2.5:7b"
    assert settings.model_url == "http://localhost:11434"
    assert settings.db_path == folders.data / "quietpost.db"
    assert settings.chroma_path == folders.data / "chroma"
    assert settings.allowed_root == folders.workspace
    assert settings.plugins_dir is None
    assert settings.telemetry_enabled is False


def test_the_model_follows_the_gpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _server_lists(monkeypatch, [])
    small = _Session("n")
    assert _run(tmp_path / "a", small, gpu_gb=4.0) == 0
    unknown = _Session("n")
    assert _run(tmp_path / "b", unknown, gpu_gb=None) == 0

    assert "suggesting qwen2.5:7b (the tested model; with less GPU memory" in small.output
    assert "Couldn't tell how much GPU memory" in unknown.output
    assert "Suggesting qwen2.5:7b: the tested model" in unknown.output


def test_declining_the_download_still_writes_the_config_and_says_how_to_get_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _server_lists(monkeypatch, [])
    session = _Session("n")

    assert _run(tmp_path, session, chat_model="llama3.1:8b") == 0

    # The size is shown where it is known.
    assert session.questions == [
        "Download llama3.1:8b and nomic-embed-text (0.3 GB) into Ollama now? [Y/n] "
    ]
    assert session.pulled == []
    assert "         ollama pull llama3.1:8b" in session.lines
    assert "         ollama pull nomic-embed-text" in session.lines
    assert "Next: download the models above, then run `quietpost serve`" in session.output
    assert _folders(tmp_path).config.is_file()


def test_a_failed_download_still_writes_the_config_but_exits_with_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _server_lists(monkeypatch, [])
    session = _Session("")  # Enter: yes, download
    session.failing_download = "nomic-embed-text"

    assert _run(tmp_path, session) == 1

    assert session.questions == [
        "Download qwen2.5:7b (4.7 GB) and nomic-embed-text (0.3 GB) into Ollama now? [Y/n] "
    ]
    assert session.pulled == ["qwen2.5:7b"]
    assert "[FAIL] Ollama could not download nomic-embed-text" in session.output
    assert "         ollama pull nomic-embed-text" in session.lines
    assert "         ollama pull qwen2.5:7b" not in session.lines
    assert _folders(tmp_path).config.is_file()


def test_with_no_terminal_nothing_is_downloaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No answer is not a yes: when input() raises EOFError, nothing is downloaded."""
    _server_lists(monkeypatch, ["qwen2.5:7b"])
    session = _Session()  # no answers: asking raises EOFError

    assert _run(tmp_path, session) == 0

    assert session.pulled == []
    warning = "[warn] Nothing was downloaded: nobody answered (no terminal?)."
    assert session.lines[session.lines.index(warning) - 1] == ""  # not on the question's line
    assert "         ollama pull nomic-embed-text" in session.lines
    assert _folders(tmp_path).config.is_file()


def test_ctrl_c_stops_without_writing_a_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _server_lists(monkeypatch, ["qwen2.5:7b"])
    session = _Session(KeyboardInterrupt())

    assert _run(tmp_path, session) == 130

    assert "Stopped. No config was written." in session.output
    assert not _folders(tmp_path).config.exists()


def test_an_existing_config_is_kept_unless_replacing_it_was_asked_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _server_lists(monkeypatch, ["qwen2.5:7b", "nomic-embed-text:latest"])
    folders = _folders(tmp_path)
    folders.config.parent.mkdir(parents=True)
    folders.config.write_text("chat_model: mine\n", encoding="utf-8")

    assert _run(tmp_path, _Session(), yes=True) == 0
    assert folders.config.read_text(encoding="utf-8") == "chat_model: mine\n"
    assert _run(tmp_path, _Session()) == 0  # nobody to answer "Replace it?"
    assert folders.config.read_text(encoding="utf-8") == "chat_model: mine\n"

    assert _run(tmp_path, _Session(), yes=True, force=True) == 0
    assert "qwen2.5:7b" in folders.config.read_text(encoding="utf-8")
    backup = folders.config.with_suffix(".yaml.bak")
    assert backup.read_text(encoding="utf-8") == "chat_model: mine\n"


def test_when_ollama_is_not_running_it_says_how_to_get_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _server_lists(monkeypatch, _server_error(httpx.ConnectError("connection refused")))
    session = _Session()

    assert _run(tmp_path, session) == 1
    assert "[FAIL] Ollama isn't answering at http://localhost:11434." in session.output
    assert "https://ollama.com/download" in session.output
    assert not _folders(tmp_path).config.exists()


def test_a_server_that_refuses_shows_its_reply_and_how_to_give_the_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 401 comes from a server that is running: "isn't answering" would mislead."""
    _server_lists(monkeypatch, _server_error(_http_error(401, '{"error": "Invalid API key"}')))
    session = _Session()

    assert _run(tmp_path, session, model_server="openai-compatible") == 1

    assert (
        "[FAIL] The model server at http://localhost:1234/v1 answered HTTP 401: "
        '{"error": "Invalid API key"}'
    ) in session.output
    assert "set QUIETPOST_MODEL_SERVER_API_KEY to it" in session.output
    assert "isn't answering" not in session.output


def test_any_other_model_server_error_is_shown_as_it_is(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _server_lists(monkeypatch, _server_error(ValueError("the reply was not JSON")))
    session = _Session()

    assert _run(tmp_path, session, model_server="openai-compatible") == 1
    assert "[FAIL] Failed to list the models: the reply was not JSON" in session.output
    assert "isn't answering" not in session.output


def test_the_api_key_comes_from_the_environment_and_is_never_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    keys = _server_lists(monkeypatch, ["qwen2.5-7b-instruct", "nomic-embed-text-v1.5"])
    monkeypatch.setenv("QUIETPOST_MODEL_SERVER_API_KEY", "a-test-key")
    session = _Session()

    code = _run(
        tmp_path,
        session,
        model_server="openai-compatible",
        chat_model="qwen2.5-7b-instruct",
        embed_model="nomic-embed-text-v1.5",
    )

    assert code == 0
    assert keys == ["a-test-key"]
    assert "a-test-key" not in _folders(tmp_path).config.read_text(encoding="utf-8")
    assert "Keep QUIETPOST_MODEL_SERVER_API_KEY set when you run `quietpost serve`." in (
        session.output
    )


def test_a_public_model_server_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _server_lists(monkeypatch, QuietpostConfigError("8.8.8.8 is a public internet address"))
    session = _Session()

    assert _run(tmp_path, session, url="http://8.8.8.8:11434") == 1
    assert "public internet address" in session.output
    assert not _folders(tmp_path).config.exists()


def test_an_openai_compatible_server_suggests_a_chat_model_that_is_not_for_embedding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Listed first, the embedding model used to be the suggested chat model too.
    _server_lists(monkeypatch, ["text-embedding-nomic-embed-text-v1.5", "qwen2.5-7b-instruct"])
    session = _Session("", "")  # Enter twice: take both suggestions

    assert _run(tmp_path, session, model_server="openai-compatible") == 0

    config = yaml.safe_load(_folders(tmp_path).config.read_text(encoding="utf-8"))
    assert config["model_server"] == "openai-compatible"
    assert config["chat_model"] == "qwen2.5-7b-instruct"
    assert config["embed_model"] == "text-embedding-nomic-embed-text-v1.5"
    assert session.pulled == []  # these servers can't download models


def test_an_openai_compatible_server_takes_the_number_of_a_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _server_lists(monkeypatch, ["text-embedding-nomic-embed-text-v1.5", "qwen2.5-7b-instruct"])
    session = _Session("x", "2", "1")

    assert _run(tmp_path, session, model_server="openai-compatible") == 0

    assert "Type a number from 1 to 2." in session.output
    config = yaml.safe_load(_folders(tmp_path).config.read_text(encoding="utf-8"))
    assert config["chat_model"] == "qwen2.5-7b-instruct"
    assert config["embed_model"] == "text-embedding-nomic-embed-text-v1.5"


def test_an_openai_compatible_server_needs_the_models_named_when_not_asking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _server_lists(monkeypatch, ["qwen2.5-7b-instruct"])
    session = _Session()

    assert _run(tmp_path, session, model_server="openai-compatible", yes=True) == 1
    assert "--chat-model" in session.output
    assert "qwen2.5-7b-instruct" in session.output


def test_an_openai_compatible_server_with_no_terminal_says_to_name_the_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _server_lists(monkeypatch, ["qwen2.5-7b-instruct", "nomic-embed-text-v1.5"])
    session = _Session()  # no answers: asking raises EOFError

    assert _run(tmp_path, session, model_server="openai-compatible") == 1
    assert "Nobody answered (no terminal?). Name the models with --chat-model" in session.output
    assert not _folders(tmp_path).config.exists()


@pytest.mark.parametrize(
    "ollama_has",
    [
        {"qwen2.5:7b": None},
        {"nomic-embed-text:latest": None},  # the other model is there to ask about
        {"gpt-oss:120b-cloud": OLLAMA_COM, "nomic-embed-text:latest": None},  # pulled before
    ],
)
def test_a_cloud_model_is_refused_before_anything_is_downloaded(
    ollama_has: dict[str, str | None], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The name is enough, and asking Ollama about it would itself reach ollama.com."""
    fake = FakeOllama(ollama_has)
    route_ollama(monkeypatch, init_wizard, fake.transport())
    session = _Session("")  # would say yes to a download

    assert _run(tmp_path, session, chat_model="gpt-oss:120b-cloud") == 1

    assert (
        "[FAIL] Ollama passes gpt-oss:120b-cloud on to ollama.com instead of running it on this "
        "computer, so everything Quietpost sent it would leave this computer."
    ) in session.lines
    assert "       Choose a model that runs on this computer with --chat-model." in session.lines
    assert session.questions == []
    assert session.pulled == []
    assert fake.paths() == ["/api/tags"]
    assert not _folders(tmp_path).config.exists()


def test_a_model_ollama_passes_on_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cloud model copied under another name keeps running on ollama.com."""
    fake = FakeOllama({"my-assistant": OLLAMA_COM, "nomic-embed-text:latest": None})
    route_ollama(monkeypatch, init_wizard, fake.transport())
    session = _Session()

    assert _run(tmp_path, session, chat_model="my-assistant", yes=True) == 1

    assert f"[FAIL] Ollama passes my-assistant on to {OLLAMA_COM} instead" in session.output
    assert "Choose a model that runs on this computer with --chat-model." in session.output
    assert fake.requests == [
        ("/api/tags", ""),
        ("/api/show", "my-assistant"),
        ("/api/show", "nomic-embed-text"),
    ]
    assert not _folders(tmp_path).config.exists()


def test_a_model_ollama_will_not_describe_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With its cloud switched off, Ollama won't say where a cloud-backed model runs."""
    fake = FakeOllama({"my-assistant": OLLAMA_COM, "nomic-embed-text": None}, cloud_off=True)
    route_ollama(monkeypatch, init_wizard, fake.transport())
    session = _Session()

    assert _run(tmp_path, session, chat_model="my-assistant", yes=True) == 1

    assert (
        "[FAIL] Ollama returned HTTP 403 for model 'my-assistant': ollama cloud is disabled"
    ) in session.output
    assert not _folders(tmp_path).config.exists()


def test_a_downloaded_model_that_ollama_passes_on_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only Ollama can tell, once it has the model."""
    _server_lists(monkeypatch, ["qwen2.5:7b"], elsewhere={"someone/assistant": OLLAMA_COM})
    session = _Session("")  # Enter: yes, download

    assert _run(tmp_path, session, chat_model="someone/assistant") == 1

    assert session.pulled == ["someone/assistant", "nomic-embed-text"]
    refusal = f"[FAIL] Ollama passes someone/assistant on to {OLLAMA_COM} instead of running it"
    assert session.output.index("[ ok ] Downloaded someone/assistant") < (
        session.output.index(refusal)
    )
    assert not _folders(tmp_path).config.exists()


def test_models_that_run_here_are_checked_and_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeOllama({"qwen2.5:7b": None, "nomic-embed-text:latest": None})
    route_ollama(monkeypatch, init_wizard, fake.transport())

    assert _run(tmp_path, _Session(), yes=True) == 0

    assert fake.paths() == ["/api/tags", "/api/show", "/api/show"]
    config = yaml.safe_load(_folders(tmp_path).config.read_text(encoding="utf-8"))
    assert (config["chat_model"], config["embed_model"]) == ("qwen2.5:7b", "nomic-embed-text")


def test_another_server_never_suggests_a_cloud_model_and_refuses_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ollama's OpenAI-compatible API lists its cloud models along with the rest."""
    _server_lists(monkeypatch, ["gpt-oss:120b-cloud", "qwen2.5:7b", "nomic-embed-text:latest"])

    suggested = _Session("", "")  # Enter twice: take both suggestions
    assert _run(tmp_path / "a", suggested, model_server="openai-compatible") == 0
    assert suggested.questions == ["Number [2]: ", "Number [3]: "]

    chosen = _Session("1", "")  # the cloud model as the chat model
    assert _run(tmp_path / "b", chosen, model_server="openai-compatible") == 1
    assert "[FAIL] Ollama passes gpt-oss:120b-cloud on to ollama.com" in chosen.output
    assert "Choose a model that runs on this computer with --chat-model." in chosen.output
    assert not _folders(tmp_path / "b").config.exists()


def test_a_401_at_ollamas_address_is_not_blamed_on_a_missing_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ollama asks for no key, so whatever answered there isn't a plain Ollama."""
    _server_lists(monkeypatch, _server_error(_http_error(401, "Unauthorized")))
    session = _Session()

    assert _run(tmp_path, session) == 1

    assert "[FAIL] Ollama at http://localhost:11434 answered HTTP 401: Unauthorized" in (
        session.output
    )
    assert "Ollama itself doesn't ask for a key: is http://localhost:11434 your Ollama?" in (
        session.output
    )
    assert "QUIETPOST_MODEL_SERVER_API_KEY" not in session.output


@pytest.mark.asyncio
async def test_remote_hosts_asks_ollama_once_about_each_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeOllama({"my-assistant": OLLAMA_COM, "qwen2.5:7b": None})
    route_ollama(monkeypatch, init_wizard, fake.transport())

    hosts = await init_wizard._remote_hosts(
        "http://localhost:11434", ["my-assistant", "qwen2.5:7b", "my-assistant"]
    )

    assert hosts == {"my-assistant": OLLAMA_COM}
    assert fake.requests == [("/api/show", "my-assistant"), ("/api/show", "qwen2.5:7b")]


def test_init_writes_the_config_that_quietpost_config_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """While QUIETPOST_CONFIG (or CORTEX_CONFIG) is set, Quietpost reads no other file."""
    assert init_wizard.folders_for(InitOptions()).config == settings_module.user_config_file()

    monkeypatch.setenv("CORTEX_CONFIG", str(tmp_path / "old.yaml"))
    assert init_wizard.folders_for(InitOptions()).config == tmp_path / "old.yaml"

    monkeypatch.setenv("QUIETPOST_CONFIG", str(tmp_path / "elsewhere" / "quietpost.yaml"))
    assert init_wizard.folders_for(InitOptions()).config == (
        tmp_path / "elsewhere" / "quietpost.yaml"
    )


def test_it_warns_when_a_config_nearer_the_current_folder_would_be_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _server_lists(monkeypatch, ["qwen2.5:7b", "nomic-embed-text:latest"])
    nearer = Path.cwd() / "quietpost.yaml"  # conftest runs each test in tmp_path
    nearer.write_text("api_port: 9500\n", encoding="utf-8")
    session = _Session()

    assert _run(tmp_path, session, yes=True) == 0

    assert f"[warn] Started from {Path.cwd()}, Quietpost reads {nearer} instead:" in (
        session.output
    )
    assert f"set QUIETPOST_CONFIG={_folders(tmp_path).config}" in session.output


def test_by_default_init_uses_the_user_folders_and_quietpost_in_the_home_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`quietpost init` with no folders given. conftest puts the user's folders in tmp_path."""
    _server_lists(monkeypatch, ["qwen2.5:7b", "nomic-embed-text:latest"])

    assert cli.main(["init", "--yes", "--chat-model", "qwen2.5:7b"]) == 0

    config_file = settings_module.user_config_file()
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    assert Path(config["db_path"]) == tmp_path / "user-data" / "quietpost.db"
    assert Path(config["allowed_root"]) == tmp_path / "home" / "Quietpost"
    assert (tmp_path / "home" / "Quietpost").is_dir()
    assert f"[ ok ] Wrote {config_file}" in capsys.readouterr().out


def test_folders_given_on_the_command_line_are_written_as_absolute_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A relative path in the config would be read from the config's own folder."""
    _server_lists(monkeypatch, ["qwen2.5:7b", "nomic-embed-text:latest"])
    args = ["init", "--yes", "--chat-model", "qwen2.5:7b"]

    assert cli.main([*args, "--data-dir", "mine/data", "--workspace", "mine/files"]) == 0

    config = yaml.safe_load(settings_module.user_config_file().read_text(encoding="utf-8"))
    assert Path(config["db_path"]) == (tmp_path / "mine" / "data" / "quietpost.db").resolve()
    assert Path(config["allowed_root"]) == (tmp_path / "mine" / "files").resolve()
    assert (tmp_path / "mine" / "files").is_dir()


@pytest.mark.asyncio
async def test_pull_model_reports_each_step_and_raises_ollama_errors() -> None:
    lines: list[dict[str, Any]] = [
        {"status": "pulling manifest"},
        {"status": "pulling 6a0746a1ec1a", "total": 4_000_000_000, "completed": 1_000_000_000},
        {"status": "success"},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content) == {"model": "qwen2.5:7b"}
        body = "\n".join(json.dumps(line) for line in lines)
        return httpx.Response(200, content=body.encode())

    steps: list[tuple[str, int | None, int | None]] = []

    def record(status: str, completed: int | None, total: int | None) -> None:
        steps.append((status, completed, total))

    transport = httpx.MockTransport(handler)
    await init_wizard.pull_model("http://localhost:11434", "qwen2.5:7b", record, transport)

    assert steps == [
        ("pulling manifest", None, None),
        ("pulling 6a0746a1ec1a", 1_000_000_000, 4_000_000_000),
        ("success", None, None),
    ]

    lines[:] = [{"error": "pull model manifest: file does not exist"}]
    with pytest.raises(QuietpostModelError, match="file does not exist"):
        await init_wizard.pull_model("http://localhost:11434", "qwen2.5:7b", record, transport)


@pytest.mark.asyncio
async def test_a_connection_lost_during_a_download_is_a_model_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadError("connection reset")

    def ignore(status: str, completed: int | None, total: int | None) -> None:
        pass

    with pytest.raises(QuietpostModelError, match="could not download qwen2.5:7b"):
        await init_wizard.pull_model(
            "http://localhost:11434", "qwen2.5:7b", ignore, httpx.MockTransport(handler)
        )


def test_download_progress_rewrites_one_line_on_a_terminal_and_keeps_a_log_short() -> None:
    steps: list[tuple[str, int | None, int | None]] = [
        ("pulling manifest", None, None),
        ("pulling 6a07", 0, 4_000_000_000),
        ("pulling 6a07", 1_000_000_000, 4_000_000_000),
        ("pulling 6a07", 1_100_000_000, 4_000_000_000),  # still in the same 10%
        ("pulling 6a07", 4_000_000_000, 4_000_000_000),
        ("success", None, None),
    ]

    log = io.StringIO()  # not a terminal
    progress = init_wizard.DownloadProgress("qwen2.5:7b", log)
    for step in steps:
        progress(*step)
    progress.close()
    assert log.getvalue().splitlines() == [
        "qwen2.5:7b: pulling manifest",
        "qwen2.5:7b: pulling 6a07 0.0 of 4.0 GB (0%)",
        "qwen2.5:7b: pulling 6a07 1.0 of 4.0 GB (25%)",
        "qwen2.5:7b: pulling 6a07 4.0 of 4.0 GB (100%)",
        "qwen2.5:7b: success",
    ]

    class Terminal(io.StringIO):
        def isatty(self) -> bool:
            return True

    terminal = Terminal()
    progress = init_wizard.DownloadProgress("qwen2.5:7b", terminal)
    for step in steps:
        progress(*step)
    progress.close()
    shown = terminal.getvalue()
    assert shown.count("\n") == 1
    assert shown.endswith("\n")
    assert shown.rsplit("\r", 1)[-1].strip() == "qwen2.5:7b: success"
