"""The quietpost command: what `serve` and `doctor` say about the config and the model server."""

from pathlib import Path

import httpx
import pytest

from quietpost import cli
from quietpost.config import settings as settings_module
from quietpost.exceptions import QuietpostModelError
from quietpost.models import factory
from tests.unit._fake_ollama import OLLAMA_COM, FakeOllama, route_ollama


def _doctor_stays_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """No HTTPS check, and no look at port 8000, where a real Quietpost may be running."""
    monkeypatch.setattr("httpx.get", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_port_free", lambda host, port: True)


def _no_server(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """The port of each server `serve` would have started."""
    started: list[int] = []
    monkeypatch.setattr("uvicorn.run", lambda *args, **kwargs: started.append(kwargs["port"]))
    monkeypatch.setattr(cli, "_port_free", lambda host, port: True)
    return started


def _refuse(request: httpx.Request) -> httpx.Response:
    """Nothing listening at the address."""
    raise httpx.ConnectError("connection refused")


def test_doctor_names_the_config_it_uses_or_every_place_it_looked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An installed Quietpost reads the user config folder from any folder."""

    async def ask_model_server(_settings: object) -> cli._ServerReport:
        return cli._ServerReport({"qwen2.5:7b", "nomic-embed-text"}, cloud_off=True)

    monkeypatch.setattr(cli, "_ask_model_server", ask_model_server)
    _doctor_stays_offline(monkeypatch)
    # Looked up at call time: conftest points it into tmp_path, never the real folder.
    user_file = settings_module.user_config_file()
    assert user_file.is_relative_to(tmp_path)

    assert cli.main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert f"No quietpost.yaml found from {Path.cwd()} upward or in {user_file.parent}" in out

    user_file.parent.mkdir(parents=True)
    user_file.write_text("api_port: 9500\n", encoding="utf-8")
    assert cli.main(["doctor"]) == 0
    assert f"[ ok ] Config: {user_file}" in capsys.readouterr().out


def test_serve_says_to_open_the_ui_only_when_the_browser_can_use_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The UI can't send an API key, so with one set only the API is useful."""
    from quietpost.api import web
    from quietpost.config import get_settings

    (tmp_path / "web").mkdir()
    (tmp_path / "web" / "index.html").write_text("<!doctype html>", encoding="utf-8")
    monkeypatch.setattr(web, "WEB_DIR", tmp_path / "web")
    started = _no_server(monkeypatch)

    async def models_run_here(_settings: object) -> None:
        pass

    monkeypatch.setattr(cli, "_check_model_server", models_run_here)

    assert cli.main(["serve", "--port", "8019"]) == 0
    assert "Quietpost -> http://localhost:8019  (open it in your browser)" in capsys.readouterr().out

    monkeypatch.setenv("QUIETPOST_API_KEY", "a-test-key")
    get_settings.cache_clear()
    assert cli.main(["serve", "--port", "8019"]) == 0
    out = capsys.readouterr().out
    assert "Quietpost API -> http://localhost:8019" in out
    assert "open it in your browser" not in out
    assert "The UI can't send an API key yet" in out
    assert started == [8019, 8019]


def test_serve_refuses_a_public_model_server(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("QUIETPOST_MODEL_SERVER_URL", "http://8.8.8.8:11434")

    assert cli.main(["serve", "--port", "8019"]) == 1
    assert "public internet address" in capsys.readouterr().out


def test_doctor_refuses_a_public_model_server(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("QUIETPOST_MODEL_SERVER_URL", "http://8.8.8.8:11434")

    assert cli.main(["doctor"]) == 1
    assert "[FAIL] Quietpost only uses models on this computer" in capsys.readouterr().out


def test_doctor_lists_the_names_another_server_uses(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """LM Studio calls the default model qwen2.5-7b-instruct, not qwen2.5:7b."""
    monkeypatch.setenv("QUIETPOST_MODEL_SERVER", "openai-compatible")

    async def ask_model_server(_settings: object) -> cli._ServerReport:
        return cli._ServerReport({"qwen2.5-7b-instruct", "text-embedding-nomic-embed-text-v1.5"})

    monkeypatch.setattr(cli, "_ask_model_server", ask_model_server)
    _doctor_stays_offline(monkeypatch)

    assert cli.main(["doctor"]) == 1
    out = capsys.readouterr().out
    assert "[ ok ] Model server reachable at http://localhost:1234/v1 (2 models)" in out
    assert "[FAIL] chat_model: 'qwen2.5:7b' is missing.  Fix: load qwen2.5:7b" in out
    assert "The server lists: qwen2.5-7b-instruct, text-embedding-nomic-embed-text-v1.5" in out
    assert "OLLAMA_NO_CLOUD" not in out  # a setting of Ollama's


def test_serve_refuses_a_cloud_model_without_asking_ollama(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The name is enough, and asking Ollama about it would itself reach ollama.com."""
    monkeypatch.setenv("QUIETPOST_CHAT_MODEL", "gpt-oss:120b-cloud")
    fake = FakeOllama({"gpt-oss:120b-cloud": OLLAMA_COM, "nomic-embed-text": None})
    route_ollama(monkeypatch, factory, fake.transport())
    started = _no_server(monkeypatch)

    assert cli.main(["serve", "--port", "8019"]) == 1

    assert (
        "[FAIL] chat_model: Ollama passes gpt-oss:120b-cloud on to ollama.com instead of "
        "running it on this computer"
    ) in capsys.readouterr().out
    assert started == []
    assert fake.requests == []


def test_serve_refuses_a_model_ollama_passes_on_to_another_computer(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A cloud model copied under another name keeps running on ollama.com."""
    monkeypatch.setenv("QUIETPOST_CHAT_MODEL", "my-assistant")
    fake = FakeOllama({"my-assistant": OLLAMA_COM, "nomic-embed-text": None})
    route_ollama(monkeypatch, factory, fake.transport())
    started = _no_server(monkeypatch)

    assert cli.main(["serve", "--port", "8019"]) == 1

    out = capsys.readouterr().out
    assert f"[FAIL] chat_model: Ollama passes my-assistant on to {OLLAMA_COM} instead" in out
    assert "Set chat_model to a model that runs on this computer, such as qwen2.5:7b." in out
    assert started == []
    assert set(fake.paths()) == {"/api/tags", "/api/show"}


def test_serve_starts_when_the_models_run_here_or_ollama_is_not_running_yet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without Ollama the server still starts; it logs how to start Ollama."""
    started = _no_server(monkeypatch)
    route_ollama(monkeypatch, factory, FakeOllama({"qwen2.5:7b": None}).transport())
    assert cli.main(["serve", "--port", "8019"]) == 0

    route_ollama(monkeypatch, factory, httpx.MockTransport(_refuse))
    assert cli.main(["serve", "--port", "8019"]) == 0
    assert started == [8019, 8019]


def test_doctor_fails_a_model_ollama_passes_on_and_recommends_ollamas_switch(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("QUIETPOST_CODE_MODEL", "my-assistant")
    models = {"qwen2.5:7b": None, "nomic-embed-text:latest": None, "my-assistant": OLLAMA_COM}
    fake = FakeOllama(models)
    route_ollama(monkeypatch, factory, fake.transport())
    _doctor_stays_offline(monkeypatch)

    assert cli.main(["doctor"]) == 1

    out = capsys.readouterr().out
    assert "[ ok ] chat_model: qwen2.5:7b" in out
    assert "[ ok ] embed_model: nomic-embed-text" in out
    assert (
        f"[FAIL] code_model: Ollama passes my-assistant on to {OLLAMA_COM} instead of running "
        "it on this computer, so everything Quietpost sent it would leave this computer. Set "
        "code_model to a model that runs on this computer, such as qwen2.5:7b."
    ) in out
    assert "[warn] Ollama can also run models on ollama.com. Quietpost won't use them" in out
    assert "set OLLAMA_NO_CLOUD=1 for Ollama and restart it." in out
    assert "1 problem(s) found." in out
    # Asked about the models; sent nothing to one.
    assert set(fake.paths()) == {"/api/tags", "/api/show", "/api/status"}


@pytest.mark.parametrize(
    ("cloud_off", "line"),
    [
        (True, "[ ok ] Ollama's cloud models are switched off"),
        (None, "[warn] Ollama didn't say whether its cloud models are switched off."),
    ],
)
def test_doctor_says_whether_ollamas_cloud_models_are_switched_off(
    cloud_off: bool | None,
    line: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake = FakeOllama({"qwen2.5:7b": None, "nomic-embed-text": None}, cloud_off=cloud_off)
    route_ollama(monkeypatch, factory, fake.transport())
    _doctor_stays_offline(monkeypatch)

    assert cli.main(["doctor"]) == 0
    assert line in capsys.readouterr().out


def test_doctor_fails_a_model_ollama_will_not_describe(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """With its cloud switched off, Ollama lists a cloud-backed model but won't describe it."""
    monkeypatch.setenv("QUIETPOST_CHAT_MODEL", "my-assistant")
    models = {"my-assistant": OLLAMA_COM, "nomic-embed-text": None}
    route_ollama(monkeypatch, factory, FakeOllama(models, cloud_off=True).transport())
    _doctor_stays_offline(monkeypatch)

    assert cli.main(["doctor"]) == 1

    out = capsys.readouterr().out
    assert "[FAIL] chat_model: Ollama didn't say where 'my-assistant' runs, so Quietpost" in out
    assert "Ollama returned HTTP 403 for model 'my-assistant': ollama cloud is disabled" in out
    assert "[ ok ] embed_model: nomic-embed-text" in out


def test_doctor_refuses_a_cloud_model_without_asking_ollama(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("QUIETPOST_EMBED_MODEL", "my-embedder:cloud")
    fake = FakeOllama({"qwen2.5:7b": None, "my-embedder:cloud": OLLAMA_COM})
    route_ollama(monkeypatch, factory, fake.transport())
    _doctor_stays_offline(monkeypatch)

    assert cli.main(["doctor"]) == 1

    out = capsys.readouterr().out
    assert "[FAIL] embed_model: Ollama passes my-embedder:cloud on to ollama.com" in out
    assert "such as nomic-embed-text." in out
    assert fake.requests == []


def test_doctor_shows_what_answered_at_ollamas_address_instead_of_calling_it_unreachable(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Ollama asks for no key, so a 401 there comes from something else."""
    fake = FakeOllama({})
    fake.refusal = (401, '{"error": "unauthorized"}')
    route_ollama(monkeypatch, factory, fake.transport())
    _doctor_stays_offline(monkeypatch)

    assert cli.main(["doctor"]) == 1

    out = capsys.readouterr().out
    assert (
        '[FAIL] Ollama at http://localhost:11434 answered HTTP 401: {"error": "unauthorized"}'
    ) in out
    assert "Ollama itself doesn't ask for a key: is http://localhost:11434 your Ollama?" in out
    assert "not reachable" not in out
    assert "QUIETPOST_MODEL_SERVER_API_KEY" not in out


def test_doctor_says_how_to_give_another_model_server_its_key(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("QUIETPOST_MODEL_SERVER", "openai-compatible")

    async def refused(_settings: object) -> cli._ServerReport:
        request = httpx.Request("GET", "http://127.0.0.1:1234/v1/models")
        response = httpx.Response(401, text='{"error": "Invalid API key"}', request=request)
        cause = httpx.HTTPStatusError("HTTP 401", request=request, response=response)
        raise QuietpostModelError(f"Failed to list the models: {cause}") from cause

    monkeypatch.setattr(cli, "_ask_model_server", refused)
    _doctor_stays_offline(monkeypatch)

    assert cli.main(["doctor"]) == 1

    out = capsys.readouterr().out
    assert (
        "[FAIL] Model server at http://localhost:1234/v1 answered HTTP 401: "
        '{"error": "Invalid API key"}'
    ) in out
    assert "Fix: set QUIETPOST_MODEL_SERVER_API_KEY (or model_server_api_key) to its API key." in out


def test_doctor_says_how_to_start_ollama_when_nothing_answers(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    route_ollama(monkeypatch, factory, httpx.MockTransport(_refuse))
    _doctor_stays_offline(monkeypatch)

    assert cli.main(["doctor"]) == 1

    out = capsys.readouterr().out
    assert "[FAIL] Ollama not reachable at http://localhost:11434" in out
    assert "Fix: Start Ollama (`ollama serve` or the Ollama app)." in out
