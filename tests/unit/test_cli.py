"""The quietpost command: what `serve` and `doctor` say about the config and the model server."""

from pathlib import Path

import pytest

from quietpost import cli
from quietpost.config import settings as settings_module


def test_doctor_names_the_config_it_uses_or_every_place_it_looked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An installed Quietpost reads the user config folder from any folder."""

    async def server_models(_settings: object) -> set[str]:
        return {"qwen2.5:7b", "nomic-embed-text"}

    monkeypatch.setattr(cli, "_server_models", server_models)
    monkeypatch.setattr("httpx.get", lambda *args, **kwargs: None)  # the HTTPS check
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

    async def server_models(_settings: object) -> set[str]:
        return {"qwen2.5-7b-instruct", "text-embedding-nomic-embed-text-v1.5"}

    monkeypatch.setattr(cli, "_server_models", server_models)
    monkeypatch.setattr("httpx.get", lambda *args, **kwargs: None)  # the HTTPS check

    assert cli.main(["doctor"]) == 1
    out = capsys.readouterr().out
    assert "[ ok ] Model server reachable at http://localhost:1234/v1 (2 models)" in out
    assert "[FAIL] chat_model: 'qwen2.5:7b' is missing.  Fix: load qwen2.5:7b" in out
    assert "The server lists: qwen2.5-7b-instruct, text-embedding-nomic-embed-text-v1.5" in out
