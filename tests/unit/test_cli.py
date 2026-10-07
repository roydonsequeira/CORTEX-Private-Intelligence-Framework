"""The quietpost command: what `serve` and `doctor` say about the model server."""

import pytest

from quietpost import cli


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
