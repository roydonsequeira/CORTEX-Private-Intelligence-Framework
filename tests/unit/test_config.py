"""Tests for the Quietpost config module."""

from pathlib import Path

import pytest
import yaml

from quietpost.config import get_settings
from quietpost.config.settings import Settings  # noqa: F401


def test_settings_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings load with built-in defaults when no yaml or env vars present."""
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    s = get_settings()
    assert s.model_server == "ollama"
    assert s.model_url == "http://localhost:11434"
    assert s.api_port == 8000
    assert s.log_level == "INFO"


def test_settings_load_from_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings load values from quietpost.yaml."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "quietpost.yaml").write_text(
        yaml.dump({"chat_model": "qwen2.5:14b", "api_port": 9000})
    )
    get_settings.cache_clear()
    s = get_settings()
    assert s.chat_model == "qwen2.5:14b"
    assert s.api_port == 9000


def test_env_var_overrides_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Environment variables take precedence over quietpost.yaml values."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "quietpost.yaml").write_text(yaml.dump({"api_port": 9000}))
    monkeypatch.setenv("QUIETPOST_API_PORT", "7777")
    get_settings.cache_clear()
    s = get_settings()
    assert s.api_port == 7777


def test_settings_load_from_parent_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """quietpost.yaml is discovered by walking up from a nested working directory."""
    (tmp_path / "quietpost.yaml").write_text(yaml.dump({"api_port": 9100}))
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    get_settings.cache_clear()
    s = get_settings()
    assert s.api_port == 9100


def test_relative_yaml_paths_are_relative_to_the_config_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Starting Quietpost from a subfolder must not move its memory or workspace."""
    # conftest points these at tmp_path through env vars, which override the file.
    monkeypatch.delenv("QUIETPOST_DB_PATH")
    monkeypatch.delenv("QUIETPOST_CHROMA_PATH")
    (tmp_path / "quietpost.yaml").write_text(
        yaml.dump(
            {
                "db_path": "./.quietpost/quietpost.db",
                "chroma_path": ".quietpost/chroma",
                "allowed_root": "./workspace",
                "plugins_dir": "./plugins",
                "task_db_path": str(tmp_path / "abs" / "tasks.db"),
            }
        )
    )
    nested = tmp_path / "ui"
    nested.mkdir()
    monkeypatch.chdir(nested)
    get_settings.cache_clear()
    s = get_settings()
    assert s.db_path == tmp_path / ".quietpost" / "quietpost.db"
    assert s.chroma_path == tmp_path / ".quietpost" / "chroma"
    assert s.allowed_root == tmp_path / "workspace"
    assert s.plugins_dir == tmp_path / "plugins"
    assert s.task_db_path == tmp_path / "abs" / "tasks.db"  # absolute paths are kept


def test_yaml_paths_expand_the_home_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`~/quietpost-data` means the home folder, not a directory literally named ~."""
    monkeypatch.delenv("QUIETPOST_DB_PATH")
    (tmp_path / "quietpost.yaml").write_text(yaml.dump({"db_path": "~/quietpost-data/quietpost.db"}))
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    assert get_settings().db_path == Path.home() / "quietpost-data" / "quietpost.db"


def test_default_workspace_is_a_dedicated_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without configuration the agent's files live in ./workspace, not the project root."""
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    assert get_settings().allowed_root == Path("workspace")


def test_settings_load_from_quietpost_config_env_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """QUIETPOST_CONFIG points the loader at an explicit config file anywhere."""
    config_file = tmp_path / "custom-config.yaml"
    config_file.write_text(yaml.dump({"api_port": 9200}))
    work_dir = tmp_path / "elsewhere"
    work_dir.mkdir()
    monkeypatch.chdir(work_dir)
    monkeypatch.setenv("QUIETPOST_CONFIG", str(config_file))
    get_settings.cache_clear()
    s = get_settings()
    assert s.api_port == 9200


def test_get_settings_is_singleton() -> None:
    """get_settings() returns the same instance on repeated calls."""
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2


def test_invalid_log_level_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An invalid log_level value raises a validation error."""
    from pydantic import ValidationError

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("QUIETPOST_LOG_LEVEL", "VERBOSE")
    get_settings.cache_clear()
    with pytest.raises(ValidationError):
        get_settings()


def test_a_cortex_yaml_from_before_the_rename_is_still_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An upgrade from 1.x keeps working without renaming cortex.yaml."""
    (tmp_path / "cortex.yaml").write_text(yaml.dump({"api_port": 9300}))
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    assert get_settings().api_port == 9300


def test_quietpost_yaml_wins_over_cortex_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "cortex.yaml").write_text(yaml.dump({"api_port": 9300}))
    (tmp_path / "quietpost.yaml").write_text(yaml.dump({"api_port": 9400}))
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    assert get_settings().api_port == 9400


def test_cortex_env_vars_from_before_the_rename_are_still_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CORTEX_* still works, below QUIETPOST_*; so does CORTEX_CONFIG."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CORTEX_API_PORT", "7001")
    get_settings.cache_clear()
    assert get_settings().api_port == 7001

    monkeypatch.setenv("QUIETPOST_API_PORT", "7002")
    get_settings.cache_clear()
    assert get_settings().api_port == 7002

    config_file = tmp_path / "old-config.yaml"
    config_file.write_text(yaml.dump({"log_level": "DEBUG"}))
    monkeypatch.setenv("CORTEX_CONFIG", str(config_file))
    get_settings.cache_clear()
    assert get_settings().log_level == "DEBUG"


def test_data_from_before_the_rename_is_used_until_the_new_folder_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing is moved on upgrade: memory in .cortex/ keeps working where it is."""
    monkeypatch.delenv("QUIETPOST_CHROMA_PATH")
    monkeypatch.delenv("QUIETPOST_DB_PATH")
    (tmp_path / "quietpost.yaml").write_text(
        yaml.dump({"chroma_path": "./.quietpost/chroma", "db_path": "./.quietpost/quietpost.db"})
    )
    (tmp_path / ".cortex" / "chroma").mkdir(parents=True)
    (tmp_path / ".cortex" / "cortex.db").write_bytes(b"")
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.chroma_path == tmp_path / ".cortex" / "chroma"
    assert settings.db_path == tmp_path / ".cortex" / "cortex.db"

    (tmp_path / ".quietpost" / "chroma").mkdir(parents=True)
    (tmp_path / ".quietpost" / "quietpost.db").write_bytes(b"")
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.chroma_path == tmp_path / ".quietpost" / "chroma"
    assert settings.db_path == tmp_path / ".quietpost" / "quietpost.db"


def test_model_url_defaults_to_the_usual_address_of_the_chosen_server() -> None:
    assert Settings().model_url == "http://localhost:11434"
    assert Settings(model_server="openai-compatible").model_url == "http://localhost:1234/v1"
    assert (
        Settings(model_server="openai-compatible", model_server_url="http://localhost:8080/v1").model_url
        == "http://localhost:8080/v1"
    )


def test_an_unknown_model_server_is_rejected() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Settings(model_server="openai")  # type: ignore[arg-type]


def test_1x_model_setting_names_are_still_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 1.x cortex.yaml names the model settings after Ollama; they still apply."""
    (tmp_path / "cortex.yaml").write_text(
        yaml.dump(
            {
                "ollama_base_url": "http://gpu-box:11434",
                "ollama_model": "qwen2.5:14b",
                "ollama_timeout_seconds": 600,
            }
        )
    )
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    s = get_settings()
    assert (s.model_url, s.chat_model, s.model_timeout_seconds) == (
        "http://gpu-box:11434",
        "qwen2.5:14b",
        600,
    )

    monkeypatch.setenv("QUIETPOST_OLLAMA_MODEL", "llama3.1:8b")
    get_settings.cache_clear()
    assert get_settings().chat_model == "llama3.1:8b"


def test_new_model_setting_names_win_over_1x_ones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "quietpost.yaml").write_text(
        yaml.dump({"ollama_model": "old-m", "chat_model": "new-m"})
    )
    monkeypatch.setenv("CORTEX_OLLAMA_BASE_URL", "http://old:11434")
    monkeypatch.setenv("QUIETPOST_MODEL_SERVER_URL", "http://new:11434")
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    s = get_settings()
    assert (s.chat_model, s.model_url) == ("new-m", "http://new:11434")


def test_a_fresh_install_uses_the_new_data_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("QUIETPOST_CHROMA_PATH")
    monkeypatch.delenv("QUIETPOST_DB_PATH")
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.chroma_path == Path(".quietpost/chroma")
    assert settings.db_path == Path(".quietpost/quietpost.db")
