"""Tests for the CORTEX config module."""

from pathlib import Path

import pytest
import yaml

from cortex.config import get_settings
from cortex.config.settings import Settings  # noqa: F401


def test_settings_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings load with built-in defaults when no yaml or env vars present."""
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    s = get_settings()
    assert s.ollama_base_url == "http://localhost:11434"
    assert s.api_port == 8000
    assert s.log_level == "INFO"


def test_settings_load_from_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings load values from cortex.yaml."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "cortex.yaml").write_text(
        yaml.dump({"ollama_model": "qwen2.5:14b", "api_port": 9000})
    )
    get_settings.cache_clear()
    s = get_settings()
    assert s.ollama_model == "qwen2.5:14b"
    assert s.api_port == 9000


def test_env_var_overrides_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Environment variables take precedence over cortex.yaml values."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "cortex.yaml").write_text(yaml.dump({"api_port": 9000}))
    monkeypatch.setenv("CORTEX_API_PORT", "7777")
    get_settings.cache_clear()
    s = get_settings()
    assert s.api_port == 7777


def test_settings_load_from_parent_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """cortex.yaml is discovered by walking up from a nested working directory."""
    (tmp_path / "cortex.yaml").write_text(yaml.dump({"api_port": 9100}))
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    get_settings.cache_clear()
    s = get_settings()
    assert s.api_port == 9100


def test_settings_load_from_cortex_config_env_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CORTEX_CONFIG points the loader at an explicit config file anywhere."""
    config_file = tmp_path / "custom-config.yaml"
    config_file.write_text(yaml.dump({"api_port": 9200}))
    work_dir = tmp_path / "elsewhere"
    work_dir.mkdir()
    monkeypatch.chdir(work_dir)
    monkeypatch.setenv("CORTEX_CONFIG", str(config_file))
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
    monkeypatch.setenv("CORTEX_LOG_LEVEL", "VERBOSE")
    get_settings.cache_clear()
    with pytest.raises(ValidationError):
        get_settings()
