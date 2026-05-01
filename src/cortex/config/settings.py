"""Typed configuration for CORTEX, loaded from cortex.yaml with env var overrides."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, field_validator
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource


class _YamlSource(PydanticBaseSettingsSource):
    """Reads settings from cortex.yaml located at the current working directory."""

    def _load(self) -> dict[str, Any]:
        yaml_path = Path("cortex.yaml")
        if not yaml_path.exists():
            return {}
        with yaml_path.open() as f:
            return yaml.safe_load(f) or {}

    def get_field_value(
        self, field: FieldInfo, field_name: str
    ) -> tuple[Any, str, bool]:
        """Return (value, field_name, is_complex) for a settings field."""
        data = self._load()
        value = data.get(field_name)
        return value, field_name, self.field_is_complex(field)

    def field_is_complex(self, field: FieldInfo) -> bool:
        return False

    def __call__(self) -> dict[str, Any]:
        data = self._load()
        result: dict[str, Any] = {}
        for field_name in self.settings_cls.model_fields:
            if field_name in data:
                result[field_name] = data[field_name]
        return result


class LATSSettings(BaseModel):
    """Language Agent Tree Search settings."""

    enabled: bool = False
    max_depth: int = 5
    n_branches: int = 3
    budget: int = 10


class SupervisorSettings(BaseModel):
    """Supervisor-worker orchestration settings."""

    max_workers: int = 3


class RateLimitSettings(BaseModel):
    """Per-IP token bucket rate limit settings."""

    enabled: bool = True
    requests_per_minute: int = 200
    chat_requests_per_minute: int = 60


class Settings(BaseSettings):
    """All CORTEX runtime configuration. No magic strings in the codebase."""

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    embed_model: str = "nomic-embed-text"
    chroma_path: Path = Path("./.cortex/chroma")
    db_path: Path = Path("./.cortex/cortex.db")
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"
    otel_endpoint: str = "http://localhost:4317"
    max_agent_steps: int = 20
    plugins_dir: Path = Path("./src/cortex/tools/plugins")
    allowed_root: Path = Path(".")
    allowed_write_extensions: list[str] = [".txt", ".md", ".json", ".csv", ".py"]
    tool_timeout_seconds: float = 30.0
    code_exec_timeout_seconds: float = 10.0
    lats: LATSSettings = LATSSettings()
    supervisor: SupervisorSettings = SupervisorSettings()
    rate_limit: RateLimitSettings = RateLimitSettings()

    model_config = {"env_prefix": "CORTEX_", "case_sensitive": False}

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log_level is one of the standard Python logging levels."""
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid:
            raise ValueError(f"log_level must be one of {valid}, got: {v!r}")
        return upper

    @property
    def use_lats(self) -> bool:
        """Compatibility accessor for Phase 6's settings.use_lats reference."""
        return self.lats.enabled

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            _YamlSource(settings_cls),
        )
