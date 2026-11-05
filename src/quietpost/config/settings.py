"""Typed configuration for Quietpost, loaded from quietpost.yaml with env var overrides."""

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from platformdirs import user_config_path
from pydantic import BaseModel, field_validator, model_validator
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, EnvSettingsSource, PydanticBaseSettingsSource

_CONFIG_FILENAME = "quietpost.yaml"
_CONFIG_ENV_VAR = "QUIETPOST_CONFIG"
# Quietpost was called CORTEX until 2.0. Its names are still read, so an
# upgrade keeps working without renaming anything: the config file, the
# environment variables, and the data folder and database file.
_LEGACY_CONFIG_FILENAME = "cortex.yaml"
_LEGACY_CONFIG_ENV_VAR = "CORTEX_CONFIG"
_LEGACY_ENV_PREFIX = "CORTEX_"
_LEGACY_DATA_NAMES = {".quietpost": ".cortex", "quietpost.db": "cortex.db"}
# The model settings had Ollama-only names before 2.0. They are still read; the
# new name wins when both are set.
_RENAMED_MODEL_SETTINGS = {
    "ollama_base_url": "model_server_url",
    "ollama_model": "chat_model",
    "ollama_timeout_seconds": "model_timeout_seconds",
}
_DEFAULT_MODEL_URLS = {
    "ollama": "http://localhost:11434",
    # LM Studio's address. llama-server listens on :8080 and vLLM on :8000.
    "openai-compatible": "http://localhost:1234/v1",
}


def user_config_file() -> Path:
    """The config `quietpost init` writes, in the user's own config folder.

    It is AppData/Local/quietpost on Windows, ~/Library/Application Support/quietpost
    on macOS and ~/.config/quietpost on Linux.
    """
    return user_config_path("quietpost", appauthor=False) / _CONFIG_FILENAME


def _resolve_config_path() -> Path | None:
    """Locate quietpost.yaml: an explicit QUIETPOST_CONFIG, else the nearest one at or
    above the current working directory, so Quietpost works from any subdirectory,
    else the one in the user's config folder (see user_config_file). A cortex.yaml
    (or CORTEX_CONFIG) from before the rename is used the same way.
    """
    override = os.environ.get(_CONFIG_ENV_VAR) or os.environ.get(_LEGACY_CONFIG_ENV_VAR)
    if override:
        candidate = Path(override)
        return candidate if candidate.is_file() else None
    cwd = Path.cwd()
    for directory in (cwd, *cwd.parents):
        for name in (_CONFIG_FILENAME, _LEGACY_CONFIG_FILENAME):
            candidate = directory / name
            if candidate.is_file():
                return candidate
    user_file = user_config_file()
    return user_file if user_file.is_file() else None


def _existing_data_path(path: Path) -> Path:
    """``path``, or the same data under its pre-rename name if only that exists.

    Before 2.0, data lived in .cortex/ (and the database was cortex.db). Nothing
    is moved on upgrade: until the .quietpost location exists, the old one is used.
    """
    if path.exists():
        return path
    old = Path(*(_LEGACY_DATA_NAMES.get(part, part) for part in path.parts))
    return old if old.exists() else path


class _YamlSource(PydanticBaseSettingsSource):
    """Reads settings from the resolved quietpost.yaml (see _resolve_config_path)."""

    def _load(self) -> dict[str, Any]:
        config_path = _resolve_config_path()
        if config_path is None:
            return {}
        # Explicit UTF-8: the Windows default (cp1252) fails on any non-ASCII byte.
        with config_path.open(encoding="utf-8") as f:
            try:
                data = yaml.safe_load(f) or {}
            except yaml.YAMLError as exc:
                raise ValueError(f"Invalid YAML in {config_path}: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"{config_path} must contain a mapping of settings.")
        return self._anchor_paths(data, config_path.parent)

    def _anchor_paths(self, data: dict[str, Any], base: Path) -> dict[str, Any]:
        """Resolve relative path settings against the config file's directory.

        quietpost.yaml is found from any subdirectory, so "./.quietpost/quietpost.db" must
        mean next to the file, not next to wherever `quietpost serve` was started;
        otherwise memory, the workspace and plugins silently move with the shell's
        current directory. "~" expands to the home directory.
        """
        for name, field in self.settings_cls.model_fields.items():
            value = data.get(name)
            if field.annotation is Path and isinstance(value, str) and value.strip():
                data[name] = str(base / Path(value.strip()).expanduser())
        return data

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
    """Language Agent Tree Search settings.

    ``escalate_on_stall`` lets the ReAct loop hand a stalled task to LATS;
    ``escalation_timeout_seconds`` bounds that search so a stuck request still
    returns a best-effort answer in reasonable time.
    """

    enabled: bool = False
    max_depth: int = 5
    n_branches: int = 3
    budget: int = 10
    evaluator: Literal["model", "heuristic"] = "model"
    escalate_on_stall: bool = True
    escalation_timeout_seconds: float = 60.0


class SupervisorSettings(BaseModel):
    """Supervisor-worker orchestration settings."""

    max_workers: int = 3


class RateLimitSettings(BaseModel):
    """Per-IP token bucket rate limit settings."""

    enabled: bool = True
    requests_per_minute: int = 200
    chat_requests_per_minute: int = 60


class Settings(BaseSettings):
    """All Quietpost runtime configuration. No magic strings in the codebase."""

    # Where the models run: "ollama", or "openai-compatible" for LM Studio,
    # llama.cpp's llama-server, vLLM and other servers with an OpenAI-style API.
    # Only a server on this machine or the local network is accepted; a public
    # address is refused (models/endpoints.py), because prompts carry email.
    model_server: Literal["ollama", "openai-compatible"] = "ollama"
    # Unset means the usual address of model_server (see _DEFAULT_MODEL_URLS).
    model_server_url: str | None = None
    # Only for a server started with an API key (vLLM's --api-key).
    model_server_api_key: str | None = None
    # One tool-capable model for every chat role keeps setup to a single pull;
    # qwen2.5:7b had the most reliable tool calling of the models tested on a 6 GB GPU.
    chat_model: str = "qwen2.5:7b"
    # Each of these uses chat_model unless set. extract_model reads one email at
    # a time, which a smaller, faster model can do.
    extract_model: str | None = None
    reasoning_model: str | None = None
    code_model: str | None = None
    embed_model: str = "nomic-embed-text"
    # A cold 7-8B model load plus a long answer can exceed two minutes on a laptop.
    model_timeout_seconds: float = 300.0
    # Ollama only. Pinned for every call: Ollama reloads the model when num_ctx
    # changes, and its small default silently truncates long tool transcripts.
    ollama_num_ctx: int = 8192
    # Ollama only. How long it keeps the model resident after a request (its default is 5m).
    ollama_keep_alive: str = "30m"
    # Ollama only. Thinking models (qwen3, qwen3.5) reason before every answer
    # unless told not to: on a 6 GB GPU that took 17-35 s per call instead of 1-2 s.
    ollama_think: bool = False
    # The 1.x names of model_server_url, chat_model and model_timeout_seconds.
    ollama_base_url: str | None = None
    ollama_model: str | None = None
    ollama_timeout_seconds: float | None = None
    # Load the chat model in the background at startup so the first request is fast.
    warmup_on_startup: bool = True
    chroma_path: Path = Path("./.quietpost/chroma")
    db_path: Path = Path("./.quietpost/quietpost.db")
    task_store: Literal["memory", "sqlite"] = "memory"
    task_db_path: Path = Path("./.quietpost/tasks.db")
    # Quietpost can run code and read/write workspace files, so the API is private by
    # default: loopback only, callable only from the local UI's origin, and only
    # under local host names (a Host check blocks DNS-rebinding attacks from web
    # pages). Widen these deliberately, together with api_key, to expose it.
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_key: str | None = None
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    # Any port on the local machine (the UI moves to :3001 when :3000 is busy).
    # Remote web pages can never have a loopback origin, so this stays private.
    cors_origin_regex: str | None = r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"
    allowed_hosts: list[str] = ["localhost", "127.0.0.1"]
    # POST /tools/{name}/execute runs a tool directly, bypassing the agent. Debug only.
    debug_tool_endpoint: bool = False
    log_level: str = "INFO"
    telemetry_enabled: bool = True
    otel_endpoint: str = "http://localhost:4317"
    max_agent_steps: int = 10
    # Wall-clock budget for one ReAct run; past it, a best-effort answer is given.
    max_run_seconds: float = 120.0
    stream_tokens: bool = True
    procedural_memory_enabled: bool = True
    # Only tool patterns from near-duplicate past tasks become planner hints.
    procedural_min_relevance: float = 0.6
    # Prior user/assistant turns of the same session replayed into each request.
    history_turns: int = 6
    # Extract durable user facts into semantic memory after each exchange
    # (runs in the background; never delays the response).
    memory_consolidation: bool = True
    # Semantic memories below this relevance (1 / (1 + distance)) are not injected.
    semantic_min_relevance: float = 0.0
    plugins_dir: Path = Path("./src/quietpost/tools/plugins")
    # The only folder the agent's tools can read and write. A dedicated folder,
    # not the project root: the agent must not be able to read quietpost.yaml or
    # edit Quietpost's own code (a planted plugin would run unsandboxed on restart).
    allowed_root: Path = Path("./workspace")
    allowed_write_extensions: list[str] = [".txt", ".md", ".json", ".csv", ".py"]
    tool_timeout_seconds: float = 30.0
    code_exec_timeout_seconds: float = 10.0
    code_sandbox: Literal["restricted", "container"] = "restricted"
    code_sandbox_image: str = "python:3.12-slim"
    code_sandbox_mem_limit: str = "256m"
    code_sandbox_pids_limit: int = 128
    code_sandbox_cpus: float = 0.5
    lats: LATSSettings = LATSSettings()
    supervisor: SupervisorSettings = SupervisorSettings()
    rate_limit: RateLimitSettings = RateLimitSettings()

    model_config = {"env_prefix": "QUIETPOST_", "case_sensitive": False}

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log_level is one of the standard Python logging levels."""
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid:
            raise ValueError(f"log_level must be one of {valid}, got: {v!r}")
        return upper

    @model_validator(mode="after")
    def _keep_pre_rename_data(self) -> "Settings":
        """Point the data paths at a pre-2.0 .cortex folder when that is all there is."""
        self.chroma_path = _existing_data_path(self.chroma_path)
        self.db_path = _existing_data_path(self.db_path)
        self.task_db_path = _existing_data_path(self.task_db_path)
        return self

    @model_validator(mode="after")
    def _read_1x_model_settings(self) -> "Settings":
        """Use ollama_model and the other 1.x names when the new name is not set."""
        for old, new in _RENAMED_MODEL_SETTINGS.items():
            value = getattr(self, old)
            if value is not None and new not in self.model_fields_set:
                setattr(self, new, value)
        return self

    @property
    def model_url(self) -> str:
        """model_server_url, or the usual address of the chosen model server."""
        return self.model_server_url or _DEFAULT_MODEL_URLS[self.model_server]

    @property
    def configured_models(self) -> dict[str, str]:
        """The model each *_model setting uses; an unset one uses chat_model."""
        return {
            "chat_model": self.chat_model,
            "extract_model": self.extract_model or self.chat_model,
            "reasoning_model": self.reasoning_model or self.chat_model,
            "code_model": self.code_model or self.chat_model,
            "embed_model": self.embed_model,
        }

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
            # CORTEX_* variables from before the rename, below the QUIETPOST_* ones.
            EnvSettingsSource(settings_cls, env_prefix=_LEGACY_ENV_PREFIX, case_sensitive=False),
            _YamlSource(settings_cls),
        )
