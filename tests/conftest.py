"""Shared pytest fixtures for the CORTEX test suite."""

import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point CORTEX data paths to a temp dir and clear the settings cache."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CORTEX_CHROMA_PATH", str(tmp_path / "chroma"))
    monkeypatch.setenv("CORTEX_DB_PATH", str(tmp_path / "cortex.db"))
    from cortex.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
