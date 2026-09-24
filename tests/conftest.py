"""Shared pytest fixtures for the CORTEX test suite."""

from collections.abc import Generator
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[None, None, None]:
    """Point CORTEX data paths to a temp dir and clear the settings cache."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CORTEX_CHROMA_PATH", str(tmp_path / "chroma"))
    monkeypatch.setenv("CORTEX_DB_PATH", str(tmp_path / "cortex.db"))
    from cortex.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
    _clear_ephemeral_chroma()


def _clear_ephemeral_chroma() -> None:
    """Empty the process-wide in-memory Chroma between tests.

    Every ``chromadb.EphemeralClient()`` in a process shares one store, so
    collections (e.g. ``cortex_semantic``) would otherwise leak entries from one
    test into the next and make similarity results order-dependent.
    """
    import chromadb

    client = chromadb.EphemeralClient()
    for collection in client.list_collections():
        client.delete_collection(collection.name)
