"""Configuration module — exposes a cached settings singleton."""

from functools import lru_cache

from quietpost.config.settings import Settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance, loaded once on first call."""
    return Settings()
