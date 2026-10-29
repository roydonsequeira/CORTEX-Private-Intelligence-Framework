"""CORTEX — Private Intelligence Framework."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("cortex-agent")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0+unknown"
