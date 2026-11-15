"""Quietpost: a private AI chief of staff that runs on your own machine."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("quietpost")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0+unknown"
