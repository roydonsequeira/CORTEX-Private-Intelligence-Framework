"""Observability module — initialised on import via explicit setup calls."""

from cortex.observability.logging import configure_logging
from cortex.observability.metrics import setup_metrics
from cortex.observability.tracing import get_tracer, setup_tracing

__all__ = ["configure_logging", "get_tracer", "setup_metrics", "setup_tracing"]
