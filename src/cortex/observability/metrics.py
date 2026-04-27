"""OpenTelemetry metrics setup — counters and histograms for the agent runtime."""

from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

_meter: metrics.Meter | None = None
_agent_steps_total: metrics.Counter | None = None
_llm_latency_seconds: metrics.Histogram | None = None


def setup_metrics(otel_endpoint: str | None = None) -> None:
    """Initialise OTel metrics SDK. Silently falls back to no-op if export fails."""
    global _meter, _agent_steps_total, _llm_latency_seconds
    try:
        if otel_endpoint:
            from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
                OTLPMetricExporter,
            )

            reader = PeriodicExportingMetricReader(
                OTLPMetricExporter(endpoint=otel_endpoint, insecure=True)
            )
            provider = MeterProvider(metric_readers=[reader])
        else:
            provider = MeterProvider()
        metrics.set_meter_provider(provider)
    except Exception:
        pass
    _meter = metrics.get_meter("cortex")
    _agent_steps_total = _meter.create_counter(
        "agent_steps_total",
        description="Total number of agent execution steps taken.",
    )
    _llm_latency_seconds = _meter.create_histogram(
        "llm_latency_seconds",
        description="Latency of LLM completion calls in seconds.",
        unit="s",
    )


def increment_agent_steps(model: str = "unknown") -> None:
    """Increment the agent_steps_total counter."""
    if _agent_steps_total is not None:
        _agent_steps_total.add(1, {"model": model})


def record_llm_latency(seconds: float, model: str = "unknown") -> None:
    """Record a single LLM call latency observation."""
    if _llm_latency_seconds is not None:
        _llm_latency_seconds.record(seconds, {"model": model})
