"""OpenTelemetry tracing setup. Telemetry must never crash the application."""

import structlog
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

logger = structlog.get_logger(__name__)

_tracer_provider: TracerProvider | None = None


def setup_tracing(service_name: str, otel_endpoint: str, enabled: bool = True) -> None:
    """Initialise OTel SDK with OTLP gRPC exporter.

    When ``enabled`` is False (or no endpoint is given) the span exporter is not
    attached, so spans are created but never exported and no connection to the
    collector is attempted — keeping a local, no-collector run free of the
    repeated "connection refused" export warnings. Silently continues if the
    exporter cannot be initialised.
    """
    global _tracer_provider
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    if enabled and otel_endpoint:
        try:
            exporter = OTLPSpanExporter(endpoint=otel_endpoint, insecure=True)
            provider.add_span_processor(BatchSpanProcessor(exporter))
        except Exception as exc:
            logger.warning("otel_exporter_init_failed", endpoint=otel_endpoint, error=str(exc))
    trace.set_tracer_provider(provider)
    _tracer_provider = provider


def get_tracer(name: str) -> trace.Tracer:
    """Return a named tracer. Uses the configured provider or a no-op fallback."""
    return trace.get_tracer(name)


def shutdown_tracing(timeout_millis: int = 5000) -> None:
    """Flush and shut down the tracer provider without crashing shutdown."""
    if _tracer_provider is None:
        return
    try:
        _tracer_provider.force_flush(timeout_millis)
        _tracer_provider.shutdown()
    except Exception as exc:
        logger.warning("otel_shutdown_failed", error=str(exc))