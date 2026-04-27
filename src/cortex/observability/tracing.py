"""OpenTelemetry tracing setup. Telemetry must never crash the application."""

import structlog
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

logger = structlog.get_logger(__name__)

_tracer_provider: TracerProvider | None = None


def setup_tracing(service_name: str, otel_endpoint: str) -> None:
    """Initialise OTel SDK with OTLP gRPC exporter. Silently continues if unreachable."""
    global _tracer_provider
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
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
