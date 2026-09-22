"""Unit tests for telemetry setup, especially the disable path."""

import pytest

from cortex.observability import tracing


class _StubProcessor:
    """Minimal span-processor stand-in so add_span_processor is happy in tests."""

    def on_start(self, *args: object, **kwargs: object) -> None:
        return None

    def on_end(self, *args: object, **kwargs: object) -> None:
        return None

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 0) -> bool:
        return True


def test_setup_tracing_disabled_skips_exporter(monkeypatch: pytest.MonkeyPatch) -> None:
    """With telemetry disabled, no OTLP exporter is constructed (no connection attempts)."""
    built: list[int] = []
    monkeypatch.setattr(tracing, "OTLPSpanExporter", lambda *a, **k: built.append(1))

    tracing.setup_tracing("svc", "http://localhost:4317", enabled=False)

    assert built == []
    # The tracer is still usable; spans are simply never exported.
    with tracing.get_tracer("t").start_as_current_span("span"):
        pass


def test_setup_tracing_enabled_attaches_exporter(monkeypatch: pytest.MonkeyPatch) -> None:
    """With telemetry enabled and an endpoint, the exporter is wired up."""
    built: list[int] = []

    def _make_processor(*args: object, **kwargs: object) -> _StubProcessor:
        built.append(1)
        return _StubProcessor()

    monkeypatch.setattr(tracing, "OTLPSpanExporter", lambda *a, **k: object())
    monkeypatch.setattr(tracing, "BatchSpanProcessor", _make_processor)

    tracing.setup_tracing("svc", "http://localhost:4317", enabled=True)

    assert built == [1]


def test_setup_tracing_no_endpoint_skips_exporter(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty endpoint also skips the exporter even when enabled."""
    built: list[int] = []
    monkeypatch.setattr(tracing, "OTLPSpanExporter", lambda *a, **k: built.append(1))

    tracing.setup_tracing("svc", "", enabled=True)

    assert built == []
