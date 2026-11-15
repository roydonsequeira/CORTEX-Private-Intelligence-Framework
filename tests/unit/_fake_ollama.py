"""A fake Ollama: it answers the requests Quietpost makes, and records each one."""

import json
from typing import Any

import httpx
import pytest

from quietpost.models.provider import OllamaProvider

# What Ollama 0.12 and later puts in /api/show for a model it passes on to ollama.com.
OLLAMA_COM = "https://ollama.com:443"


class FakeOllama:
    """An Ollama with the given models, served through an httpx transport.

    ``models`` maps each model it has to the host it passes the model on to, or to
    None for a model it runs itself. ``cloud_off`` is what /api/status says about
    Ollama's switch for its cloud models; with None there is no /api/status, as
    before Ollama 0.16.2. With the switch off, Ollama won't describe a model it
    would pass on.
    """

    def __init__(self, models: dict[str, str | None], cloud_off: bool | None = False) -> None:
        self.models = models
        self.cloud_off = cloud_off
        # When set, every request gets this status and body, as from a proxy that wants a key.
        self.refusal: tuple[int, str] | None = None
        # (path, model) for each request, with "" for a request about no model.
        self.requests: list[tuple[str, str]] = []

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._answer)

    def paths(self) -> list[str]:
        return [path for path, _model in self.requests]

    def _find(self, model: str) -> str | None:
        """The name Ollama has ``model`` under: "nomic-embed-text" is "nomic-embed-text:latest"."""
        return next((name for name in (model, f"{model}:latest") if name in self.models), None)

    def _answer(self, request: httpx.Request) -> httpx.Response:
        body: dict[str, Any] = json.loads(request.content) if request.content else {}
        model = str(body.get("model", ""))
        path = request.url.path
        self.requests.append((path, model))
        if self.refusal:
            return httpx.Response(self.refusal[0], text=self.refusal[1])
        if path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": name} for name in self.models]})
        if path == "/api/status":
            if self.cloud_off is None:
                return httpx.Response(404, text="404 page not found")
            source = "env" if self.cloud_off else "none"
            return httpx.Response(200, json={"cloud": {"disabled": self.cloud_off, "source": source}})
        name = self._find(model)
        if name is None:
            return httpx.Response(404, json={"error": f"model '{model}' not found"})
        host = self.models[name]
        if path == "/api/show" and host and self.cloud_off:
            error = "ollama cloud is disabled: remote model details are unavailable"
            return httpx.Response(403, json={"error": error})
        if path == "/api/show":
            details: dict[str, Any] = {"details": {"format": "gguf"}, "capabilities": ["completion"]}
            if host:
                details |= {"remote_host": host, "remote_model": name}
            return httpx.Response(200, json=details)
        if path == "/api/chat" and body.get("stream"):
            line = {"model": name, "message": {"role": "assistant", "content": "hi"}, "done": True}
            return httpx.Response(200, content=(json.dumps(line) + "\n").encode())
        if path == "/api/chat":
            message = {"role": "assistant", "content": "hi"}
            return httpx.Response(200, json={"model": name, "message": message, "done": True})
        if path == "/api/embed":
            return httpx.Response(200, json={"embeddings": [[0.1, 0.2] for _ in body["input"]]})
        if path == "/api/generate":
            return httpx.Response(200, json={"model": name, "response": "", "done": True})
        return httpx.Response(404, text="404 page not found")


def route_ollama(
    monkeypatch: pytest.MonkeyPatch, module: object, transport: httpx.AsyncBaseTransport
) -> None:
    """Every OllamaProvider that ``module`` builds sends its requests to ``transport``."""

    class _Routed(OllamaProvider):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **{**kwargs, "transport": transport})

    monkeypatch.setattr(module, "OllamaProvider", _Routed)
