"""A small HTTP client for the local Ollama server.

Talks to Ollama directly with `requests`. One shared session keeps connections
alive, which matters: a new connection per request is measurably slower, and by
about two seconds per call on Windows when the host is `localhost`.
"""
from __future__ import annotations

import json
from typing import Dict, Iterator, List, Optional, Sequence

import requests

from src.config import KEEP_ALIVE, OLLAMA_HOST


class OllamaError(RuntimeError):
    """Ollama answered with an error."""


class OllamaUnavailable(OllamaError):
    """Nothing is listening at the configured address."""


class ModelMissing(OllamaError):
    """The requested model has not been pulled."""

    def __init__(self, model: str):
        super().__init__(f"Model {model!r} is not installed. Run: ollama pull {model}")
        self.model = model


class OllamaClient:
    def __init__(self, host: str = OLLAMA_HOST, session: Optional[requests.Session] = None):
        self.host = host.rstrip("/")
        self.session = session or requests.Session()

    def _post(self, path: str, payload: Dict, *, stream: bool = False, timeout: float = 300) -> requests.Response:
        try:
            response = self.session.post(f"{self.host}{path}", json=payload, stream=stream, timeout=timeout)
        except requests.RequestException as exc:
            raise OllamaUnavailable(f"Cannot reach Ollama at {self.host}. Is `ollama serve` running?") from exc
        if response.status_code == 404:
            raise ModelMissing(str(payload.get("model")))
        if not response.ok:
            detail = response.text[:300]
            response.close()
            raise OllamaError(f"Ollama returned HTTP {response.status_code}: {detail}")
        return response

    def embed(self, model: str, texts: Sequence[str]) -> List[List[float]]:
        """Embed several texts in one request."""
        data = self._post("/api/embed", {"model": model, "input": list(texts), "keep_alive": KEEP_ALIVE}).json()
        embeddings = data.get("embeddings")
        if not embeddings or len(embeddings) != len(texts):
            raise OllamaError(f"Expected {len(texts)} embeddings, got {len(embeddings or [])}")
        return embeddings

    def chat_stream(
        self,
        model: str,
        messages: List[Dict],
        options: Optional[Dict] = None,
        keep_alive: str = KEEP_ALIVE,
    ) -> Iterator[Dict]:
        """Yield Ollama's streamed chat objects. Closing the iterator stops generation."""
        payload = {"model": model, "messages": messages, "stream": True, "options": options or {}, "keep_alive": keep_alive}
        response = self._post("/api/chat", payload, stream=True)
        try:
            for raw in response.iter_lines():
                if raw:
                    yield json.loads(raw)
        except requests.RequestException as exc:
            # The server went away partway through an answer, for example when it was stopped.
            raise OllamaUnavailable(f"Lost the connection to Ollama at {self.host}.") from exc
        finally:
            response.close()

    def chat(self, model: str, messages: List[Dict], options: Optional[Dict] = None) -> str:
        """Return a complete reply as one string."""
        return "".join(part.get("message", {}).get("content", "") for part in self.chat_stream(model, messages, options))

    def installed_models(self, timeout: float = 3) -> List[str]:
        """Names of installed models. Raises OllamaUnavailable when the server is down."""
        try:
            response = self.session.get(f"{self.host}/api/tags", timeout=timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise OllamaUnavailable(f"Cannot reach Ollama at {self.host}. Is `ollama serve` running?") from exc
        return [m["name"] for m in response.json().get("models", [])]
