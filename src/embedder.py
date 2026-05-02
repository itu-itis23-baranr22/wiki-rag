"""Local embedding via Ollama.

We talk to Ollama's HTTP API directly with `requests` — no SDK — to keep
the dependency surface tiny and to make it obvious what's going on.

The endpoint we use:
    POST {OLLAMA_HOST}/api/embeddings
    body: {"model": "nomic-embed-text", "prompt": "<text>"}
    response: {"embedding": [...]}
"""
from __future__ import annotations

from typing import List, Sequence

import requests

from src.config import EMBED_MODEL, OLLAMA_HOST


class OllamaEmbedder:
    """Tiny wrapper that returns embedding vectors for given texts."""

    def __init__(self, model: str = EMBED_MODEL, host: str = OLLAMA_HOST):
        self.model = model
        self.host = host.rstrip("/")
        self._endpoint = f"{self.host}/api/embeddings"

    def embed_one(self, text: str) -> List[float]:
        if not text.strip():
            raise ValueError("Cannot embed empty text")
        resp = requests.post(
            self._endpoint,
            json={"model": self.model, "prompt": text},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        if "embedding" not in data:
            raise RuntimeError(f"Ollama embedding response missing 'embedding': {data}")
        return data["embedding"]

    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        """Sequential batching — Ollama doesn't currently expose a batch API,
        so we just loop. For our scale (~500 chunks) this is fine."""
        return [self.embed_one(t) for t in texts]

    # ----------------------------------------------------------------- Chroma
    # Chroma's EmbeddingFunction protocol — Chroma 0.5+ strictly validates
    # that __call__ has signature exactly (self, input). No **kwargs allowed.
    def __call__(self, input):
        if isinstance(input, str):
            input = [input]
        return self.embed_batch(input)

    def embed_documents(self, input):
        if isinstance(input, str):
            input = [input]
        return self.embed_batch(input)

    def embed_query(self, input):
        if isinstance(input, list):
            return self.embed_batch(input)
        return [self.embed_one(input)]

    # Chroma 0.5+ requires this on custom embedding functions
    def name(self) -> str:
        return f"ollama-{self.model}"