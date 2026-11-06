"""Local embeddings through Ollama, with the task prefixes the model expects."""
from __future__ import annotations

from functools import lru_cache
from typing import Callable, List, Optional, Sequence

from src.config import EMBED_BATCH_SIZE, EMBED_MODEL, EMBED_PREFIXES, EMBED_USE_PREFIXES
from src.ollama_client import OllamaClient


class OllamaEmbedder:
    """Embeds documents and queries. Repeated queries come from a small cache."""

    def __init__(
        self,
        model: str = EMBED_MODEL,
        client: Optional[OllamaClient] = None,
        use_prefixes: bool = EMBED_USE_PREFIXES,
        batch_size: int = EMBED_BATCH_SIZE,
    ):
        self.model = model
        self.client = client or OllamaClient()
        self.batch_size = batch_size
        prefixes = EMBED_PREFIXES.get(model.split(":")[0], ("", "")) if use_prefixes else ("", "")
        self.document_prefix, self.query_prefix = prefixes
        self._embed_query_cached = lru_cache(maxsize=256)(self._embed_query)

    @property
    def fingerprint(self) -> str:
        """Identifies vectors made this way. An index built differently must be rebuilt."""
        return f"{self.model}|{self.document_prefix}|{self.query_prefix}"

    def embed_documents(
        self,
        texts: Sequence[str],
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> List[List[float]]:
        vectors: List[List[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = [f"{self.document_prefix}{t}" for t in texts[start : start + self.batch_size]]
            vectors.extend(self.client.embed(self.model, batch))
            if on_progress:
                on_progress(len(vectors), len(texts))
        return vectors

    def _embed_query(self, text: str) -> tuple:
        return tuple(self.client.embed(self.model, [f"{self.query_prefix}{text}"])[0])

    def embed_query(self, text: str) -> List[float]:
        if not text.strip():
            raise ValueError("Cannot embed an empty query")
        return list(self._embed_query_cached(text))
