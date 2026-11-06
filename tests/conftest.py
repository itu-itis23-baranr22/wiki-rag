"""Shared fixtures. Nothing here needs Ollama or the network."""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from typing import Dict, Iterator, List, Optional, Sequence

import pytest

from src.catalog import Catalog
from src.embedder import OllamaEmbedder
from src.ingest import index_manifest, index_page
from src.language import QueryTranslator
from src.ollama_client import OllamaUnavailable
from src.rag import RagSystem
from src.router import Router
from src.vector_store import WikiVectorStore
from src.wiki import Page

DIMENSIONS = 256


def bag_of_words(text: str) -> List[float]:
    """A deterministic embedding in which texts sharing words end up close together."""
    vector = [0.0] * DIMENSIONS
    for word in re.findall(r"\w+", text.lower()):
        vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIMENSIONS] += 1.0
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


class FakeOllama:
    """Stands in for OllamaClient and records every call."""

    def __init__(self, reply: str = "Marie Curie was born in Warsaw [1]."):
        self.reply = reply
        self.translation = "Where is Hagia Sophia?"
        self.embed_calls: List[List[str]] = []
        self.chats: List[List[Dict]] = []
        self.available = True

    def embed(self, model: str, texts: Sequence[str]) -> List[List[float]]:
        self.embed_calls.append(list(texts))
        return [bag_of_words(t) for t in texts]

    def chat_stream(self, model, messages: List[Dict], options: Optional[Dict] = None, keep_alive: str = "") -> Iterator[Dict]:
        self.chats.append(messages)
        for word in re.findall(r"\S+\s*", self.reply):
            yield {"message": {"content": word}}
        yield {"done": True, "prompt_eval_count": 100, "eval_count": 10, "eval_duration": 1_000_000_000}

    def chat(self, model: str, messages: List[Dict], options: Optional[Dict] = None) -> str:
        return self.translation

    def installed_models(self, timeout: float = 3) -> List[str]:
        if not self.available:
            raise OllamaUnavailable("Ollama is not running")
        return ["llama3.2:3b", "nomic-embed-text:latest"]


PAGES: Dict[str, List] = {
    "Marie Curie": [
        ("", "Marie Curie was a Polish and naturalised-French physicist and chemist. She was born in Warsaw in 1867."),
        ("Career", "She discovered polonium and radium. She won two Nobel Prizes for her research on radioactivity."),
    ],
    "Albert Einstein": [
        ("", "Albert Einstein was a German-born theoretical physicist. He developed the theory of relativity."),
        ("Awards", "He received the 1921 Nobel Prize in Physics. He was born in Ulm in 1879."),
    ],
    "Steve Jobs": [("", "Steve Jobs co-founded Apple and led Pixar. His jobs included running NeXT.")],
    "Taylor Swift": [("", "Taylor Swift is an American singer. Swift released the album Folklore in 2020.")],
    "Eiffel Tower": [("", "The Eiffel Tower is a wrought-iron lattice tower in Paris, France. It is 330 metres tall.")],
    "Hagia Sophia": [("", "Hagia Sophia is a mosque and former church in Istanbul, Turkey. It was built in 537.")],
    "Cappadocia": [("", "Cappadocia is a historical region in Central Anatolia, Turkey, known for fairy chimneys.")],
    "Colosseum": [("", "The Colosseum is an elliptical amphitheatre in the centre of Rome, Italy.")],
}


def make_page(title: str) -> Page:
    url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
    return Page(title=title, url=url, revision=1, fetched_at=time.time(), sections=[tuple(s) for s in PAGES[title]])


@pytest.fixture
def fake_ollama() -> FakeOllama:
    return FakeOllama()


@pytest.fixture
def catalog(tmp_path) -> Catalog:
    return Catalog(tmp_path / "custom_pages.json")


@pytest.fixture
def embedder(fake_ollama) -> OllamaEmbedder:
    return OllamaEmbedder(client=fake_ollama)


@pytest.fixture
def store(tmp_path, catalog, embedder) -> WikiVectorStore:
    """A real Chroma index holding the small test corpus."""
    built = WikiVectorStore(path=tmp_path / "index", collection="test")
    built.write_manifest(index_manifest(embedder))
    for title in PAGES:
        index_page(built, embedder, catalog.get(title), make_page(title))
    return built


@pytest.fixture
def rag(store, fake_ollama, embedder, catalog) -> RagSystem:
    return RagSystem(
        store=store,
        client=fake_ollama,
        embedder=embedder,
        catalog=catalog,
        router=Router(catalog),
        translator=QueryTranslator(client=fake_ollama),
    )


def parse_sse(body: str) -> List[Dict]:
    return [json.loads(line[len("data: ") :]) for line in body.splitlines() if line.startswith("data: ")]
