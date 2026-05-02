"""End-to-end RAG orchestrator.

Glues the retriever and generator together so the UI layer (Streamlit / CLI)
can call a single object.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional

from src.generator import OllamaGenerator
from src.retriever import Retriever
from src.vector_store import WikiVectorStore


@dataclass
class RagResult:
    answer: str
    chunks: List[Dict]
    route: str          # "person" | "place" | "both"
    rationale: str      # why the router chose that route


class RagSystem:
    def __init__(
        self,
        store: Optional[WikiVectorStore] = None,
        generator: Optional[OllamaGenerator] = None,
    ):
        self.store = store or WikiVectorStore()
        self.retriever = Retriever(self.store)
        self.generator = generator or OllamaGenerator()

    def ask(self, question: str) -> RagResult:
        retrieved = self.retriever.retrieve(question)
        chunks = retrieved["chunks"]
        route = retrieved["route"]
        answer = self.generator.generate(question, chunks)
        return RagResult(
            answer=answer,
            chunks=chunks,
            route=route.target,
            rationale=route.rationale,
        )

    def ask_stream(
        self,
        question: str,
        top_k: int | None = None,
        model: str | None = None,
        temperature: float | None = None,
    ) -> Iterator[Dict]:
        """Generator-friendly streaming variant.

        Optional `top_k`, `model`, `temperature` override the defaults from
        config.py for this single call (so a UI can wire them up to live
        sliders / dropdowns without re-creating the RAG system).

        Yields events as dicts:
            {"event": "retrieved", "chunks": [...], "route": "person", "rationale": "..."}
            {"event": "token",     "token": "..."}
            {"event": "done"}
        """
        retrieved = self.retriever.retrieve(question, top_k=top_k)
        chunks = retrieved["chunks"]
        route = retrieved["route"]
        yield {
            "event": "retrieved",
            "chunks": chunks,
            "route": route.target,
            "rationale": route.rationale,
        }
        for tok in self.generator.stream(
            question, chunks, model=model, temperature=temperature
        ):
            yield {"event": "token", "token": tok}
        yield {"event": "done"}