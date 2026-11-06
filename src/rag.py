"""End-to-end question answering. The server, CLI and evaluation all go through this."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence

from src.attribution import attribute
from src.catalog import Catalog
from src.config import MAX_TOP_K
from src.embedder import OllamaEmbedder
from src.generator import REFUSALS, Generator, is_refusal
from src.ingest import index_manifest
from src.language import QueryTranslator, detect_language
from src.ollama_client import OllamaClient
from src.retriever import RetrievalSettings, Retriever
from src.router import Router
from src.vector_store import WikiVectorStore


@dataclass
class RagResult:
    answer: str
    language: str
    refused: bool
    route: Dict
    sources: List[Dict]
    attribution: Dict
    timings: Dict = field(default_factory=dict)


def public_source(number: int, chunk: Dict) -> Dict:
    """The fields of a retrieved passage that the interface and API expose."""
    meta = chunk.get("metadata") or {}
    return {
        "number": number,
        "id": chunk["id"],
        "entity": meta.get("entity", ""),
        "type": meta.get("type", ""),
        "section": meta.get("section", ""),
        "chunk_index": meta.get("chunk_idx"),
        "page_passages": meta.get("n_chunks"),
        "url": meta.get("source", ""),
        "text": chunk["document"],
        "vector_rank": chunk.get("vector_rank"),
        "lexical_rank": chunk.get("lexical_rank"),
        "distance": chunk.get("distance"),
    }


class RagSystem:
    def __init__(
        self,
        store: Optional[WikiVectorStore] = None,
        client: Optional[OllamaClient] = None,
        embedder: Optional[OllamaEmbedder] = None,
        catalog: Optional[Catalog] = None,
        generator: Optional[Generator] = None,
        translator: Optional[QueryTranslator] = None,
        settings: Optional[RetrievalSettings] = None,
        router: Optional[Router] = None,
        translate: bool = True,
    ):
        self.client = client or OllamaClient()
        self.store = store or WikiVectorStore()
        self.embedder = embedder or OllamaEmbedder(client=self.client)
        self.catalog = catalog or Catalog()
        self.router = router or Router(self.catalog)
        self.retriever = Retriever(self.store, self.embedder, self.router, settings=settings)
        self.generator = generator or Generator(client=self.client)
        self.translator = translator or QueryTranslator(client=self.client, model=self.generator.model)
        self.translate = translate
        self.store.check_manifest(index_manifest(self.embedder))

    def prepare(self, question: str, top_k: Optional[int] = None, history: Optional[Sequence[Dict]] = None) -> Dict:
        """Everything that happens before the model writes: language, routing and search."""
        started = time.perf_counter()
        language = detect_language(question)
        translated = None
        if language != "en" and self.translate:
            translated = self.translator.translate(question)
        translate_seconds = time.perf_counter() - started

        previous = (history[-1].get("entities") if history else None) or None
        limit = min(top_k, MAX_TOP_K) if top_k else None
        started = time.perf_counter()
        retrieved = self.retriever.retrieve(question, top_k=limit, translated=translated, previous_entities=previous)
        retrieve_seconds = time.perf_counter() - started
        return {
            "language": language,
            "translated": translated,
            "retrieved": retrieved,
            "translate_seconds": translate_seconds,
            "retrieve_seconds": retrieve_seconds,
        }

    def ask_stream(
        self,
        question: str,
        top_k: Optional[int] = None,
        model: Optional[str] = None,
        history: Optional[Sequence[Dict]] = None,
    ) -> Iterator[Dict]:
        """Yield events as work progresses:

            {"event": "route",   ...}   how the question was understood
            {"event": "sources", ...}   the retrieved passages, before the model starts
            {"event": "token",   ...}   a piece of the answer
            {"event": "done",    ...}   final answer, attribution and timings
        """
        total_started = time.perf_counter()
        prepared = self.prepare(question, top_k, history)
        retrieved = prepared["retrieved"]
        language = prepared["language"]
        route = retrieved.route
        chunks = retrieved.chunks
        sources = [public_source(number, chunk) for number, chunk in enumerate(chunks, start=1)]

        yield {
            "event": "route",
            "language": language,
            "translated": prepared["translated"],
            "target": route.target,
            "reason": route.reason,
            "rationale": route.rationale,
            "entities": route.entity_names,
            "indexed_passages": self.store.count(),
            "translate_ms": round(prepared["translate_seconds"] * 1000),
            "retrieve_ms": round(prepared["retrieve_seconds"] * 1000),
        }
        yield {"event": "sources", "sources": sources}

        parts: List[str] = []
        stats: Dict = {}
        generate_started = time.perf_counter()
        first_token_seconds: Optional[float] = None
        for piece in self.generator.stream(question, chunks, language, history=history, model=model):
            if "token" in piece:
                if first_token_seconds is None:
                    first_token_seconds = time.perf_counter() - generate_started
                parts.append(piece["token"])
                yield {"event": "token", "token": piece["token"]}
            else:
                stats = piece["stats"]

        answer = "".join(parts).strip()
        refused = is_refusal(answer) or not answer
        attribution = None if refused else attribute(answer, [s["text"] for s in sources], language)
        # An answer with no sentence traceable to any passage came from outside the pages,
        # so it is withheld rather than shown as if it were sourced.
        withheld = attribution is not None and not any(s["evidence"] for p in attribution["paragraphs"] for s in p)
        if refused or withheld:
            refused = True
            answer = REFUSALS[language]
            attribution = {"paragraphs": [], "used_sources": [], "invalid_citations": 0, "uncited_sentences": 0}

        eval_seconds = stats.get("eval_seconds") or 0
        yield {
            "event": "done",
            "answer": answer,
            "refused": refused,
            "withheld": withheld,
            "language": language,
            "entities": sorted({s["entity"] for s in sources if s["number"] in attribution["used_sources"]})
            or route.entity_names,
            "attribution": attribution,
            "timings": {
                "translate_ms": round(prepared["translate_seconds"] * 1000),
                "retrieve_ms": round(prepared["retrieve_seconds"] * 1000),
                "first_token_ms": round((first_token_seconds or 0) * 1000),
                "total_ms": round((time.perf_counter() - total_started) * 1000),
                "prompt_tokens": stats.get("prompt_tokens"),
                "answer_tokens": stats.get("answer_tokens"),
                "tokens_per_second": round(stats["answer_tokens"] / eval_seconds, 1)
                if eval_seconds and stats.get("answer_tokens")
                else None,
            },
        }

    def ask(
        self,
        question: str,
        top_k: Optional[int] = None,
        model: Optional[str] = None,
        history: Optional[Sequence[Dict]] = None,
    ) -> RagResult:
        """Answer one question and return the complete result."""
        route: Dict = {}
        sources: List[Dict] = []
        done: Dict = {}
        for event in self.ask_stream(question, top_k=top_k, model=model, history=history):
            if event["event"] == "route":
                route = event
            elif event["event"] == "sources":
                sources = event["sources"]
            elif event["event"] == "done":
                done = event
        return RagResult(
            answer=done["answer"],
            language=done["language"],
            refused=done["refused"],
            route=route,
            sources=sources,
            attribution=done["attribution"],
            timings=done["timings"],
        )
