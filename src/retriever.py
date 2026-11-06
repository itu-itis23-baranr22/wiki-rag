"""Hybrid retrieval: vector search and BM25, merged by reciprocal rank fusion.

When a question names pages, search stays inside them and shares the passage budget
between them, so a comparison gets context from both sides. When it names none,
search runs across the corpus with a per-page cap, so a question like "which famous
place is in Turkey?" can surface several pages rather than five passages from one.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence

from src.config import CANDIDATE_POOL, ENTITY_CAP, LEXICAL_WEIGHT, RRF_K, TOP_K, VECTOR_WEIGHT
from src.embedder import OllamaEmbedder
from src.lexical import LexicalIndex
from src.router import RouteDecision, Router
from src.textnorm import search_terms
from src.vector_store import WikiVectorStore


@dataclass(frozen=True)
class RetrievalSettings:
    """Switches for each technique, so the evaluation can measure them separately."""

    use_vector: bool = True
    use_lexical: bool = True
    type_filter: bool = True  # firm keyword cues, such as "place", restrict search to that type
    entity_cap: int = ENTITY_CAP
    vector_weight: float = VECTOR_WEIGHT
    lexical_weight: float = LEXICAL_WEIGHT
    pool: int = CANDIDATE_POOL

    def with_changes(self, **changes) -> RetrievalSettings:
        return replace(self, **changes)


@dataclass
class RetrievalResult:
    route: RouteDecision
    chunks: List[Dict] = field(default_factory=list)
    search_query: str = ""


def fuse(vector_hits: List[Dict], lexical_hits: List[Dict], settings: RetrievalSettings) -> List[Dict]:
    """Reciprocal rank fusion. Each passage keeps its rank in both source lists."""
    merged: Dict[str, Dict] = {}
    for rank, hit in enumerate(vector_hits, start=1):
        merged[hit["id"]] = {
            **hit,
            "vector_rank": rank,
            "lexical_rank": None,
            "score": settings.vector_weight / (RRF_K + rank),
        }
    for rank, hit in enumerate(lexical_hits, start=1):
        addition = settings.lexical_weight / (RRF_K + rank)
        if hit["id"] in merged:
            merged[hit["id"]]["lexical_rank"] = rank
            merged[hit["id"]]["score"] += addition
        else:
            merged[hit["id"]] = {
                **hit,
                "distance": None,
                "vector_rank": None,
                "lexical_rank": rank,
                "score": addition,
            }
    return sorted(merged.values(), key=lambda item: item["score"], reverse=True)


def asks_for_overview(text: str, names: Sequence[str]) -> bool:
    """True for questions like "Who was X?" or "Compare X and Y", with no specific topic."""
    name_terms = {term for name in names for term in search_terms(name)}
    remaining = [term for term in search_terms(text) if term not in name_terms]
    return len(remaining) <= 1


def with_lead(picked: List[Dict], lead: Optional[Dict]) -> List[Dict]:
    """Put a page's opening passage first, keeping the same number of passages."""
    if lead is None or any(hit["id"] == lead["id"] for hit in picked):
        return picked
    top_score = picked[0]["score"] if picked else 1.0
    entry = {**lead, "distance": None, "vector_rank": None, "lexical_rank": None, "score": top_score}
    return [entry, *picked[:-1]] if picked else [entry]


def cap_per_entity(ranked: Sequence[Dict], limit: int, cap: int) -> List[Dict]:
    """Take up to `limit` passages, at most `cap` from any page unless too few pages match."""
    chosen: List[Dict] = []
    counts: Dict[str, int] = {}
    for hit in ranked:
        name = hit["metadata"]["entity"]
        if counts.get(name, 0) < cap:
            chosen.append(hit)
            counts[name] = counts.get(name, 0) + 1
            if len(chosen) == limit:
                return chosen
    chosen_ids = {hit["id"] for hit in chosen}
    chosen.extend(hit for hit in ranked if hit["id"] not in chosen_ids)
    return chosen[:limit]


class Retriever:
    def __init__(
        self,
        store: WikiVectorStore,
        embedder: OllamaEmbedder,
        router: Router,
        lexical: Optional[LexicalIndex] = None,
        settings: Optional[RetrievalSettings] = None,
    ):
        self.store = store
        self.embedder = embedder
        self.router = router
        self.settings = settings or RetrievalSettings()
        self.lexical = lexical if lexical is not None else LexicalIndex(store.all_records())

    def refresh(self) -> None:
        """Pick up pages added or removed since startup."""
        self.lexical = LexicalIndex(self.store.all_records())
        self.router.refresh()

    def _search(self, embedding, text: str, type_filter=None, entity_filter=None) -> List[Dict]:
        pool = self.settings.pool
        vector_hits = (
            self.store.query(embedding, top_k=pool, type_filter=type_filter, entity_filter=entity_filter)
            if self.settings.use_vector
            else []
        )
        lexical_hits = (
            self.lexical.search(text, top_k=pool, type_filter=type_filter, entity_filter=entity_filter)
            if self.settings.use_lexical
            else []
        )
        return fuse(vector_hits, lexical_hits, self.settings)

    def retrieve(
        self,
        question: str,
        top_k: Optional[int] = None,
        translated: Optional[str] = None,
        previous_entities: Optional[Sequence[str]] = None,
    ) -> RetrievalResult:
        """Route the question and return its passages, best first.

        `top_k` is the total passage count. When several pages are named, each still
        gets at least one passage, so the total can exceed a very small `top_k`.
        """
        limit = top_k if top_k is not None else TOP_K
        route = self.router.classify(question, translated=translated, previous_entities=previous_entities)
        text = translated or question
        if route.reason == "followup":
            text = f"{' '.join(route.entity_names)} {text}"
        embedding = self.embedder.embed_query(text) if self.settings.use_vector else None

        names = route.entity_names
        if names:
            quota = max(1, limit // len(names))
            extra = max(0, limit - quota * len(names))
            overview = len(names) > 1 or asks_for_overview(text, names)
            chunks: List[Dict] = []
            for position, name in enumerate(names):
                ranked = self._search(embedding, text, entity_filter=[name])
                picked = ranked[: quota + (1 if position < extra else 0)]
                chunks.extend(with_lead(picked, self.lexical.lead(name)) if overview else picked)
            chunks.sort(key=lambda item: item["score"], reverse=True)
        else:
            type_filter = route.target if self.settings.type_filter and route.strong_cue else None
            ranked = self._search(embedding, text, type_filter=type_filter)
            chunks = cap_per_entity(ranked, limit, self.settings.entity_cap)
        return RetrievalResult(route=route, chunks=chunks, search_query=text)
