"""Query routing and retrieval.

The router decides whether a question is about people, places, or both, then
asks the vector store for the most relevant chunks (filtered or unfiltered
accordingly).

The classification is intentionally simple — rule-based — because the brief
explicitly says keyword/rule-based approaches are acceptable, and because
a small static system like ours has a known, finite vocabulary of entities.

Decision algorithm:

  1. Look for any canonical entity name (or alias) in the lowered query.
     Collect their types (person/place).
       - If only "person" entities matched → route to people.
       - If only "place" entities matched   → route to places.
       - If both kinds matched              → route to both.
  2. If no entity matched, fall back to keyword cues
     (PERSON_KEYWORDS / PLACE_KEYWORDS).
       - Unique winner → route to that type.
       - Tie or none   → route to both (let semantic search decide).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Set

from src.config import (
    ALIASES,
    PEOPLE,
    PERSON_KEYWORDS,
    PLACE_KEYWORDS,
    PLACES,
    TOP_K,
    TOP_K_BOTH,
)
from src.vector_store import WikiVectorStore

# Build a single dict {alias_lower: (canonical_name, type)} once.
def _build_alias_index() -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for name in PEOPLE:
        out[name.lower()] = {"name": name, "type": "person"}
    for name in PLACES:
        out[name.lower()] = {"name": name, "type": "place"}
    for alias, canonical in ALIASES.items():
        if canonical in PEOPLE:
            out[alias.lower()] = {"name": canonical, "type": "person"}
        elif canonical in PLACES:
            out[alias.lower()] = {"name": canonical, "type": "place"}
    return out


_ALIAS_INDEX = _build_alias_index()
# Sort aliases by length, longest first, so "great wall" matches before "wall".
_SORTED_ALIASES = sorted(_ALIAS_INDEX.keys(), key=len, reverse=True)
_WORD_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)


@dataclass
class RouteDecision:
    target: str                          # "person", "place", or "both"
    matched_entities: List[Dict] = field(default_factory=list)
    rationale: str = ""


def classify(query: str) -> RouteDecision:
    q = query.lower()

    # 1. Entity-name detection (longest match first, with word-boundary check)
    matched: List[Dict] = []
    seen_canonical: Set[str] = set()
    for alias in _SORTED_ALIASES:
        # word-boundary match — \b doesn't always behave well with non-ASCII
        # so we use a manual boundary check.
        idx = 0
        while True:
            pos = q.find(alias, idx)
            if pos == -1:
                break
            left_ok = pos == 0 or not q[pos - 1].isalnum()
            right_end = pos + len(alias)
            right_ok = right_end == len(q) or not q[right_end].isalnum()
            if left_ok and right_ok:
                info = _ALIAS_INDEX[alias]
                if info["name"] not in seen_canonical:
                    matched.append(info)
                    seen_canonical.add(info["name"])
                break
            idx = pos + 1

    types_in_match = {m["type"] for m in matched}
    if types_in_match == {"person"}:
        return RouteDecision("person", matched, "matched person entity name(s)")
    if types_in_match == {"place"}:
        return RouteDecision("place", matched, "matched place entity name(s)")
    if types_in_match == {"person", "place"}:
        return RouteDecision("both", matched, "matched entities of both types")

    # 2. Keyword-cue fallback
    tokens = set(_WORD_RE.findall(q))
    person_score = len(tokens & PERSON_KEYWORDS)
    place_score = len(tokens & PLACE_KEYWORDS)
    if person_score > place_score:
        return RouteDecision("person", matched, f"keyword cues ({person_score} vs {place_score})")
    if place_score > person_score:
        return RouteDecision("place", matched, f"keyword cues ({place_score} vs {person_score})")
    return RouteDecision("both", matched, "no clear cues — searching both")


# --------------------------------------------------------------------------- #
# Retrieval                                                                   #
# --------------------------------------------------------------------------- #
class Retriever:
    def __init__(self, store: WikiVectorStore):
        self.store = store

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        top_k_both: int | None = None,
    ) -> Dict:
        """Run the full route + retrieve pipeline.

        `top_k` and `top_k_both` override the values from config.py for this
        call only. This lets the UI expose a Top-K slider without having to
        rebuild the whole RAG system.

        Returns a dict:
            {
                "route": RouteDecision,
                "chunks": [ {id, document, metadata, distance}, ... ],
            }
        Chunks are sorted ascending by distance (best first).

        When the router has already identified specific entities by name, we
        scope the vector search to JUST those entities (via the `entity`
        metadata filter). Reasons:
          * For short queries like "Albert Einstein" or "Compare Messi and
            Ronaldo", semantic search across the whole index is dominated by
            high-level category similarity (e.g. "famous woman" or "historical
            figure") rather than name match. Scoping to the entities we already
            recognised avoids that failure mode entirely.
          * It also gives the LLM a clean, focused context window — only chunks
            from the entities the user actually asked about — which reduces
            comparison-question confusion.
        Anonymous queries (no entity matched) still fall back to broad
        semantic search filtered only by type, as before.
        """
        eff_top_k = top_k if top_k is not None else TOP_K
        eff_top_k_both = top_k_both if top_k_both is not None else TOP_K_BOTH
        route = classify(query)
        matched_names = [m["name"] for m in route.matched_entities]

        if matched_names:
            # Scoped search — split chunks roughly evenly across the named
            # entities so a comparison question gets context from BOTH.
            per_entity_k = max(3, eff_top_k // max(1, len(matched_names)))
            collected: List[Dict] = []
            for name in matched_names:
                hits = self.store.query(
                    query,
                    top_k=per_entity_k,
                    entity_filter=[name],
                )
                collected.extend(hits)
            chunks = sorted(collected, key=lambda x: x["distance"])
            return {"route": route, "chunks": chunks}

        # No entity name matched — fall back to type-filtered semantic search.
        if route.target == "both":
            person_hits = self.store.query(query, top_k=eff_top_k_both, type_filter="person")
            place_hits = self.store.query(query, top_k=eff_top_k_both, type_filter="place")
            chunks = sorted(person_hits + place_hits, key=lambda x: x["distance"])
        else:
            chunks = self.store.query(query, top_k=eff_top_k, type_filter=route.target)

        return {"route": route, "chunks": chunks}