"""In-memory BM25 keyword search over the indexed passages.

Vector search understands paraphrases but is weak on specific terms such as a
country name; a keyword index is the opposite. The retriever combines both.
Implemented directly because the corpus fits in memory and the formula is short.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from src.textnorm import search_terms

K1 = 1.2
B = 0.75
TITLE_BOOST = 2  # page title terms count this many extra times in every passage


class LexicalIndex:
    def __init__(self, records: Iterable[Dict]):
        self._records: List[Dict] = []
        self._lengths: List[int] = []
        self._postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self._leads: Dict[str, Dict] = {}
        for record in records:
            meta = record["metadata"]
            if meta.get("chunk_idx") == 0:
                self._leads[meta.get("entity", "")] = record
            terms = (
                search_terms(record["document"])
                + search_terms(meta.get("section", ""))
                + search_terms(meta.get("entity", "")) * TITLE_BOOST
            )
            index = len(self._records)
            self._records.append(record)
            self._lengths.append(len(terms))
            for term, count in Counter(terms).items():
                self._postings[term].append((index, count))
        self._average_length = sum(self._lengths) / len(self._lengths) if self._lengths else 0.0

    def __len__(self) -> int:
        return len(self._records)

    def lead(self, entity: str) -> Optional[Dict]:
        """A page's opening passage, usually the best summary of the subject."""
        return self._leads.get(entity)

    def _idf(self, term: str) -> float:
        frequency = len(self._postings.get(term, ()))
        return math.log(1 + (len(self._records) - frequency + 0.5) / (frequency + 0.5))

    def search(
        self,
        query: str,
        top_k: int = 40,
        type_filter: Optional[str] = None,
        entity_filter: Optional[Sequence[str]] = None,
    ) -> List[Dict]:
        """Best-matching passages as {id, document, metadata, lexical_score}."""
        allowed = set(entity_filter) if entity_filter else None
        scores: Dict[int, float] = defaultdict(float)
        for term in set(search_terms(query)):
            idf = self._idf(term)
            for index, count in self._postings.get(term, ()):
                meta = self._records[index]["metadata"]
                if type_filter and meta.get("type") != type_filter:
                    continue
                if allowed and meta.get("entity") not in allowed:
                    continue
                norm = K1 * (1 - B + B * self._lengths[index] / self._average_length)
                scores[index] += idf * count * (K1 + 1) / (count + norm)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        return [{**self._records[index], "lexical_score": score} for index, score in ranked]
