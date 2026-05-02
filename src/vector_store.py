"""Vector store wrapper around ChromaDB.

Design choice (Option B from the brief): one collection, with a `type`
metadata field per document set to either "person" or "place". Reasons:

  * Mixed queries ("compare Einstein and the Eiffel Tower", "which famous
    place is in Turkey") become natural single-collection searches with
    optional filters.
  * One HNSW index, one persistence dir, one set of operational concerns.
  * We can still scope a query to one type with `where={"type": ...}`,
    which is what the router does for unambiguous queries.

We persist to disk under `CHROMA_DIR` so re-running the chat doesn't
re-embed everything.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import chromadb

from src.config import CHROMA_DIR, COLLECTION_NAME
from src.embedder import OllamaEmbedder


class WikiVectorStore:
    """Thin wrapper around a single persistent Chroma collection."""

    def __init__(self, embedder: Optional[OllamaEmbedder] = None):
        self.embedder = embedder or OllamaEmbedder()
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        # `get_or_create_collection` is idempotent — reusing across runs.
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedder,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------ write
    def add(
        self,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Sequence[Dict],
    ) -> None:
        """Insert chunks. Chroma will call our embedder for us."""
        if not ids:
            return
        self.collection.add(
            ids=list(ids),
            documents=list(documents),
            metadatas=list(metadatas),
        )

    def reset(self) -> None:
        """Delete the collection and recreate it empty."""
        try:
            self.client.delete_collection(COLLECTION_NAME)
        except Exception:
            # collection might not exist — fine
            pass
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedder,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------- read
    def count(self) -> int:
        return self.collection.count()

    def query(
        self,
        text: str,
        top_k: int = 5,
        type_filter: Optional[str] = None,
        entity_filter: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Semantic search. Returns a list of dicts:
            {id, document, metadata, distance}
        sorted ascending by distance (so [0] is the closest match).

        - `type_filter`: "person" or "place" to scope by type.
        - `entity_filter`: list of canonical entity titles to restrict to.
        """
        where: Optional[Dict] = None
        clauses = []
        if type_filter:
            clauses.append({"type": type_filter})
        if entity_filter:
            clauses.append({"entity": {"$in": list(entity_filter)}})
        if len(clauses) == 1:
            where = clauses[0]
        elif len(clauses) > 1:
            where = {"$and": clauses}

        # We compute the query embedding ourselves and pass query_embeddings
        # rather than query_texts. This sidesteps fragile internal contracts
        # in Chroma about how its EmbeddingFunction is invoked at query time.
        q_emb = self.embedder.embed_one(text)
        res = self.collection.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            where=where,
        )
        # Chroma returns parallel lists keyed by query; we have one query.
        ids = res["ids"][0]
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        return [
            {"id": i, "document": d, "metadata": m, "distance": dist}
            for i, d, m, dist in zip(ids, docs, metas, dists)
        ]