"""Persistent vector store: one ChromaDB collection with page metadata.

Embeddings are always computed by the caller and passed in, so Chroma never needs
its own embedding function. A small manifest records how the index was built,
because vectors made with different settings must not be mixed.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import chromadb
from chromadb.config import Settings

from src.config import COLLECTION_NAME, INDEX_DIR

_PAGE_SIZE = 1000


class IndexMismatch(RuntimeError):
    """The index on disk was built with different settings and has to be rebuilt."""


class WikiVectorStore:
    def __init__(self, path: Path = INDEX_DIR, collection: str = COLLECTION_NAME):
        self.path = Path(path)
        self.collection_name = collection
        self.path.mkdir(parents=True, exist_ok=True)
        self._stats: Optional[Dict[str, Dict]] = None
        # Telemetry is off: answering questions must not contact any remote service.
        self.client = chromadb.PersistentClient(path=str(self.path), settings=Settings(anonymized_telemetry=False))
        self.collection = self._open()

    def _open(self):
        return self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def _manifest_path(self) -> Path:
        return self.path / "manifest.json"

    def read_manifest(self) -> Dict:
        if not self._manifest_path.exists():
            return {}
        return json.loads(self._manifest_path.read_text(encoding="utf-8"))

    def write_manifest(self, manifest: Dict) -> None:
        self._manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def check_manifest(self, expected: Dict) -> None:
        """Raise IndexMismatch if a non-empty index was built with other settings."""
        current = self.read_manifest()
        if self.count() and current != expected:
            raise IndexMismatch(
                "The index was built with different settings. Rebuild it with: python -m src.ingest --reset"
            )

    # ------------------------------------------------------------------ write
    def upsert(
        self,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Sequence[Dict],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        if ids:
            self._stats = None
            self.collection.upsert(
                ids=list(ids),
                documents=list(documents),
                metadatas=list(metadatas),
                embeddings=[list(e) for e in embeddings],
            )

    def delete_entity(self, title: str) -> None:
        self._stats = None
        self.collection.delete(where={"entity": title})

    def reset(self) -> None:
        self._stats = None
        self.client.delete_collection(self.collection_name)
        self.collection = self._open()
        self._manifest_path.unlink(missing_ok=True)

    # ------------------------------------------------------------------- read
    def count(self) -> int:
        return self.collection.count()

    def all_records(self) -> List[Dict]:
        """Every stored passage as {id, document, metadata}."""
        records: List[Dict] = []
        total = self.count()
        for offset in range(0, total, _PAGE_SIZE):
            page = self.collection.get(include=["documents", "metadatas"], limit=_PAGE_SIZE, offset=offset)
            records.extend(
                {"id": i, "document": d, "metadata": m}
                for i, d, m in zip(page["ids"], page["documents"], page["metadatas"], strict=True)
            )
        return records

    def entity_stats(self) -> Dict[str, Dict]:
        """Per-page facts: passage count, URL, type, revision and content hash. Cached until a write."""
        if self._stats is not None:
            return self._stats
        stats: Dict[str, Dict] = {}
        total = self.count()
        for offset in range(0, total, _PAGE_SIZE):
            page = self.collection.get(include=["metadatas"], limit=_PAGE_SIZE, offset=offset)
            for meta in page["metadatas"]:
                entry = stats.setdefault(
                    meta["entity"],
                    {
                        "passages": 0,
                        "url": meta.get("source", ""),
                        "type": meta.get("type", "other"),
                        "revision": meta.get("revision"),
                        "content_hash": meta.get("content_hash"),
                    },
                )
                entry["passages"] += 1
        self._stats = stats
        return stats

    def query(
        self,
        embedding: Sequence[float],
        top_k: int = 5,
        type_filter: Optional[str] = None,
        entity_filter: Optional[Sequence[str]] = None,
    ) -> List[Dict]:
        """Nearest passages, closest first, as {id, document, metadata, distance}."""
        clauses: List[Dict] = []
        if type_filter:
            clauses.append({"type": type_filter})
        if entity_filter:
            clauses.append({"entity": {"$in": list(entity_filter)}})
        where: Optional[Dict] = None
        if len(clauses) == 1:
            where = clauses[0]
        elif clauses:
            where = {"$and": clauses}

        result = self.collection.query(query_embeddings=[list(embedding)], n_results=top_k, where=where)
        return [
            {"id": i, "document": d, "metadata": m, "distance": dist}
            for i, d, m, dist in zip(
                result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0], strict=True
            )
        ]

    def close(self) -> None:
        """Release file handles, used before removing a temporary index in tests."""
        try:
            self.client.close()
        except AttributeError:
            pass


def remove_index(path: Path = INDEX_DIR) -> None:
    shutil.rmtree(path, ignore_errors=True)
