"""Building and updating the index.

    python -m src.ingest                 # add what is missing or changed
    python -m src.ingest --refresh       # also re-download every page
    python -m src.ingest --reset         # rebuild everything from the cached pages

Downloaded pages are cached, and a page whose chunks have not changed is skipped
without re-embedding, so an interrupted run can simply be started again.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from src.catalog import Catalog, PageEntry
from src.chunker import Chunk, chunk_sections
from src.config import CHUNK_MIN, CHUNK_OVERLAP, CHUNK_SIZE, EMBED_MODEL, PAGES_DIR
from src.embedder import OllamaEmbedder
from src.textnorm import fold
from src.vector_store import WikiVectorStore
from src.wiki import Page, load_page

SCHEMA_VERSION = 2
Progress = Callable[[Dict], None]


@dataclass
class IngestReport:
    title: str
    passages: int
    skipped: bool


def index_manifest(embedder: OllamaEmbedder) -> Dict:
    """Settings that determine the vectors. A change means the index must be rebuilt."""
    return {
        "schema": SCHEMA_VERSION,
        "embedder": embedder.fingerprint,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "chunk_min": CHUNK_MIN,
    }


def embedding_text(entity: str, chunk: Chunk) -> str:
    """Text that gets embedded: the passage plus where it comes from.

    A passage often says only "he" or "the tower", so the title and section give the
    vector the context the sentence leaves out.
    """
    heading = f"{entity}, {chunk.section}" if chunk.section else entity
    return f"{heading}\n{chunk.text}"


def _content_hash(chunks: List[Chunk]) -> str:
    payload = json.dumps([(c.section, c.text) for c in chunks], ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _page_slug(title: str) -> str:
    digest = hashlib.sha1(title.encode("utf-8")).hexdigest()[:8]
    return f"{fold(title).replace(' ', '-')}-{digest}"


def index_page(
    store: WikiVectorStore,
    embedder: OllamaEmbedder,
    entry: PageEntry,
    page: Page,
    existing_hash: Optional[str] = None,
    on_progress: Optional[Progress] = None,
) -> IngestReport:
    """Chunk and embed one page, replacing its old passages. Unchanged pages are skipped."""
    chunks = chunk_sections(page.sections, CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_MIN)
    content_hash = _content_hash(chunks)
    if existing_hash == content_hash:
        return IngestReport(entry.title, len(chunks), skipped=True)

    def report(done: int, total: int) -> None:
        if on_progress:
            on_progress({"stage": "embedding", "title": entry.title, "done": done, "total": total})

    embeddings = embedder.embed_documents([embedding_text(entry.title, c) for c in chunks], on_progress=report)
    slug = _page_slug(entry.title)
    metadatas = [
        {
            "entity": entry.title,
            "type": entry.type,
            "chunk_idx": c.index,
            "n_chunks": len(chunks),
            "section": c.section,
            "source": page.url,
            "revision": page.revision,
            "content_hash": content_hash,
        }
        for c in chunks
    ]
    store.delete_entity(entry.title)
    store.upsert(
        ids=[f"{slug}::{c.index}" for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=metadatas,
        embeddings=embeddings,
    )
    return IngestReport(entry.title, len(chunks), skipped=False)


def run(reset: bool = False, refresh: bool = False, only: Optional[List[str]] = None) -> int:
    """Index the catalog from the command line. Returns the number of failed pages."""
    store = WikiVectorStore()
    embedder = OllamaEmbedder()
    catalog = Catalog()
    manifest = index_manifest(embedder)

    if reset:
        print("Clearing the index...")
        store.reset()
    store.check_manifest(manifest)
    store.write_manifest(manifest)

    entries = [e for e in catalog.entries() if not only or e.title in only]
    stats = store.entity_stats()
    failures = 0
    started = time.time()
    for position, entry in enumerate(entries, start=1):
        print(f"[{position:>3}/{len(entries)}] {entry.type:<7} {entry.title:<32} ", end="", flush=True)

        def show(update: Dict) -> None:
            sys.stdout.write(".")
            sys.stdout.flush()

        try:
            page = load_page(entry.title, refresh=refresh, pages_dir=PAGES_DIR)
            result = index_page(
                store,
                embedder,
                entry,
                page,
                existing_hash=stats.get(entry.title, {}).get("content_hash"),
                on_progress=show,
            )
        except Exception as exc:  # one bad page should not lose the rest of the run
            failures += 1
            print(f" FAILED: {exc}", flush=True)
            continue
        print(f" {result.passages} passages{' (unchanged)' if result.skipped else ''}", flush=True)

    print(f"\nDone in {time.time() - started:.0f}s. The index holds {store.count()} passages.")
    if failures:
        print(f"{failures} page(s) failed. Run the same command again to retry them.")
    print(f"Embedding model: {EMBED_MODEL}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Build or update the local Wikipedia index.")
    parser.add_argument("--reset", action="store_true", help="delete the index and rebuild it")
    parser.add_argument("--refresh", action="store_true", help="download pages again instead of using the cache")
    parser.add_argument("--only", action="append", metavar="TITLE", help="process only this page (repeatable)")
    args = parser.parse_args()
    return 1 if run(reset=args.reset, refresh=args.refresh, only=args.only) else 0


if __name__ == "__main__":
    raise SystemExit(main())
