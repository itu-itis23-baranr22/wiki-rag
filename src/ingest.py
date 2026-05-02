"""Wikipedia ingestion script.

Run this once before chatting:
    python -m src.ingest               # incremental
    python -m src.ingest --reset       # wipe and re-ingest

For each canonical entity in `PEOPLE` and `PLACES`:
  1. Fetch the page text from Wikipedia (English).
  2. Chunk it.
  3. Add the chunks to the Chroma collection with metadata
       {"entity": "Albert Einstein", "type": "person", "chunk_idx": 3, ...}.

We use the `wikipedia-api` library because it's a thin, well-behaved wrapper
around the MediaWiki API. We could roll our own HTTP code, but Wikipedia
asks API users to set a descriptive User-Agent and follow some etiquette,
and `wikipedia-api` already handles all of that.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import wikipediaapi

from src.chunker import chunk_text
from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    PEOPLE,
    PLACES,
    RAW_DATA_DIR,
)
from src.vector_store import WikiVectorStore


WIKI_USER_AGENT = "wiki-rag-class-project/1.0 (educational use)"


# ---------------------------------------------------------------- fetch step
def _make_wiki_client() -> wikipediaapi.Wikipedia:
    return wikipediaapi.Wikipedia(
        user_agent=WIKI_USER_AGENT,
        language="en",
        extract_format=wikipediaapi.ExtractFormat.WIKI,
    )


def fetch_page(wiki: wikipediaapi.Wikipedia, title: str) -> Optional[str]:
    """Return the full plain-text content of `title`, or None if missing."""
    page = wiki.page(title)
    if not page.exists():
        return None
    # `text` includes section headings prefixed with `=`. We strip those
    # markers so chunks read more naturally.
    text = page.text
    text = re.sub(r"^=+\s*(.*?)\s*=+\s*$", r"\1", text, flags=re.MULTILINE)
    return text.strip()


# --------------------------------------------------------------- ingest step
def ingest_entity(
    store: WikiVectorStore,
    wiki: wikipediaapi.Wikipedia,
    title: str,
    etype: str,
    raw_dir: Path,
) -> int:
    """Fetch + chunk + add to store. Returns number of chunks indexed."""
    text = fetch_page(wiki, title)
    if not text:
        print(f"  ! could not find Wikipedia page for {title!r}")
        return 0

    # Cache the raw text so re-runs without --reset can avoid re-fetching.
    raw_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_\-]", "_", title)
    (raw_dir / f"{safe}.txt").write_text(text, encoding="utf-8")

    chunks = chunk_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    if not chunks:
        return 0

    # Add chunks one at a time so the user sees progress on slow CPUs.
    # Chroma will call our embedder once per add — same total work, just
    # interactive feedback.
    metadatas: List[Dict] = []
    src_url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
    for i, chunk in enumerate(chunks):
        chunk_id = f"{etype}::{safe}::{i}"
        meta = {
            "entity": title,
            "type": etype,
            "chunk_idx": i,
            "source": src_url,
        }
        store.add(ids=[chunk_id], documents=[chunk], metadatas=[meta])
        metadatas.append(meta)
        sys.stdout.write(".")
        sys.stdout.flush()
    return len(chunks)


# ------------------------------------------------------------------ run loop
def run(reset: bool = False) -> None:
    store = WikiVectorStore()
    if reset:
        print("Resetting vector store...")
        store.reset()
        # also clear cached raw dumps
        if RAW_DATA_DIR.exists():
            for f in RAW_DATA_DIR.glob("*.txt"):
                f.unlink()
    elif store.count() > 0:
        print(
            f"Note: collection already has {store.count()} chunks. "
            f"Continuing in incremental mode (existing IDs will fail to re-add).\n"
            f"Use --reset to start clean."
        )

    wiki = _make_wiki_client()
    todo = [(t, "person") for t in PEOPLE] + [(t, "place") for t in PLACES]
    total_chunks = 0
    start = time.time()
    for i, (title, etype) in enumerate(todo, start=1):
        print(f"[{i:>2}/{len(todo)}] {etype:<7}: {title:<32} ", end="", flush=True)
        try:
            n = ingest_entity(store, wiki, title, etype, RAW_DATA_DIR)
        except Exception as e:
            print(f" FAILED: {e}", flush=True)
            continue
        total_chunks += n
        print(f" ({n} chunks)", flush=True)
    elapsed = time.time() - start
    print(
        f"\nDone. Indexed {total_chunks} chunks across {len(todo)} entities "
        f"in {elapsed:.1f}s."
    )
    print(f"Collection now has {store.count()} chunks.")


# ----------------------------------------------------------------- entrypoint
def _cli():
    ap = argparse.ArgumentParser(description="Ingest Wikipedia into the local Chroma DB.")
    ap.add_argument("--reset", action="store_true", help="Wipe the collection before ingesting.")
    args = ap.parse_args()
    run(reset=args.reset)


if __name__ == "__main__":
    _cli()