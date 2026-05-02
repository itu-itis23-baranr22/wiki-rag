"""Paragraph-aware overlapping chunker.

We deliberately don't use a library like LangChain's RecursiveCharacterTextSplitter
because the project asks us to use language-native functionality where possible.

Strategy:
1. Split text into paragraphs (blank-line delimited).
2. Greedily pack paragraphs into a chunk until adding the next paragraph would
   exceed `chunk_size`. If a single paragraph is larger than `chunk_size` we
   sub-split it on sentence boundaries, then on word boundaries as a last resort.
3. Between chunks, include a `chunk_overlap` tail of the previous chunk so that
   facts spanning chunk boundaries are recoverable.
"""
from __future__ import annotations

import re
from typing import List

# Sentence boundary heuristic — good enough for English Wikipedia prose.
# We avoid heavy NLP libs to keep the system lightweight.
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'\(])")


def _split_paragraphs(text: str) -> List[str]:
    """Split on blank lines, strip whitespace, drop empties."""
    paras = re.split(r"\n\s*\n", text.strip())
    return [p.strip() for p in paras if p.strip()]


def _split_sentences(paragraph: str) -> List[str]:
    """Naive sentence splitter — good enough for our use case."""
    parts = _SENT_RE.split(paragraph.strip())
    return [s.strip() for s in parts if s.strip()]


def _hard_split(text: str, size: int) -> List[str]:
    """Last-resort word-level split for paragraphs without sentence boundaries."""
    words = text.split()
    out, cur, cur_len = [], [], 0
    for w in words:
        # +1 for the space we'll add when joining
        if cur and cur_len + len(w) + 1 > size:
            out.append(" ".join(cur))
            cur, cur_len = [w], len(w)
        else:
            cur.append(w)
            cur_len += len(w) + (1 if cur_len else 0)
    if cur:
        out.append(" ".join(cur))
    return out


def _atomize(text: str, max_size: int) -> List[str]:
    """Break text into atomic units no bigger than `max_size`.

    Atoms are paragraphs, or — if a paragraph is too long — sentences, or
    — if a sentence is too long — fixed-size word groups.
    """
    atoms: List[str] = []
    for para in _split_paragraphs(text):
        if len(para) <= max_size:
            atoms.append(para)
            continue
        for sent in _split_sentences(para):
            if len(sent) <= max_size:
                atoms.append(sent)
            else:
                atoms.extend(_hard_split(sent, max_size))
    return atoms


def chunk_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> List[str]:
    """Return a list of chunks of approximately `chunk_size` characters with
    `chunk_overlap` characters of overlap between consecutive chunks.

    Each chunk is the concatenation of one or more atoms (paragraphs/sentences)
    so we never split mid-sentence unless a single sentence is itself larger
    than `chunk_size`.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    atoms = _atomize(text, chunk_size)
    if not atoms:
        return []

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for atom in atoms:
        # +1 for the join separator " " between atoms
        sep = 1 if current else 0
        if current and current_len + sep + len(atom) > chunk_size:
            # flush
            chunks.append(" ".join(current))
            # build overlap tail from the chunk we just emitted
            tail = chunks[-1][-chunk_overlap:] if chunk_overlap else ""
            current = [tail, atom] if tail else [atom]
            current_len = sum(len(x) for x in current) + max(0, len(current) - 1)
        else:
            current.append(atom)
            current_len += sep + len(atom)

    if current:
        chunks.append(" ".join(current))

    # Defensive: drop empty/whitespace-only and collapse internal whitespace
    return [re.sub(r"\s+", " ", c).strip() for c in chunks if c.strip()]


if __name__ == "__main__":
    # tiny smoke test
    sample = (
        "Albert Einstein was a German-born physicist.\n\n"
        "He developed the theory of relativity, one of the two pillars of "
        "modern physics. " * 20
        + "\n\nHe received the 1921 Nobel Prize in Physics."
    )
    out = chunk_text(sample, chunk_size=300, chunk_overlap=50)
    for i, c in enumerate(out):
        print(f"--- chunk {i} ({len(c)} chars) ---")
        print(c[:120], "...")
