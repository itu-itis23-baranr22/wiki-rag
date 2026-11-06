"""Section-aware, sentence-aligned chunking.

Each Wikipedia section is split on its own, so a chunk never mixes topics from two
sections. Chunks are built from whole sentences, and the overlap between
neighbouring chunks is also made of whole sentences, so no chunk starts or ends in
the middle of a word.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

_BOUNDARY_RE = re.compile(r"(?<=[.!?])[\"')\]]*\s+")
_ABBREVIATIONS = re.compile(
    r"(?:\b(?:Dr|Mr|Mrs|Ms|Prof|Sr|Jr|St|Mt|Ft|No|vs|etc|c|ca|e\.g|i\.e|U\.S|U\.K|Gen|Col|Lt|Sgt|Rev|Hon|Jan|Feb|"
    r"Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)|\b[A-Z])\.$"
)


@dataclass(frozen=True)
class Chunk:
    text: str
    section: str
    index: int


def split_sentences(text: str) -> List[str]:
    """Split prose into sentences, keeping abbreviations and initials together."""
    sentences: List[str] = []
    for paragraph in re.split(r"\n+", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        pending = ""
        cursor = 0
        for match in _BOUNDARY_RE.finditer(paragraph):
            piece = paragraph[cursor : match.end()].strip()
            cursor = match.end()
            next_char = paragraph[cursor : cursor + 1]
            pending = f"{pending} {piece}".strip()
            if _ABBREVIATIONS.search(pending) or (next_char and next_char.islower()):
                continue
            sentences.append(pending)
            pending = ""
        tail = paragraph[cursor:].strip()
        pending = f"{pending} {tail}".strip()
        if pending:
            sentences.append(pending)
    return sentences


def _split_long(sentence: str, size: int) -> List[str]:
    """Break a sentence longer than `size` at word boundaries."""
    parts: List[str] = []
    current = ""
    for word in sentence.split():
        while len(word) > size:
            if current:
                parts.append(current)
                current = ""
            parts.append(word[:size])
            word = word[size:]
        candidate = f"{current} {word}".strip()
        if len(candidate) > size:
            parts.append(current)
            current = word
        else:
            current = candidate
    if current:
        parts.append(current)
    return parts


def _overlap_tail(sentences: Sequence[str], budget: int) -> List[str]:
    """The trailing whole sentences of a chunk that fit in `budget` characters."""
    tail: List[str] = []
    used = 0
    for sentence in reversed(sentences):
        cost = len(sentence) + (1 if tail else 0)
        if used + cost > budget:
            break
        tail.insert(0, sentence)
        used += cost
    return tail


def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 150, chunk_min: int = 120) -> List[str]:
    """Split text into chunks of at most `chunk_size` characters, on sentence boundaries."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    atoms: List[str] = []
    for sentence in split_sentences(text):
        atoms.extend([sentence] if len(sentence) <= chunk_size else _split_long(sentence, chunk_size))
    if not atoms:
        return []

    chunks: List[List[str]] = []
    current: List[str] = []
    length = 0
    fresh = 0  # sentences in `current` that are not overlap from the previous chunk
    for atom in atoms:
        cost = len(atom) + (1 if current else 0)
        if current and length + cost > chunk_size:
            chunks.append(current)
            budget = min(chunk_overlap, chunk_size - len(atom) - 1)
            current = _overlap_tail(current, budget)
            length = sum(len(s) for s in current) + max(0, len(current) - 1)
            fresh = 0
            cost = len(atom) + (1 if current else 0)
        current.append(atom)
        length += cost
        fresh += 1
    if fresh:
        chunks.append(current)

    joined = [" ".join(c) for c in chunks]
    if len(joined) > 1 and len(joined[-1]) < chunk_min:
        unseen = [s for s in chunks[-1] if s not in chunks[-2]]
        merged = " ".join([joined[-2], *unseen])
        if len(merged) <= chunk_size:
            joined[-2:] = [merged]
    return joined


def chunk_sections(
    sections: Iterable[Tuple[str, str]],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    chunk_min: int = 120,
) -> List[Chunk]:
    """Chunk `(section title, text)` pairs in order, numbering chunks across the page."""
    out: List[Chunk] = []
    for title, text in sections:
        for piece in chunk_text(text, chunk_size, chunk_overlap, chunk_min):
            out.append(Chunk(text=piece, section=title, index=len(out)))
    return out
