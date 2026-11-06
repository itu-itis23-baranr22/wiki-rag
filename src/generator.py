"""Grounded answer generation with the local model.

The prompt keeps answers inside the retrieved passages, asks for numbered citations
and pins the exact sentence used to decline, so a refusal can be recognised reliably.
"""
from __future__ import annotations

import re
from typing import Dict, Iterator, List, Optional, Sequence

from src.config import GEN_MAX_TOKENS, GEN_MODEL, GEN_NUM_CTX, GEN_TEMPERATURE, KEEP_ALIVE
from src.ollama_client import OllamaClient
from src.textnorm import fold

REFUSALS = {
    "en": "I don't know based on the available information.",
    "tr": "Elimdeki bilgilere göre bunu bilmiyorum.",
}
LANGUAGE_NAMES = {"en": "English", "tr": "Turkish"}
# The model is told never to mention the passages, so saying they lack something is declining too.
_REFUSAL_MARKERS = (
    "i don't know", "i dont know", "i do not know", "does not mention", "do not mention", "not mentioned",
    "no information", "bilmiyorum", "bilgim yok", "bilgi yok", "bahsedilme", "belirtilme", "yer almiyor",
)
_CITATION = re.compile(r"\s*\[(?:passage |source )?\d+(?:\s*,\s*\d+)*\]", flags=re.IGNORECASE)
HISTORY_TURNS = 2
HISTORY_CHARS = 400


def system_prompt(language: str) -> str:
    refusal = REFUSALS[language]
    return f"""You answer questions about Wikipedia pages using only the numbered passages in the user's message.

Rules:
1. Use only facts stated in the passages. Do not add outside knowledge, even if you are sure of it.
2. If the passages do not answer the question, reply with exactly this sentence and nothing else: {refusal}
3. End every sentence with the number of the passage that supports it, in square brackets, like: Curie was born in Warsaw [2].
4. Be brief and direct: one short paragraph, or two for a comparison. Cover both sides of a comparison.
5. Write in {LANGUAGE_NAMES[language]}. Keep names of people and places as they are.
6. Do not mention the passages or say "according to". Just state the facts with their numbers."""


def format_passages(chunks: Sequence[Dict]) -> str:
    blocks = []
    for number, chunk in enumerate(chunks, start=1):
        meta = chunk.get("metadata") or {}
        origin = meta.get("entity", "?")
        if meta.get("section"):
            origin = f"{origin}, {meta['section']}"
        blocks.append(f"[{number}] ({origin})\n{chunk['document']}")
    return "\n\n".join(blocks)


def strip_citations(text: str) -> str:
    return _CITATION.sub("", text).strip()


def build_messages(
    question: str,
    chunks: Sequence[Dict],
    language: str = "en",
    history: Optional[Sequence[Dict]] = None,
) -> List[Dict]:
    """Assemble the chat messages. Recent turns are included to resolve words like "she"."""
    messages: List[Dict] = [{"role": "system", "content": system_prompt(language)}]
    for turn in (history or [])[-HISTORY_TURNS:]:
        messages.append({"role": "user", "content": turn["question"]})
        messages.append({"role": "assistant", "content": strip_citations(turn.get("answer", ""))[:HISTORY_CHARS]})
    passages = format_passages(chunks) if chunks else "(no passages found)"
    # Small models follow the instruction nearest the end, so the key rules are repeated here.
    reminder = (
        f"Answer in {LANGUAGE_NAMES[language]}, using only the passages. Name the person or place you mean, and end "
        f"every sentence with its passage number, like [1]. If the passages do not answer the question, reply only: "
        f"{REFUSALS[language]}"
    )
    messages.append({"role": "user", "content": f"Passages:\n\n{passages}\n\nQuestion: {question}\n\n{reminder}"})
    return messages


def is_refusal(answer: str) -> bool:
    """True when the answer declines rather than states facts."""
    opening = fold(answer.strip())[:120]
    return any(fold(marker) in opening for marker in _REFUSAL_MARKERS)


class Generator:
    def __init__(
        self,
        client: Optional[OllamaClient] = None,
        model: str = GEN_MODEL,
        temperature: float = GEN_TEMPERATURE,
        max_tokens: int = GEN_MAX_TOKENS,
    ):
        self.client = client or OllamaClient()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def stream(
        self,
        question: str,
        chunks: Sequence[Dict],
        language: str = "en",
        history: Optional[Sequence[Dict]] = None,
        model: Optional[str] = None,
    ) -> Iterator[Dict]:
        """Yield {"token": str} for each piece, then {"stats": {...}} at the end."""
        options = {"temperature": self.temperature, "num_predict": self.max_tokens, "num_ctx": GEN_NUM_CTX}
        messages = build_messages(question, chunks, language, history)
        stream = self.client.chat_stream(model or self.model, messages, options, keep_alive=KEEP_ALIVE)
        try:
            for part in stream:
                token = part.get("message", {}).get("content", "")
                if token:
                    yield {"token": token}
                if part.get("done"):
                    yield {
                        "stats": {
                            "prompt_tokens": part.get("prompt_eval_count"),
                            "answer_tokens": part.get("eval_count"),
                            "eval_seconds": (part.get("eval_duration") or 0) / 1e9,
                        }
                    }
        finally:
            stream.close()
