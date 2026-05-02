"""Local answer generation via Ollama.

We talk to Ollama's `/api/generate` endpoint directly and stream the response
so the UI can render tokens as they arrive.

The prompt is engineered to keep answers grounded in retrieved context and
to fall back to "I don't know" when the context is insufficient.
"""
from __future__ import annotations

import json
from typing import Dict, Iterator, List, Optional

import requests

from src.config import GEN_MAX_TOKENS, GEN_MODEL, GEN_TEMPERATURE, OLLAMA_HOST


SYSTEM_PROMPT = """You are a helpful assistant that answers questions about famous people and famous places.

You will be given a user question and a set of context passages drawn from Wikipedia.

Rules you MUST follow:
1. Use ONLY the information in the provided context to answer.
2. If the context does not contain enough information to answer the question, reply with exactly: "I don't know based on the available information."
3. Do NOT invent facts, dates, names, or numbers that are not in the context.
4. Keep answers concise and factual. Prefer 1-3 short paragraphs.
5. If the question asks to compare two entities, structure the answer with a brief comparison covering the most relevant differences/similarities mentioned in the context.
6. Never mention "the context" or "the passages" in your answer — speak naturally."""


def _format_context(chunks: List[Dict]) -> str:
    """Render retrieved chunks as a numbered list the model can read."""
    lines = []
    for i, c in enumerate(chunks, start=1):
        meta = c.get("metadata") or {}
        entity = meta.get("entity", "?")
        etype = meta.get("type", "?")
        lines.append(f"[{i}] ({etype}: {entity})\n{c['document']}")
    return "\n\n".join(lines)


def build_prompt(question: str, chunks: List[Dict]) -> str:
    if not chunks:
        context_block = "(no relevant context found)"
    else:
        context_block = _format_context(chunks)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"=== CONTEXT ===\n{context_block}\n\n"
        f"=== QUESTION ===\n{question}\n\n"
        f"=== ANSWER ==="
    )


class OllamaGenerator:
    def __init__(
        self,
        model: str = GEN_MODEL,
        host: str = OLLAMA_HOST,
        temperature: float = GEN_TEMPERATURE,
        max_tokens: int = GEN_MAX_TOKENS,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._endpoint = f"{self.host}/api/generate"

    def generate(
        self,
        question: str,
        chunks: List[Dict],
        model: str | None = None,
        temperature: float | None = None,
    ) -> str:
        """Non-streaming version. Returns the full answer string."""
        prompt = build_prompt(question, chunks)
        resp = requests.post(
            self._endpoint,
            json={
                "model": model or self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature if temperature is not None else self.temperature,
                    "num_predict": self.max_tokens,
                },
            },
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()

    def stream(
        self,
        question: str,
        chunks: List[Dict],
        model: str | None = None,
        temperature: float | None = None,
    ) -> Iterator[str]:
        """Yield response tokens as they arrive."""
        prompt = build_prompt(question, chunks)
        with requests.post(
            self._endpoint,
            json={
                "model": model or self.model,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": temperature if temperature is not None else self.temperature,
                    "num_predict": self.max_tokens,
                },
            },
            stream=True,
            timeout=300,
        ) as resp:
            resp.raise_for_status()
            for raw in resp.iter_lines():
                if not raw:
                    continue
                try:
                    obj = json.loads(raw.decode("utf-8"))
                except json.JSONDecodeError:
                    continue
                tok = obj.get("response", "")
                if tok:
                    yield tok
                if obj.get("done"):
                    break