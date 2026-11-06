"""Detecting Turkish questions and translating them for retrieval.

The corpus is English Wikipedia, and both search methods work best in English, so
a Turkish question is searched through a short local translation. The answer still
comes back in Turkish.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Optional

from src.config import GEN_MODEL
from src.ollama_client import OllamaClient, OllamaError

_TURKISH_WORDS = frozenset(
    """
    ne nedir neden niçin niye nasıl nerede nereye nereden neresi hangi hangisi hangileri hangisidir kim kimdir kimin
    kime kimi kaç kaçta mı mi mu mü mıdır midir ve ile için bir bu şu onun ona onu ondan orada orası oraya bunun şunun
    peki ya da de ünlü doğdu öldü yaptı yazdı yaşadı bulunur tarafından olarak gibi daha en çok var yok hakkında
    anlat söyle karşılaştır geliştirdi keşfetti kurdu inşa kaçıncı zaman ülke şehir dağ yer yerler kişi
    """.split()
)
_ENGLISH_WORDS = frozenset(
    """
    who what when where why how which whose whom is was are were the a an of in on at to did does do tell me about
    compare and or from with for his her he she it its they their there this that write born died built
    """.split()
)
_TURKISH_LETTERS = re.compile(r"[şğı]")
_WEAK_TURKISH_LETTERS = re.compile(r"[üöç]")
_APOSTROPHE_SUFFIX = re.compile(r"\w['’](?:[ıiuü]|y[ıiuü]|n[ıiuü]n|[ıiuü]n|y[ae]|[ae]|d[ae]|d[ae]n|d[ae]ki|l[ae]r\w*)\b")
_WORD = re.compile(r"\w+", flags=re.UNICODE)

_TRANSLATION_PROMPT = (
    "Translate the user's Turkish question into English. Reply with only the English question, "
    "nothing else. Use the standard English name for people and places. Examples: "
    "Ayasofya nerede? = Where is Hagia Sophia? "
    "Fransa'daki kule hangisi? = Which tower is in France? "
    "Marie Curie ne zaman doğdu? = When was Marie Curie born? "
    "Tesla ne zaman öldü? = When did Tesla die?"
)


def turkish_lower(text: str) -> str:
    return text.replace("İ", "i").replace("I", "ı").lower()


def detect_language(text: str) -> str:
    """Return "tr" or "en". Names such as Atatürk alone do not make a question Turkish."""
    lowered = turkish_lower(text)
    words = _WORD.findall(lowered)
    turkish = 2.0 * sum(w in _TURKISH_WORDS for w in words)
    turkish += 2.0 * len(_APOSTROPHE_SUFFIX.findall(lowered))
    turkish += 1.0 * sum(bool(_TURKISH_LETTERS.search(w)) for w in words)
    turkish += 0.5 * sum(bool(_WEAK_TURKISH_LETTERS.search(w)) for w in words)
    english = 2.0 * sum(w.replace("ı", "i") in _ENGLISH_WORDS for w in words)
    return "tr" if turkish > english else "en"


class QueryTranslator:
    """Translates a question to English with the local model and remembers the result."""

    def __init__(self, client: Optional[OllamaClient] = None, model: str = GEN_MODEL):
        self.client = client or OllamaClient()
        self.model = model
        self.translate = lru_cache(maxsize=256)(self._translate)

    def _translate(self, question: str) -> Optional[str]:
        """Return the English question, or None when translation fails or adds nothing."""
        try:
            reply = self.client.chat(
                self.model,
                [
                    {"role": "system", "content": _TRANSLATION_PROMPT},
                    {"role": "user", "content": question},
                ],
                options={"temperature": 0, "num_predict": 60},
            )
        except OllamaError:
            return None
        lines = [line for line in reply.strip().splitlines() if line.strip()]
        if not lines:
            return None
        cleaned = lines[0].strip().strip("\"'“”")
        return cleaned if cleaned and cleaned.casefold() != question.casefold() else None
