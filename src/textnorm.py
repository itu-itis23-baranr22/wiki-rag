"""Text normalisation shared by routing, lexical search and attribution."""
from __future__ import annotations

import re
import unicodedata
from typing import List

_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)

# Words that carry no search signal. Kept short on purpose: words like "who" or
# "where" matter to the router but are removed here because passages rarely use them.
STOP_WORDS = frozenset(
    """
    a an and are as at be been but by can could did do does for from had has have he her hers him his how i if in
    into is it its me my no not of on or our she so some than that the their them then there these they this those
    to us was we were what when where which while who whom whose why will with would you your famous
    about also tell located ki bir ve ile de da mi mu ne nedir kim kimdir nerede hangi hangisi hangileri
    """.split()
)


def fold(text: str) -> str:
    """Lower-case and strip accents, treating Turkish dotted and dotless i alike."""
    text = text.replace("İ", "i").replace("I", "i").replace("ı", "i").casefold()
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def tokens(text: str) -> List[str]:
    """Folded word tokens."""
    return _TOKEN_RE.findall(fold(text))


def stem(token: str) -> str:
    """Conservative English plural stripping, so "pyramids" finds "pyramid"."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def search_terms(text: str) -> List[str]:
    """Tokens for lexical matching: folded, stemmed, without stop words."""
    return [stem(t) for t in tokens(text) if t not in STOP_WORDS]
