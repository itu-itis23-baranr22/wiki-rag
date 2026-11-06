"""Deciding which pages a question is about.

The rules, in order:
  1. Exact page names and aliases, ignoring case and accents. Aliases that are also
     ordinary words ("jobs", "swift") count only when capitalised.
  2. Close misspellings of a name.
  3. A follow-up such as "where was she born?" reuses the previous turn's pages.
  4. Otherwise keyword cues suggest a type, and search runs across all pages.

All of it is deterministic, needs no model and runs in under a millisecond.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from src.catalog import Catalog
from src.config import CASE_SENSITIVE_ALIASES, PERSON_KEYWORDS, PLACE_KEYWORDS, WEAK_KEYWORDS
from src.textnorm import STOP_WORDS, fold

FUZZY_CUTOFF = 0.85
FUZZY_MIN_LENGTH = 5

# Noun case endings that follow a name in Turkish without an apostrophe ("Teslanın"),
# already folded. Longest first is not needed: the whole remainder must match.
_TURKISH_SUFFIXES = frozenset(
    "nin nun in un yi yu i u ye ya e a de da den dan te ta ten tan le la yle yla ler lar deki daki nde nda".split()
)
_FOLLOWUP_WORDS = frozenset(
    """
    he she it his her hers him its they them their there this that those these
    o onun ona onu ondan orada orasi oraya bunun bu onlar onlarin
    """.split()
)
_FOLLOWUP_OPENERS = ("and ", "what about ", "how about ", "peki", "ya ")
_WORD = re.compile(r"\w+", flags=re.UNICODE)

# Words too common to treat as a misspelt name, on top of the search stop words.
_FUZZY_IGNORE = frozenset(
    """
    during tower mount falls statue where which world their first after before other great house
    """.split()
)


@dataclass
class RouteDecision:
    target: str  # "person", "place" or "both"
    matched_entities: List[Dict] = field(default_factory=list)
    reason: str = "none"  # "entity", "fuzzy", "followup", "keywords" or "none"
    rationale: str = ""
    strong_cue: bool = False  # the keyword evidence is firm enough to exclude other types

    @property
    def entity_names(self) -> List[str]:
        return [m["name"] for m in self.matched_entities]


def is_followup(question: str) -> bool:
    folded = fold(question).strip()
    if folded.startswith(_FOLLOWUP_OPENERS):
        return True
    return bool(set(_WORD.findall(folded)) & _FOLLOWUP_WORDS)


class Router:
    def __init__(self, catalog: Optional[Catalog] = None, fuzzy: bool = True, followups: bool = True):
        self.catalog = catalog or Catalog()
        self.fuzzy = fuzzy
        self.followups = followups
        self.refresh()

    def refresh(self) -> None:
        """Rebuild the name tables. Call after pages are added or removed."""
        self._types = {entry.title: entry.type for entry in self.catalog.entries()}
        self._aliases: Dict[str, str] = {fold(alias): title for alias, title in self.catalog.aliases().items()}
        self._ordered = sorted(self._aliases, key=len, reverse=True)
        case_sensitive = {fold(alias) for alias in CASE_SENSITIVE_ALIASES}
        self._fuzzy_keys = [a for a in self._ordered if len(a) >= FUZZY_MIN_LENGTH and a not in case_sensitive]

    def _exact(self, text: str) -> Tuple[List[Tuple[int, str]], str]:
        """Return (position, title) matches and the folded text with those spans blanked."""
        folded = fold(text)
        found: List[Tuple[int, str]] = []
        for alias in self._ordered:
            cased = CASE_SENSITIVE_ALIASES.get(alias)
            if cased and not re.search(rf"\b{cased}\b", text):
                continue
            start = 0
            while (position := folded.find(alias, start)) != -1:
                start = position + 1
                end = position + len(alias)
                if position > 0 and folded[position - 1].isalnum():
                    continue
                if end < len(folded) and folded[end].isalnum():
                    tail = _WORD.match(folded, end)
                    if len(alias) < 4 or not tail or tail.group() not in _TURKISH_SUFFIXES:
                        continue
                    end = tail.end()
                found.append((position, self._aliases[alias]))
                folded = folded[:position] + " " * (end - position) + folded[end:]
        return found, folded

    def _misspelt(self, masked: str) -> List[Tuple[int, str]]:
        """Find close matches for words not already explained by an exact match."""
        words = [(m.start(), m.group()) for m in _WORD.finditer(masked)]
        found: List[Tuple[int, str]] = []
        for size in (3, 2, 1):
            for i in range(len(words) - size + 1):
                window = words[i : i + size]
                if window[0][1] in STOP_WORDS or window[-1][1] in STOP_WORDS:
                    continue
                if size == 1 and window[0][1] in _FUZZY_IGNORE:
                    continue
                phrase = " ".join(w for _, w in window)
                if len(phrase) < FUZZY_MIN_LENGTH:
                    continue
                best, score = None, 0.0
                for key in self._fuzzy_keys:
                    if key.count(" ") != size - 1 or abs(len(key) - len(phrase)) > 2:
                        continue
                    ratio = SequenceMatcher(None, phrase, key).ratio()
                    if ratio > score:
                        best, score = key, ratio
                if best and score >= FUZZY_CUTOFF:
                    found.append((window[0][0], self._aliases[best]))
        return found

    def _entities(self, names: Iterable[str]) -> List[Dict]:
        seen: Set[str] = set()
        out: List[Dict] = []
        for name in names:
            if name not in seen and name in self._types:
                seen.add(name)
                out.append({"name": name, "type": self._types[name]})
        return out

    def _decision_for(self, entities: List[Dict], reason: str, rationale: str) -> RouteDecision:
        types = {e["type"] for e in entities}
        target = types.pop() if len(types) == 1 and "other" not in types else "both"
        return RouteDecision(target, entities, reason, rationale)

    def classify(
        self,
        question: str,
        translated: Optional[str] = None,
        previous_entities: Optional[Sequence[str]] = None,
    ) -> RouteDecision:
        texts = [question] + ([translated] if translated else [])
        exact: List[Tuple[int, str]] = []
        misspelt: List[Tuple[int, str]] = []
        for text in texts:
            matches, masked = self._exact(text)
            exact.extend(matches)
            if self.fuzzy:
                misspelt.extend(self._misspelt(masked))

        ordered = [name for _, name in sorted(exact, key=lambda m: m[0])]
        if misspelt:
            entities = self._entities(ordered + [name for _, name in sorted(misspelt, key=lambda m: m[0])])
            return self._decision_for(entities, "fuzzy", "matched a close spelling of a page name")
        if ordered:
            return self._decision_for(self._entities(ordered), "entity", "matched page name(s)")

        if self.followups and previous_entities and is_followup(question):
            entities = self._entities(previous_entities)
            if entities:
                return self._decision_for(entities, "followup", "follow-up to the previous question")

        words = set(_WORD.findall(fold(translated or question)))
        person = len(words & PERSON_KEYWORDS)
        place = len(words & PLACE_KEYWORDS)
        strong = bool(words - WEAK_KEYWORDS & (PERSON_KEYWORDS | PLACE_KEYWORDS))
        if person > place:
            return RouteDecision("person", [], "keywords", f"keyword cues ({person} vs {place})", strong)
        if place > person:
            return RouteDecision("place", [], "keywords", f"keyword cues ({place} vs {person})", strong)
        return RouteDecision("both", [], "none", "no clear cues")
