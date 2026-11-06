"""The set of pages the system knows about: the defaults plus pages added by the user."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from src.config import ALIASES, CUSTOM_PAGES_PATH, PEOPLE, PLACES

PAGE_TYPES = ("person", "place", "other")


@dataclass(frozen=True)
class PageEntry:
    title: str
    type: str
    custom: bool = False


class Catalog:
    def __init__(self, path: Path = CUSTOM_PAGES_PATH):
        self.path = Path(path)
        self._defaults = [PageEntry(t, "person") for t in PEOPLE] + [PageEntry(t, "place") for t in PLACES]
        self._custom = self._load_custom()

    def _load_custom(self) -> List[PageEntry]:
        if not self.path.exists():
            return []
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        return [PageEntry(item["title"], item["type"], custom=True) for item in raw]

    def _save_custom(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = [{"title": e.title, "type": e.type} for e in self._custom]
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def entries(self) -> List[PageEntry]:
        return [*self._defaults, *self._custom]

    def get(self, title: str) -> Optional[PageEntry]:
        return next((e for e in self.entries() if e.title == title), None)

    def add(self, title: str, page_type: str) -> PageEntry:
        if page_type not in PAGE_TYPES:
            raise ValueError(f"Unknown page type: {page_type}")
        existing = self.get(title)
        if existing:
            return existing
        entry = PageEntry(title, page_type, custom=True)
        self._custom.append(entry)
        self._save_custom()
        return entry

    def remove(self, title: str) -> None:
        """Remove a user-added page. Default pages are part of the product and stay."""
        entry = self.get(title)
        if entry is None or not entry.custom:
            raise ValueError(f"{title!r} is not a user-added page")
        self._custom = [e for e in self._custom if e.title != title]
        self._save_custom()

    def aliases(self) -> Dict[str, str]:
        """Every name people may type for a page, mapped to its title."""
        out: Dict[str, str] = {e.title: e.title for e in self.entries()}
        for entry in self.entries():
            bare = re.sub(r"\s*\(.*?\)\s*$", "", entry.title)
            out[bare] = entry.title
            if entry.custom and entry.type == "person" and " " in bare:
                out[bare.split()[-1]] = entry.title
        titles = {e.title for e in self.entries()}
        out.update({alias: title for alias, title in ALIASES.items() if title in titles})
        return out
