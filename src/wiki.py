"""Fetching Wikipedia pages and keeping them in a local cache."""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, List, Tuple

import wikipediaapi

from src.config import BOILERPLATE_SECTIONS, PAGES_DIR
from src.textnorm import fold

USER_AGENT = "wiki-rag/2.0 (https://github.com/itu-itis23-baranr22/wiki-rag)"
LEAD_SECTION = ""  # the section title stored for a page's introduction


class PageNotFound(LookupError):
    """Wikipedia has no page with this title."""


class AmbiguousPage(LookupError):
    """The title is a disambiguation page, so it does not describe one subject."""


@dataclass
class Page:
    title: str
    url: str
    revision: int
    fetched_at: float
    sections: List[Tuple[str, str]]

    @property
    def text_length(self) -> int:
        return sum(len(text) for _, text in self.sections)


def _flatten(section, parent: str = "") -> List[Tuple[str, str]]:
    path = f"{parent} > {section.title}" if parent else section.title
    out: List[Tuple[str, str]] = []
    if section.text.strip():
        out.append((path, section.text.strip()))
    for child in section.sections:
        out.extend(_flatten(child, path))
    return out


def fetch_page(title: str) -> Page:
    """Download a page. Raises PageNotFound or AmbiguousPage."""
    wiki = wikipediaapi.Wikipedia(
        user_agent=USER_AGENT,
        language="en",
        extract_format=wikipediaapi.ExtractFormat.WIKI,
    )
    page = wiki.page(title)
    if not page.exists():
        raise PageNotFound(title)
    if any("disambiguation pages" in category.lower() for category in page.categories):
        raise AmbiguousPage(title)

    sections: List[Tuple[str, str]] = []
    if page.summary.strip():
        sections.append((LEAD_SECTION, page.summary.strip()))
    for section in page.sections:
        # Only top-level headings count as boilerplate. A "Sources" subsection inside
        # "Life and career" is real content about the subject.
        if section.title.strip().casefold() not in BOILERPLATE_SECTIONS:
            sections.extend(_flatten(section))
    return Page(title=title, url=page.fullurl, revision=page.lastrevid, fetched_at=time.time(), sections=sections)


def page_cache_path(title: str, pages_dir: Path = PAGES_DIR) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "-", fold(title)).strip("-")
    digest = hashlib.sha1(title.encode("utf-8")).hexdigest()[:8]
    return Path(pages_dir) / f"{slug}-{digest}.json"


def load_page(
    title: str,
    refresh: bool = False,
    fetcher: Callable[[str], Page] = fetch_page,
    pages_dir: Path = PAGES_DIR,
) -> Page:
    """Return a page from the cache, downloading it when missing or when asked to refresh."""
    path = page_cache_path(title, pages_dir)
    if path.exists() and not refresh:
        data = json.loads(path.read_text(encoding="utf-8"))
        data["sections"] = [tuple(pair) for pair in data["sections"]]
        return Page(**data)
    page = fetcher(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(page), ensure_ascii=False, indent=1), encoding="utf-8")
    return page
