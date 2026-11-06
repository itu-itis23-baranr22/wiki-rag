"""Record real answers for the static demo that reviewers can open without installing anything.

    python -m scripts.record_demo

Every event comes from a real run of the current system, including its timings.
The page replays them and says that it is a recording.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

from src.rag import RagSystem

OUTPUT = Path(__file__).resolve().parent.parent / "web" / "data" / "demo.json"
EXAMPLES_FILE = Path(__file__).resolve().parent.parent / "web" / "i18n.js"


def example_questions() -> List[str]:
    """Read the example questions from the interface, so the two cannot drift apart."""
    source = EXAMPLES_FILE.read_text(encoding="utf-8")
    block = source[source.index("export const EXAMPLES") :]
    return re.findall(r'question: "([^"]+)"', block)


def record(rag: RagSystem, question: str, history: List[Dict]) -> Dict:
    print(f"Recording: {question}", flush=True)
    events = list(rag.ask_stream(question, history=history))
    return {"question": question, "events": events}


def main() -> None:
    rag = RagSystem()
    answers = []
    for question in example_questions():
        entry = record(rag, question, [])
        answers.append(entry)
        if question == "Who was Marie Curie?":
            done = entry["events"][-1]
            history = [{"question": question, "answer": done["answer"], "entities": done["entities"]}]
            answers.append(record(rag, "Where was she born?", history))

    stats = rag.store.entity_stats()
    pages = [
        {
            "title": entry.title,
            "type": entry.type,
            "custom": False,
            "passages": stats.get(entry.title, {}).get("passages", 0),
            "url": stats.get(entry.title, {}).get("url", ""),
            "revision": stats.get(entry.title, {}).get("revision"),
        }
        for entry in rag.catalog.entries()
        if not entry.custom
    ]
    payload = {
        "status": {
            "ready": False,
            "ollama": {"reachable": False, "models": []},
            "index": {"passages": rag.store.count(), "pages": len(pages)},
            "defaults": {"model": rag.generator.model, "embed_model": rag.embedder.model, "top_k": 5, "max_top_k": 10},
        },
        "library": {"pages": pages, "total_passages": sum(p["passages"] for p in pages)},
        "answers": answers,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
