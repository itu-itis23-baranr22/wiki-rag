"""Simple CLI chat interface for the Wiki-RAG system.

Run:
    python cli.py

Slash commands:
    /help     - list commands
    /sources  - toggle showing source chunks
    /reset    - clear the in-process chat history (does NOT touch the index)
    /quit     - exit
"""
from __future__ import annotations

import sys

from src.rag import RagSystem


HELP = """
Commands:
  /help      show this help
  /sources   toggle showing source chunks under each answer
  /reset     clear chat history (the index on disk stays)
  /quit      exit
"""


def _print_sources(chunks):
    if not chunks:
        print("  (no source chunks)")
        return
    for i, c in enumerate(chunks, start=1):
        meta = c.get("metadata") or {}
        entity = meta.get("entity", "?")
        etype = meta.get("type", "?")
        dist = c.get("distance", float("nan"))
        snippet = c["document"][:160].replace("\n", " ")
        print(f"  [{i}] ({etype}: {entity}, dist={dist:.3f})")
        print(f"      {snippet}{'...' if len(c['document']) > 160 else ''}")


def main() -> int:
    print("Wiki-RAG — local chat over Wikipedia people & places.")
    print("Type your question, or /help for commands.\n")

    rag = RagSystem()
    show_sources = False
    history = []   # local, just for the user's convenience; not sent to the LLM

    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not q:
            continue

        if q.startswith("/"):
            cmd = q.lower()
            if cmd in ("/quit", "/exit"):
                return 0
            if cmd == "/help":
                print(HELP)
                continue
            if cmd == "/sources":
                show_sources = not show_sources
                print(f"  sources display: {'ON' if show_sources else 'OFF'}")
                continue
            if cmd == "/reset":
                history.clear()
                print("  chat history cleared.")
                continue
            print(f"  unknown command: {q}")
            continue

        try:
            print("  (thinking...)")
            # Stream tokens for snappier feedback
            answer_parts = []
            chunks = []
            route = ""
            rationale = ""
            for ev in rag.ask_stream(q):
                if ev["event"] == "retrieved":
                    chunks = ev["chunks"]
                    route = ev["route"]
                    rationale = ev["rationale"]
                    print(f"  [route={route} — {rationale}, {len(chunks)} chunks]")
                    print()
                elif ev["event"] == "token":
                    sys.stdout.write(ev["token"])
                    sys.stdout.flush()
                    answer_parts.append(ev["token"])
                elif ev["event"] == "done":
                    print("\n")

            if show_sources:
                print("  --- sources ---")
                _print_sources(chunks)
                print()

            history.append({"q": q, "a": "".join(answer_parts), "chunks": chunks})

        except Exception as e:
            print(f"  ! error: {e}")
            print("    (is `ollama serve` running and are the models pulled?)")


if __name__ == "__main__":
    raise SystemExit(main())
