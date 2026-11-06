"""Command-line chat for Wiki-RAG:  python cli.py

Commands:
    /help     list commands
    /sources  toggle showing the passages behind each answer
    /reset    forget the conversation, so the next question starts fresh
    /quit     exit
"""
from __future__ import annotations

import sys
from typing import Dict, List

from src.ollama_client import OllamaError
from src.rag import RagSystem
from src.vector_store import IndexMismatch

HELP = """
Commands:
  /help      show this help
  /sources   toggle showing the passages behind each answer
  /reset     forget the conversation, so the next question starts fresh
  /quit      exit
"""


def print_sources(sources: List[Dict], used: List[int]) -> None:
    for source in sources:
        marker = "*" if source["number"] in used else " "
        section = f", {source['section']}" if source["section"] else ""
        snippet = source["text"][:160].replace("\n", " ")
        print(f"  {marker}[{source['number']}] {source['entity']}{section}")
        print(f"      {snippet}{'...' if len(source['text']) > 160 else ''}")
    print("  (* cited in the answer)")


def main() -> int:
    try:
        rag = RagSystem()
    except IndexMismatch as exc:
        print(exc)
        return 1
    if not rag.store.count():
        print("The index is empty. Build it first with: python -m src.ingest")
        return 1

    print(f"Wiki-RAG. {rag.store.count()} passages indexed. Ask in English or Turkish, or type /help.\n")
    show_sources = False
    history: List[Dict] = []

    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not question:
            continue

        if question.startswith("/"):
            command = question.lower()
            if command in ("/quit", "/exit"):
                return 0
            if command == "/help":
                print(HELP)
            elif command == "/sources":
                show_sources = not show_sources
                print(f"  sources: {'on' if show_sources else 'off'}")
            elif command == "/reset":
                history.clear()
                print("  conversation cleared.")
            else:
                print(f"  unknown command: {question}")
            continue

        try:
            sources: List[Dict] = []
            done: Dict = {}
            for event in rag.ask_stream(question, history=history):
                if event["event"] == "route":
                    details = ", ".join(event["entities"]) or event["reason"]
                    translated = f" | searched as: {event['translated']}" if event["translated"] else ""
                    print(f"  [{details}{translated}]\n")
                elif event["event"] == "sources":
                    sources = event["sources"]
                elif event["event"] == "token":
                    sys.stdout.write(event["token"])
                    sys.stdout.flush()
                elif event["event"] == "done":
                    done = event
                    print("\n")
            if show_sources:
                print_sources(sources, done["attribution"]["used_sources"])
                print()
            history.append({"question": question, "answer": done["answer"], "entities": done["entities"]})
        except KeyboardInterrupt:
            print("\n  stopped.\n")
        except OllamaError as exc:
            print(f"  ! {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
