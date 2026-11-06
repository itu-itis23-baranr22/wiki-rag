"""Run the golden question set against the current system.

    python -m eval.run_eval --split dev                  # retrieval only, about a minute
    python -m eval.run_eval --split dev --answers        # also generate answers, slow on CPU
    python -m eval.run_eval --split test --answers --label final

`dev` is for tuning. `test` is meant to be run rarely, to report a final score that
did not influence the tuning. Results are written to eval/results/<label>.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from eval.metrics import format_table, load_golden, numbers_grounded, score_answer, score_retrieval, summarize
from src.ollama_client import OllamaError
from src.rag import RagSystem
from src.retriever import RetrievalSettings

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def build_history(rag: RagSystem, item: Dict) -> List[Dict]:
    """Turn the item's earlier questions into history without generating their answers."""
    history: List[Dict] = []
    for earlier in item.get("history", []):
        prepared = rag.prepare(earlier)
        history.append({"question": earlier, "answer": "", "entities": prepared["retrieved"].route.entity_names})
    return history


def evaluate_one(rag: RagSystem, item: Dict, answers: bool, top_k: Optional[int] = None) -> Dict:
    started = time.time()
    history = build_history(rag, item)
    row: Dict = {"id": item["id"], "split": item["split"], "category": item["category"], "question": item["question"]}
    if answers:
        result = rag.ask(item["question"], top_k=top_k, history=history)
        passages = [{"entity": s["entity"], "document": s["text"]} for s in result.sources]
        row["answer"] = result.answer
        row["language"] = result.language
        row["translated"] = result.route.get("translated")
        row["answer_score"] = score_answer(item, result.answer, result.refused, len(result.attribution["used_sources"]))
        row["invalid_citations"] = result.attribution["invalid_citations"]
        row["uncited_sentences"] = result.attribution["uncited_sentences"]
        row["inferred_sentences"] = sum(s["inferred"] for p in result.attribution["paragraphs"] for s in p)
        row["numbers_grounded"] = None if result.refused else numbers_grounded(result.answer, passages)
        row["timings"] = result.timings
        row["latency_s"] = round(time.time() - started, 1)
    else:
        prepared = rag.prepare(item["question"], top_k=top_k, history=history)
        passages = [{"entity": c["metadata"]["entity"], "document": c["document"]} for c in prepared["retrieved"].chunks]
        row["translated"] = prepared["translated"]
    row["retrieved"] = [p["entity"] for p in passages]
    row["retrieval"] = score_retrieval(item, passages)
    return row


def run_questions(
    rag: RagSystem,
    items: List[Dict],
    answers: bool,
    path: Path,
    metadata: Dict,
    top_k: Optional[int] = None,
    resume: bool = False,
) -> Dict:
    """Score the questions, saving after each one, because a CPU run can take an hour.

    If Ollama goes away, what is done so far stays on disk marked incomplete, and
    running again with `resume=True` continues from there.
    """
    rows: List[Dict] = []
    if resume and path.exists():
        rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
    finished = {row["id"] for row in rows}
    remaining = [item for item in items if item["id"] not in finished]

    def save(complete: bool) -> Dict:
        payload = {
            **metadata,
            "complete": complete,
            "indexed_passages": rag.store.count(),
            "summary": summarize(rows),
            "rows": rows,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        return payload

    for position, item in enumerate(remaining, start=1):
        rows.append(evaluate_one(rag, item, answers, top_k))
        save(complete=False)
        print(f"[{position}/{len(remaining)}] {item['id']}", flush=True)
    return save(complete=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the system on the golden questions.")
    parser.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    parser.add_argument("--answers", action="store_true", help="also generate answers and score them")
    parser.add_argument("--label", default=None, help="name of the results file")
    parser.add_argument("--no-lexical", action="store_true", help="vector search only")
    parser.add_argument("--no-vector", action="store_true", help="keyword search only")
    parser.add_argument("--no-translate", action="store_true", help="search Turkish questions untranslated")
    parser.add_argument("--entity-cap", type=int, default=None, help="passages allowed per page, 99 for no cap")
    parser.add_argument("--no-type-filter", action="store_true", help="do not let keyword cues restrict the page type")
    parser.add_argument("--lexical-weight", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="continue an interrupted run with the same label")
    args = parser.parse_args()

    settings = RetrievalSettings(
        use_lexical=not args.no_lexical,
        use_vector=not args.no_vector,
        type_filter=not args.no_type_filter,
    )
    if args.entity_cap is not None:
        settings = settings.with_changes(entity_cap=args.entity_cap)
    if args.lexical_weight is not None:
        settings = settings.with_changes(lexical_weight=args.lexical_weight)

    label = args.label or f"{args.split}{'-answers' if args.answers else ''}"
    path = RESULTS_DIR / f"{label}.json"
    metadata = {"label": label, "split": args.split, "settings": vars(args)}
    rag = RagSystem(settings=settings, translate=not args.no_translate)
    try:
        payload = run_questions(rag, load_golden(args.split), args.answers, path, metadata, args.top_k, args.resume)
    except OllamaError as exc:
        raise SystemExit(f"\nStopped: {exc}\nFinished questions are saved in {path}. Add --resume to continue.") from exc
    print()
    print(format_table(payload["summary"]))
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
