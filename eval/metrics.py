"""Scoring for the golden question set.

This module deliberately imports nothing from the application, so the same
scoring code can measure any version of the pipeline, including old commits.
A system under test only has to return, for each question, its ranked passages
as ``{"entity": str, "document": str}`` dictionaries and optionally an answer.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

GOLDEN_PATH = Path(__file__).resolve().parent / "golden.jsonl"
CATEGORIES = ("anonymous", "named", "list", "typo", "ambiguous", "compare", "turkish", "followup", "refusal")


def load_golden(split: Optional[str] = None, path: Path = GOLDEN_PATH) -> List[Dict]:
    """Return golden items. `split` is "dev", "test" or None for every item."""
    items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [item for item in items if split in (None, "all", item["split"])]


def score_retrieval(item: Dict, passages: List[Dict], k: int = 5) -> Dict:
    """Score one question's ranked passages, or return {} for refusal-only items."""
    wants_any = item.get("entities_any") or []
    wants_all = item.get("entities_all") or []
    forbid = item.get("entities_forbid") or []
    regex = item.get("passage_regex")
    if not (wants_any or wants_all or regex):
        return {}

    top = passages[:k]
    entities = [p["entity"] for p in top]
    wanted = set(wants_any) | set(wants_all)

    first_rank = next((i for i, e in enumerate(passages) if e["entity"] in wanted), None)
    if wants_all:
        recall = sum(e in entities for e in wants_all) / len(wants_all)
    elif wants_any:
        recall = 1.0 if set(wants_any) & set(entities) else 0.0
    else:
        recall = 1.0

    passage_hit = True
    if regex:
        pattern = re.compile(regex, flags=re.IGNORECASE)
        passage_hit = any(pattern.search(p["document"]) for p in top)

    violated = bool(set(forbid) & set(entities))
    success = recall == 1.0 and passage_hit and not violated
    return {
        "success": success,
        "recall": recall,
        "hit_at_1": first_rank == 0,
        "reciprocal_rank": 0.0 if first_rank is None else 1.0 / (first_rank + 1),
        "passage_hit": passage_hit if regex else None,
        "forbidden_retrieved": violated,
    }


def score_answer(item: Dict, answer: str, refused: bool, valid_citations: Optional[int] = None) -> Dict:
    """Score one generated answer. `refused` says whether the system declined."""
    expect_refusal = bool(item.get("expect_refusal"))
    out: Dict = {"refused": refused, "expect_refusal": expect_refusal}
    if expect_refusal:
        out["correct"] = refused
    else:
        regex = item.get("answer_regex")
        matched = bool(re.search(regex, answer, flags=re.IGNORECASE)) if regex else None
        out["answer_match"] = matched
        out["correct"] = (not refused) and (matched is not False)
    if valid_citations is not None:
        out["cited"] = valid_citations > 0
    return out


def numbers_grounded(answer: str, passages: Iterable[Dict]) -> Optional[bool]:
    """True if every multi-digit number in the answer appears in the passages.

    A cheap, objective hallucination signal: years and quantities are the facts a
    small model most often invents. Returns None when the answer has no numbers.
    """
    numbers = {re.sub(r"[.,]", "", n) for n in re.findall(r"\d[\d.,]*\d", answer)}
    if not numbers:
        return None
    haystack = re.sub(r"[.,]", "", " ".join(p["document"] for p in passages))
    return all(n in haystack for n in numbers)


def mean(values: Iterable[float]) -> Optional[float]:
    values = list(values)
    return sum(values) / len(values) if values else None


def summarize(rows: List[Dict]) -> Dict:
    """Aggregate per-question rows into overall and per-category scores."""
    def block(subset: List[Dict]) -> Dict:
        retrieval = [r["retrieval"] for r in subset if r.get("retrieval")]
        answers = [r["answer_score"] for r in subset if r.get("answer_score")]
        answerable = [a for a in answers if not a["expect_refusal"]]
        refusals = [a for a in answers if a["expect_refusal"]]
        grounded = [r["numbers_grounded"] for r in subset if r.get("numbers_grounded") is not None]
        cited = [a["cited"] for a in answerable if not a["refused"] and "cited" in a]
        report: Dict = {"questions": len(subset)}
        if retrieval:
            report["retrieval_success"] = mean(float(r["success"]) for r in retrieval)
            report["entity_recall"] = mean(r["recall"] for r in retrieval)
            report["hit_at_1"] = mean(float(r["hit_at_1"]) for r in retrieval)
            report["mrr"] = mean(r["reciprocal_rank"] for r in retrieval)
        if answerable:
            report["answer_correct"] = mean(float(a["correct"]) for a in answerable)
            report["false_refusal"] = mean(float(a["refused"]) for a in answerable)
        if refusals:
            report["refusal_correct"] = mean(float(a["correct"]) for a in refusals)
        if grounded:
            report["numbers_grounded"] = mean(float(g) for g in grounded)
        if cited:
            report["cited_answers"] = mean(float(c) for c in cited)
        latencies = [r["latency_s"] for r in subset if r.get("latency_s") is not None]
        if latencies:
            report["median_latency_s"] = sorted(latencies)[len(latencies) // 2]
        return report

    categories = {c: block([r for r in rows if r["category"] == c]) for c in CATEGORIES}
    return {"overall": block(rows), "categories": {c: v for c, v in categories.items() if v["questions"]}}


def format_table(summary: Dict) -> str:
    """Render a summary as a plain-text table for the terminal."""
    columns = ["retrieval_success", "hit_at_1", "mrr", "answer_correct", "refusal_correct", "numbers_grounded", "cited_answers"]
    header = f"{'category':<11}{'n':>4} " + " ".join(f"{c[:11]:>11}" for c in columns)
    lines = [header, "-" * len(header)]

    def row(name: str, data: Dict) -> str:
        cells = []
        for c in columns:
            v = data.get(c)
            cells.append(f"{v:>11.3f}" if isinstance(v, float) else f"{'-':>11}")
        return f"{name:<11}{data['questions']:>4} " + " ".join(cells)

    for name, data in summary["categories"].items():
        lines.append(row(name, data))
    lines.append("-" * len(header))
    lines.append(row("overall", summary["overall"]))
    return "\n".join(lines)
