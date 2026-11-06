"""Combine evaluation results into the file behind the Measurements page.

    python -m eval.publish

Reads eval/results/ and writes web/data/metrics.json. Nothing is typed in by hand:
each number on the page comes from a saved per-question result.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Dict, List

from eval.metrics import load_golden
from src.config import EMBED_MODEL, GEN_MODEL

RESULTS = Path(__file__).resolve().parent / "results"
OUTPUT = Path(__file__).resolve().parent.parent / "web" / "data" / "metrics.json"

COMMANDS = [
    "python -m src.ingest",
    "python -m eval.run_eval --split dev --answers --label final-dev",
    "python -m eval.run_eval --split test --answers --label final-test",
    "python -m eval.publish",
]

ABLATION = [
    ("baseline-dev", "Original version", "İlk sürüm"),
    (
        "ablation-1-chunking-router",
        "New chunking, and a router that handles typos, trap words and follow-ups",
        "Yeni parçalama; yazım hatası, kelime tuzağı ve takip sorularını çözen yönlendirici",
    ),
    ("ablation-2-prefixes", "The task prefixes the embedding model expects", "Embedding modelinin beklediği görev önekleri"),
    ("ablation-3-hybrid", "Keyword search merged with vector search", "Vektör aramasıyla birleştirilen anahtar kelime araması"),
    (
        "ablation-4-cap",
        "At most two passages per page when no page is named",
        "Sayfa adı geçmeyen sorularda sayfa başına en çok iki pasaj",
    ),
    (
        "ablation-5-type",
        "Firm cues such as “place” restrict the page type",
        "“Place” gibi güçlü ipuçlarının sayfa türünü sınırlaması",
    ),
    (
        "ablation-6-translate",
        "Turkish questions searched through a local translation",
        "Türkçe soruların yerel çeviriyle aranması",
    ),
]


def load(label: str) -> Dict:
    data = json.loads((RESULTS / f"{label}.json").read_text(encoding="utf-8"))
    if data.get("complete") is False:
        raise SystemExit(f"{label} is an interrupted run. Finish it with --resume before publishing.")
    return data


def with_answer_rate(summary: Dict) -> Dict:
    """Add one per-category answer score: correct answers, or correct refusals."""
    for block in [summary["overall"], *summary["categories"].values()]:
        rate = block.get("answer_correct", block.get("refusal_correct"))
        if rate is not None:
            block["answer_rate"] = rate
    return summary


def main() -> None:
    baseline = load("baseline")
    splits = {}
    for split in ("test", "dev"):
        splits[split] = {
            "baseline": with_answer_rate(baseline[split]),
            "current": with_answer_rate(load(f"final-{split}")["summary"]),
        }

    ablation: List[Dict] = []
    for label, english, turkish in ABLATION:
        if label == "baseline-dev":
            overall = baseline["dev"]["overall"]
        else:
            overall = load(label)["summary"]["overall"]
        ablation.append(
            {"en": english, "tr": turkish, "retrieval_success": overall["retrieval_success"], "hit_at_1": overall["hit_at_1"]}
        )

    speed = load("speed")
    speed.pop("rows", None)
    review = load("manual-review")

    payload = {
        "speed": speed,
        "manual_review": {key: value for key, value in review.items() if isinstance(value, int)},
        "date": date.today().isoformat(),
        "questions": len(load_golden()),
        "passages": load("final-test")["indexed_passages"],
        "model": GEN_MODEL,
        "embed_model": EMBED_MODEL,
        "commands": COMMANDS,
        "splits": splits,
        "ablation": ablation,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote {OUTPUT}\n")
    print(markdown_table(splits["test"]))


def markdown_table(split: Dict) -> str:
    """The per-category table for the README, printed so it never needs retyping."""
    names = {
        "anonymous": "No name in the question",
        "named": "Name in the question",
        "list": "\"Which places are in…\"",
        "typo": "With a typo",
        "ambiguous": "Ordinary-word traps",
        "compare": "Comparisons",
        "turkish": "In Turkish",
        "followup": "Follow-ups",
        "refusal": "Should decline",
    }

    def percent(value) -> str:
        return "n/a" if value is None else f"{value:.0%}"

    lines = [
        "| Question type | Questions | Search, before | Search, now | Answer, before | Answer, now |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    before, after = split["baseline"]["categories"], split["current"]["categories"]
    for key, label in names.items():
        lines.append(
            f"| {label} | {after[key]['questions']} "
            f"| {percent(before[key].get('retrieval_success'))} | {percent(after[key].get('retrieval_success'))} "
            f"| {percent(before[key].get('answer_rate'))} | {percent(after[key].get('answer_rate'))} |"
        )
    overall_before, overall_after = split["baseline"]["overall"], split["current"]["overall"]
    lines.append(
        f"| **All** | **{overall_after['questions']}** "
        f"| **{percent(overall_before.get('retrieval_success'))}** | **{percent(overall_after.get('retrieval_success'))}** "
        f"| | |"
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
