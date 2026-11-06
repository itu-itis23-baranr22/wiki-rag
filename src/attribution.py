"""Connecting each answer sentence to the source sentence that supports it.

The model cites passages by number. This module turns that into something a reader
can check: for every answer sentence it finds the sentence inside the cited passage
that best matches, and reports where it is. Citations to passages that do not exist
are dropped and counted, and sentences with no usable citation are reported as such.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Set, Tuple

from src.chunker import split_sentences
from src.textnorm import STOP_WORDS, fold

_CITATION = re.compile(r"\[(?:passage |source )?(\d+(?:\s*,\s*\d+)*)\]", flags=re.IGNORECASE)
_CITATION_RUN = r"(?:\s*\[(?:passage |source )?\d+(?:\s*,\s*\d+)*\])+"
_CITE_AFTER_STOP = re.compile(rf"([.!?])({_CITATION_RUN})", flags=re.IGNORECASE)
_WORD = re.compile(r"\w+", flags=re.UNICODE)

MIN_SUPPORT = 0.30  # share of an answer sentence's weighted terms found in the evidence
# A Turkish answer to English passages shares only names and numbers with them, so a much
# smaller overlap still identifies the right sentence.
MIN_SUPPORT_CROSS_LANGUAGE = 0.15


def _weighted_terms(sentence: str) -> Dict[str, float]:
    """Content words of a sentence. Numbers and names count for more than common words."""
    weights: Dict[str, float] = {}
    for match in _WORD.finditer(sentence):
        word = match.group()
        folded = fold(word)
        if folded in STOP_WORDS or len(folded) < 2:
            continue
        if word.isdigit():
            weight = 5.0  # a year or quantity identifies the right sentence better than anything else
        elif word[0].isupper():
            weight = 2.0
        else:
            weight = 1.0
        weights[folded] = max(weights.get(folded, 0.0), weight)
    return weights


def support_score(answer_sentence: str, source_sentence: str) -> float:
    """Weighted share of the answer sentence's terms that the source sentence contains."""
    terms = _weighted_terms(answer_sentence)
    if not terms:
        return 0.0
    source = {fold(w) for w in _WORD.findall(source_sentence)}
    return sum(w for term, w in terms.items() if term in source) / sum(terms.values())


def sentence_spans(passage: str) -> List[Tuple[int, int]]:
    """Character spans of each sentence in a passage."""
    spans: List[Tuple[int, int]] = []
    cursor = 0
    for sentence in split_sentences(passage):
        start = passage.find(sentence, cursor)
        if start == -1:
            continue
        end = start + len(sentence)
        spans.append((start, end))
        cursor = end
    return spans


def _focus(answer_sentence: str, source_sentence: str) -> float:
    """Share of the source sentence's terms that the answer uses. Long sentences score low."""
    source_terms = _weighted_terms(source_sentence)
    if not source_terms:
        return 0.0
    answer = {fold(w) for w in _WORD.findall(answer_sentence)}
    return sum(term in answer for term in source_terms) / len(source_terms)


def best_evidence(answer_sentence: str, passage: str, minimum: float = MIN_SUPPORT) -> Optional[Dict]:
    """The passage sentence that best supports the answer sentence, if any is good enough.

    A long sentence can match many words by chance, so among sentences covering the
    answer about equally well, the more focused one wins.
    """
    best: Optional[Dict] = None
    best_rank = -1.0
    numbers = set(re.findall(r"\d+", answer_sentence))
    for start, end in sentence_spans(passage):
        sentence = passage[start:end]
        score = support_score(answer_sentence, sentence)
        rank = score + 0.5 * _focus(answer_sentence, sentence)
        # A sentence stating the answer's year or quantity is the fact itself, not just a mention.
        if numbers & set(re.findall(r"\d+", sentence)):
            rank += 0.25
        if rank > best_rank:
            best_rank = rank
            best = {"start": start, "end": end, "score": round(score, 3)}
    return best if best and best["score"] >= minimum else None


def _parse_numbers(group: str) -> List[int]:
    return [int(n) for n in re.findall(r"\d+", group)]


def attribute(answer: str, passages: Sequence[str], language: str = "en") -> Dict:
    """Split an answer into sentences, each with its sources and evidence spans.

    Returns:
        {
          "paragraphs": [[{"text", "sources": [1, 2], "inferred": bool,
                           "evidence": [{"source": 1, "start", "end", "score"}]}]],
          "used_sources": [1, 2],
          "invalid_citations": 0,
          "uncited_sentences": 0,
        }
    Source numbers are 1-based, matching the numbers shown to the model.
    When the model cites nothing for a sentence but a passage clearly supports it, that
    passage is attached with "inferred": true, so the interface can tell the two apart.
    """
    minimum = MIN_SUPPORT if language == "en" else MIN_SUPPORT_CROSS_LANGUAGE
    invalid = 0
    uncited = 0
    used: Set[int] = set()
    paragraphs: List[List[Dict]] = []

    for raw_paragraph in re.split(r"\n\s*\n|\n(?=[-*•\d])", answer.strip()):
        if not raw_paragraph.strip():
            continue
        # "Paris. [1]" and "Paris [1]." should read the same, so move citations inside.
        normalized = _CITE_AFTER_STOP.sub(lambda m: f"{m.group(2)}{m.group(1)}", raw_paragraph.strip())
        sentences: List[Dict] = []
        for raw_sentence in split_sentences(normalized):
            cited: List[int] = []
            for group in _CITATION.findall(raw_sentence):
                for number in _parse_numbers(group):
                    if 1 <= number <= len(passages):
                        if number not in cited:
                            cited.append(number)
                    else:
                        invalid += 1
            text = re.sub(rf"{_CITATION_RUN}", "", raw_sentence, flags=re.IGNORECASE).strip()
            text = re.sub(r"\s+([.,;:!?])", r"\1", text)
            if not text:
                continue

            inferred = False
            if not cited:
                scored = [
                    (ev["score"], number)
                    for number, passage in enumerate(passages, start=1)
                    if (ev := best_evidence(text, passage, minimum))
                ]
                if scored:
                    cited = [max(scored)[1]]
                    inferred = True
                else:
                    uncited += 1

            evidence = []
            for number in cited:
                found = best_evidence(text, passages[number - 1], minimum)
                if found:
                    evidence.append({"source": number, **found})
            used.update(cited)
            sentences.append({"text": text, "sources": cited, "inferred": inferred, "evidence": evidence})
        if sentences:
            paragraphs.append(sentences)

    return {
        "paragraphs": paragraphs,
        "used_sources": sorted(used),
        "invalid_citations": invalid,
        "uncited_sentences": uncited,
    }
