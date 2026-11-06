import json
import time

import pytest

from eval.metrics import load_golden, numbers_grounded, score_answer, score_retrieval, summarize
from eval.run_eval import run_questions
from src.embedder import OllamaEmbedder
from src.generator import REFUSALS
from src.ingest import index_manifest, index_page
from src.ollama_client import OllamaUnavailable
from src.rag import RagSystem
from src.vector_store import IndexMismatch
from src.wiki import Page, load_page
from tests.conftest import make_page


def page_with(text: str) -> Page:
    return Page("Marie Curie", "https://example.org/curie", 2, time.time(), [("", text)])


def test_unchanged_page_is_not_embedded_again(store, embedder, catalog, fake_ollama):
    stats = store.entity_stats()["Marie Curie"]
    before = len(fake_ollama.embed_calls)
    entry = catalog.get("Marie Curie")
    report = index_page(store, embedder, entry, make_page("Marie Curie"), existing_hash=stats["content_hash"])
    assert report.skipped
    assert len(fake_ollama.embed_calls) == before


def test_changed_page_replaces_its_old_passages(store, embedder, catalog):
    index_page(store, embedder, catalog.get("Marie Curie"), page_with("A brand new sentence about her life."))
    stats = store.entity_stats()["Marie Curie"]
    assert stats["passages"] == 1
    documents = [r["document"] for r in store.all_records() if r["metadata"]["entity"] == "Marie Curie"]
    assert documents == ["A brand new sentence about her life."]


def test_documents_are_embedded_with_prefix_and_title(store, embedder, catalog, fake_ollama):
    index_page(store, embedder, catalog.get("Marie Curie"), page_with("She studied in Paris."))
    assert fake_ollama.embed_calls[-1][0] == "search_document: Marie Curie\nShe studied in Paris."


def test_passage_metadata_is_complete(store):
    record = next(r for r in store.all_records() if r["metadata"]["entity"] == "Marie Curie")
    meta = record["metadata"]
    assert meta["type"] == "person"
    assert meta["source"].startswith("https://en.wikipedia.org/")
    assert meta["n_chunks"] >= 1
    assert "section" in meta


def test_page_cache_avoids_a_second_download(tmp_path):
    calls = []

    def fetcher(title):
        calls.append(title)
        return make_page("Marie Curie")

    first = load_page("Marie Curie", fetcher=fetcher, pages_dir=tmp_path)
    second = load_page("Marie Curie", fetcher=fetcher, pages_dir=tmp_path)
    assert calls == ["Marie Curie"]
    assert second.sections == first.sections
    load_page("Marie Curie", refresh=True, fetcher=fetcher, pages_dir=tmp_path)
    assert len(calls) == 2


def test_index_built_with_other_settings_is_refused(store, fake_ollama, catalog):
    other = OllamaEmbedder(client=fake_ollama, use_prefixes=False)
    with pytest.raises(IndexMismatch):
        store.check_manifest(index_manifest(other))
    with pytest.raises(IndexMismatch):
        RagSystem(store=store, client=fake_ollama, embedder=other, catalog=catalog)


def test_events_arrive_in_order_and_sources_precede_the_answer(rag):
    events = list(rag.ask_stream("Where was Marie Curie born?"))
    names = [e["event"] for e in events]
    assert names[:2] == ["route", "sources"]
    assert names[-1] == "done"
    assert set(names[2:-1]) == {"token"}


def test_done_event_has_attribution_and_timings(rag):
    done = list(rag.ask_stream("Where was Marie Curie born?"))[-1]
    assert done["refused"] is False
    assert done["attribution"]["used_sources"] == [1]
    assert done["timings"]["tokens_per_second"] == 10.0
    assert done["entities"] == ["Marie Curie"]


def test_refusal_is_normalized(rag, fake_ollama):
    fake_ollama.reply = "I don't know, sorry."
    result = rag.ask("Who is the president of Mars?")
    assert result.refused
    assert result.answer == REFUSALS["en"]
    assert result.attribution["used_sources"] == []


def test_answer_with_no_support_in_any_passage_is_withheld(rag, fake_ollama):
    fake_ollama.reply = "Bananas grow in tropical climates [1]."
    done = list(rag.ask_stream("Who was Marie Curie?"))[-1]
    assert done["refused"] is True
    assert done["withheld"] is True
    assert done["answer"] == REFUSALS["en"]
    assert done["attribution"]["used_sources"] == []


def test_supported_answer_is_not_withheld(rag):
    done = list(rag.ask_stream("Where was Marie Curie born?"))[-1]
    assert done["withheld"] is False


def test_empty_answer_counts_as_a_refusal(rag, fake_ollama):
    fake_ollama.reply = ""
    assert rag.ask("Who was Marie Curie?").refused


def test_turkish_question_is_translated_for_search_and_answered_in_turkish(rag, fake_ollama):
    fake_ollama.reply = "Ayasofya İstanbul'dadır [1]."
    result = rag.ask("Ayasofya nerede?")
    assert result.language == "tr"
    assert result.route["translated"] == "Where is Hagia Sophia?"
    assert result.route["entities"] == ["Hagia Sophia"]
    assert "Write in Turkish" in fake_ollama.chats[-1][0]["content"]


def test_failed_translation_falls_back_to_the_original_question(rag, fake_ollama):
    fake_ollama.translation = ""
    result = rag.ask("Ayasofya nerede?")
    assert result.route["translated"] is None
    assert result.route["entities"] == ["Hagia Sophia"]


def test_english_questions_skip_translation(rag):
    assert rag.ask("Where was Marie Curie born?").route["translated"] is None


def test_followup_uses_history(rag, fake_ollama):
    history = [{"question": "Who was Marie Curie?", "answer": "A physicist [1].", "entities": ["Marie Curie"]}]
    result = rag.ask("Where was she born?", history=history)
    assert result.route["reason"] == "followup"
    assert {s["entity"] for s in result.sources} == {"Marie Curie"}
    assert fake_ollama.chats[-1][1]["content"] == "Who was Marie Curie?"
    assert fake_ollama.chats[-1][2]["content"] == "A physicist."


def test_top_k_above_the_maximum_is_capped(rag):
    assert len(rag.ask("things in Turkey", top_k=500).sources) <= 10


# ------------------------------------------------------------------ evaluation runs
def test_interrupted_evaluation_keeps_its_results_and_can_resume(rag, tmp_path, monkeypatch):
    common = {"split": "dev", "category": "named"}
    items = [
        {**common, "id": "a", "question": "Who was Marie Curie?", "entities_all": ["Marie Curie"]},
        {**common, "id": "b", "question": "Who was Albert Einstein?", "entities_all": ["Albert Einstein"]},
    ]
    path = tmp_path / "run.json"
    real_prepare = rag.prepare
    calls = []

    def fail_on_second(question, *args, **kwargs):
        calls.append(question)
        if len(calls) == 2:
            raise OllamaUnavailable("gone")
        return real_prepare(question, *args, **kwargs)

    monkeypatch.setattr(rag, "prepare", fail_on_second)
    with pytest.raises(OllamaUnavailable):
        run_questions(rag, items, False, path, {"label": "run"})
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["complete"] is False
    assert [row["id"] for row in saved["rows"]] == ["a"]

    monkeypatch.setattr(rag, "prepare", real_prepare)
    finished = run_questions(rag, items, False, path, {"label": "run"}, resume=True)
    assert finished["complete"] is True
    assert [row["id"] for row in finished["rows"]] == ["a", "b"]


# ------------------------------------------------------------------ metrics
def test_golden_set_is_valid():
    items = load_golden()
    assert len({i["id"] for i in items}) == len(items)
    assert {i["split"] for i in items} == {"dev", "test"}
    for item in items:
        assert item.get("expect_refusal") or item.get("entities_any") or item.get("entities_all")


def passages(*entities, text="x"):
    return [{"entity": e, "document": text} for e in entities]


def test_retrieval_scoring():
    item = {"entities_all": ["A", "B"], "entities_forbid": ["C"]}
    assert score_retrieval(item, passages("A", "B"))["success"]
    assert not score_retrieval(item, passages("A", "D"))["success"]
    assert not score_retrieval(item, passages("A", "B", "C"))["success"]
    assert score_retrieval(item, passages("D", "A", "B"))["reciprocal_rank"] == 0.5
    assert score_retrieval({"expect_refusal": True}, passages("A")) == {}


def test_retrieval_scoring_checks_the_answer_is_in_the_passage():
    item = {"entities_all": ["A"], "passage_regex": "1867"}
    assert score_retrieval(item, passages("A", text="born in 1867"))["success"]
    assert not score_retrieval(item, passages("A", text="born long ago"))["success"]


def test_answer_scoring():
    assert score_answer({"expect_refusal": True}, "...", True)["correct"]
    assert not score_answer({"expect_refusal": True}, "Something", False)["correct"]
    assert score_answer({"answer_regex": "Warsaw"}, "In Warsaw.", False)["correct"]
    assert not score_answer({"answer_regex": "Warsaw"}, "In Paris.", False)["correct"]
    assert not score_answer({"answer_regex": "Warsaw"}, "I don't know, Warsaw?", True)["correct"]


def test_numbers_grounded():
    source = passages("A", text="Height is 1,083 ft, built in 1889.")
    assert numbers_grounded("It was built in 1889 and is 1083 ft tall.", source) is True
    assert numbers_grounded("It was built in 1901.", source) is False
    assert numbers_grounded("It is very tall.", source) is None


def test_summary_counts_categories():
    rows = [
        {"category": "named", "retrieval": score_retrieval({"entities_all": ["A"]}, passages("A"))},
        {"category": "named", "retrieval": score_retrieval({"entities_all": ["A"]}, passages("B"))},
    ]
    summary = summarize(rows)
    assert summary["overall"]["retrieval_success"] == 0.5
    assert summary["categories"]["named"]["questions"] == 2
