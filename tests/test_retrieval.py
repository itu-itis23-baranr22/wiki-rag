import pytest

from src.lexical import LexicalIndex
from src.retriever import RetrievalSettings, Retriever, cap_per_entity, fuse


def record(identifier, text, entity="Page", kind="person", section=""):
    return {"id": identifier, "document": text, "metadata": {"entity": entity, "type": kind, "section": section}}


def test_bm25_prefers_rarer_terms_and_handles_plurals():
    index = LexicalIndex(
        [
            record("a", "The pyramid stands in Egypt near the river."),
            record("b", "The river flows through the river valley and the river delta."),
            record("c", "Nothing relevant is mentioned here."),
        ]
    )
    assert index.search("pyramids in Egypt")[0]["id"] == "a"
    assert index.search("xyzzy") == []


def test_bm25_filters_by_type_and_entity():
    index = LexicalIndex([record("a", "Turkey is here", "One", "person"), record("b", "Turkey is there", "Two", "place")])
    assert [h["id"] for h in index.search("Turkey", type_filter="place")] == ["b"]
    assert [h["id"] for h in index.search("Turkey", entity_filter=["One"])] == ["a"]


def test_bm25_counts_page_title_terms():
    index = LexicalIndex([record("a", "It is very tall.", "Eiffel Tower"), record("b", "It is very tall.", "Colosseum")])
    assert index.search("Eiffel")[0]["id"] == "a"


def test_fusion_rewards_agreement_and_keeps_both_ranks():
    settings = RetrievalSettings()
    vector = [record("a", "x"), record("b", "x")]
    lexical = [record("b", "x"), record("c", "x")]
    fused = fuse(vector, lexical, settings)
    assert [hit["id"] for hit in fused] == ["b", "a", "c"]
    assert (fused[0]["vector_rank"], fused[0]["lexical_rank"]) == (2, 1)
    assert fused[2]["vector_rank"] is None


def test_cap_limits_each_page_then_fills_when_pages_run_out():
    ranked = [record(str(i), "x", "One") for i in range(4)] + [record("9", "x", "Two")]
    assert [h["id"] for h in cap_per_entity(ranked, 3, 2)] == ["0", "1", "9"]
    assert len(cap_per_entity(ranked, 5, 2)) == 5


@pytest.fixture
def retriever(rag):
    return rag.retriever


def entities(result):
    return [c["metadata"]["entity"] for c in result.chunks]


def test_a_named_page_scopes_search_to_it(retriever):
    assert set(entities(retriever.retrieve("Who was Marie Curie?"))) == {"Marie Curie"}


def test_comparison_gets_passages_from_both_pages(retriever):
    assert set(entities(retriever.retrieve("Compare Marie Curie and Albert Einstein."))) == {"Marie Curie", "Albert Einstein"}


def test_trap_words_do_not_pull_in_the_wrong_page(retriever):
    assert "Steve Jobs" not in entities(retriever.retrieve("What jobs did Marie Curie have?"))
    assert "Taylor Swift" not in entities(retriever.retrieve("How swift was Albert Einstein?"))


@pytest.mark.parametrize("top_k", [1, 2, 3, 5])
def test_top_k_is_respected_without_named_pages(retriever, top_k):
    assert len(retriever.retrieve("famous things in Turkey and France", top_k=top_k).chunks) == top_k


@pytest.mark.parametrize("top_k", [1, 2])
def test_top_k_is_respected_for_one_page(retriever, top_k):
    assert len(retriever.retrieve("Who was Marie Curie?", top_k=top_k).chunks) == top_k


def test_every_named_page_gets_a_passage_even_at_top_k_one(retriever):
    result = retriever.retrieve("Compare Marie Curie and Albert Einstein.", top_k=1)
    assert set(entities(result)) == {"Marie Curie", "Albert Einstein"}


def test_questions_with_no_named_page_surface_several_pages(retriever):
    found = entities(retriever.retrieve("Which famous place is located in Turkey?", top_k=3))
    assert {"Hagia Sophia", "Cappadocia"} <= set(found)


def test_query_is_embedded_once_per_retrieval(retriever, fake_ollama):
    for question in ["Compare Marie Curie and Albert Einstein.", "famous things in Turkey"]:
        before = len(fake_ollama.embed_calls)
        retriever.retrieve(question)
        assert len(fake_ollama.embed_calls) - before == 1


def test_repeated_query_uses_the_embedding_cache(retriever, fake_ollama):
    retriever.retrieve("Who was Marie Curie?")
    before = len(fake_ollama.embed_calls)
    retriever.retrieve("Who was Marie Curie?")
    assert len(fake_ollama.embed_calls) == before


def test_query_embedding_uses_the_model_prefix(retriever, fake_ollama):
    retriever.retrieve("Who was Marie Curie?")
    assert fake_ollama.embed_calls[-1][0].startswith("search_query: ")


def test_keyword_only_and_vector_only_modes_work(rag):
    for settings in (RetrievalSettings(use_vector=False), RetrievalSettings(use_lexical=False)):
        only = Retriever(rag.store, rag.embedder, rag.router, settings=settings)
        assert only.retrieve("Which famous place is located in Turkey?").chunks


def test_firm_keyword_cue_restricts_the_type(retriever):
    result = retriever.retrieve("Which scientist lived in Turkey?")
    assert {c["metadata"]["type"] for c in result.chunks} == {"person"}


def test_weak_keyword_cue_does_not_exclude_pages(retriever):
    result = retriever.retrieve("Who lived in Turkey?")
    assert "place" in {c["metadata"]["type"] for c in result.chunks}


def test_type_filter_can_be_turned_off(rag):
    unfiltered = Retriever(rag.store, rag.embedder, rag.router, settings=RetrievalSettings(type_filter=False))
    result = unfiltered.retrieve("Which scientist lived in Turkey?")
    assert "place" in {c["metadata"]["type"] for c in result.chunks}


def test_overview_question_starts_from_the_page_introduction(retriever):
    chunks = retriever.retrieve("Who was Albert Einstein?", top_k=2).chunks
    assert 0 in {c["metadata"]["chunk_idx"] for c in chunks}


def test_comparison_includes_each_page_introduction(retriever):
    chunks = retriever.retrieve("Compare Marie Curie and Albert Einstein.", top_k=2).chunks
    assert {(c["metadata"]["entity"], c["metadata"]["chunk_idx"]) for c in chunks} == {("Marie Curie", 0), ("Albert Einstein", 0)}


def test_specific_question_is_not_forced_to_the_introduction(retriever):
    chunks = retriever.retrieve("What Nobel Prize research did Marie Curie do on radioactivity?", top_k=1).chunks
    assert chunks[0]["metadata"]["section"] == "Career"


def test_followup_searches_with_the_page_name(retriever):
    result = retriever.retrieve("Where was she born?", previous_entities=["Marie Curie"])
    assert result.search_query.startswith("Marie Curie")
    assert set(entities(result)) == {"Marie Curie"}
