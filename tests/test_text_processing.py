import pytest

from src.chunker import chunk_sections, chunk_text, split_sentences
from src.config import normalize_ollama_host
from src.language import detect_language
from src.textnorm import fold, search_terms

PROSE = " ".join(f"Sentence number {i} talks about the subject of this article in detail." for i in range(60))


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("http://localhost:11434", "http://127.0.0.1:11434"),
        ("localhost:11434", "http://127.0.0.1:11434"),
        ("0.0.0.0:11434", "http://127.0.0.1:11434"),
        ("http://192.168.1.5:9000/", "http://192.168.1.5:9000"),
        ("https://ollama.example.com:8443", "https://ollama.example.com:8443"),
        ("127.0.0.1", "http://127.0.0.1:11434"),
    ],
)
def test_ollama_host_is_normalized(raw, expected):
    assert normalize_ollama_host(raw) == expected


def test_sentences_keep_abbreviations_and_initials_together():
    text = "Dr. Smith went to St. Louis in the U.S. in 1900. He met J. Edison there! Was it good? Yes."
    assert split_sentences(text) == [
        "Dr. Smith went to St. Louis in the U.S. in 1900.",
        "He met J. Edison there!",
        "Was it good?",
        "Yes.",
    ]


def test_chunks_respect_the_size_limit():
    chunks = chunk_text(PROSE, chunk_size=300, chunk_overlap=80)
    assert len(chunks) > 3
    assert all(len(c) <= 300 for c in chunks)


def test_chunks_start_and_end_on_sentence_boundaries():
    for chunk in chunk_text(PROSE, chunk_size=300, chunk_overlap=80):
        assert chunk.startswith("Sentence number")
        assert chunk.endswith(".")


def test_overlap_repeats_whole_sentences():
    chunks = chunk_text(PROSE, chunk_size=300, chunk_overlap=80)
    last_sentence = split_sentences(chunks[0])[-1]
    assert chunks[1].startswith(last_sentence)


def test_no_text_is_lost():
    chunks = chunk_text(PROSE, chunk_size=300, chunk_overlap=80)
    joined = " ".join(chunks)
    assert all(f"Sentence number {i} " in joined for i in range(60))


def test_oversized_sentence_is_split_at_words():
    long_sentence = " ".join(["word"] * 400) + "."
    chunks = chunk_text(long_sentence, chunk_size=200, chunk_overlap=50)
    assert all(len(c) <= 200 for c in chunks)
    assert all(not c.startswith("ord") for c in chunks)


def test_short_tail_merges_into_previous_chunk():
    text = f"{'A long opening sentence about the topic. ' * 6}Tiny end."
    chunks = chunk_text(text, chunk_size=300, chunk_overlap=0, chunk_min=50)
    assert chunks[-1].endswith("Tiny end.")
    assert len(chunks[-1]) > 50


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_text("text", chunk_size=100, chunk_overlap=100)


def test_empty_text_has_no_chunks():
    assert chunk_text("  \n ") == []


def test_chunks_never_cross_sections_and_number_across_the_page():
    chunks = chunk_sections([("", "Intro sentence."), ("History", "History sentence."), ("Legacy", "Legacy sentence.")])
    assert [(c.index, c.section) for c in chunks] == [(0, ""), (1, "History"), (2, "Legacy")]


def test_fold_handles_turkish_and_accents():
    assert fold("ATATÜRK") == "ataturk"
    assert fold("Ayasofya'nın İSTANBUL") == "ayasofya'nin istanbul"
    assert fold("Sagrada Família") == "sagrada familia"
    assert fold("Çin Seddi") == "cin seddi"


def test_search_terms_drop_stop_words_and_plurals():
    assert search_terms("Which pyramids are in the cities?") == ["pyramid", "city"]


@pytest.mark.parametrize(
    "question, language",
    [
        ("Who was Atatürk?", "en"),
        ("Where is Sagrada Família?", "en"),
        ("Atatürk ne zaman doğdu?", "tr"),
        ("Mona Lisa'yı kim yaptı?", "tr"),
        ("Ayasofya nerede?", "tr"),
        ("Compare Mount Fuji and Mount Everest.", "en"),
    ],
)
def test_language_detection(question, language):
    assert detect_language(question) == language


def test_language_detection_on_every_golden_question():
    from eval.metrics import load_golden

    wrong = [i["id"] for i in load_golden() if detect_language(i["question"]) != i["lang"]]
    assert wrong == []
