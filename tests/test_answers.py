from src.attribution import attribute, best_evidence, support_score
from src.generator import REFUSALS, build_messages, is_refusal, strip_citations

PASSAGE = "Marie Curie was a physicist and chemist. She was born in Warsaw in 1867. She won two Nobel Prizes."


def sentences(result):
    return [s for paragraph in result["paragraphs"] for s in paragraph]


def test_citations_are_parsed_and_removed_from_text():
    result = attribute("Curie was born in Warsaw [1]. She won two Nobel Prizes [1].", [PASSAGE])
    parsed = sentences(result)
    assert [s["text"] for s in parsed] == ["Curie was born in Warsaw.", "She won two Nobel Prizes."]
    assert result["used_sources"] == [1]
    assert result["uncited_sentences"] == 0


def test_citation_styles_are_all_understood():
    passages = [PASSAGE, "The tower is in Paris."]
    for answer in (
        "Curie was born in Warsaw. [1]",
        "Curie was born in Warsaw [1, 2].",
        "Curie was born in Warsaw [1][2].",
        "Curie was born in Warsaw [Passage 1].",
    ):
        assert 1 in attribute(answer, passages)["used_sources"], answer


def test_citations_to_missing_passages_are_dropped_and_counted():
    result = attribute("Curie was born in Warsaw [7].", [PASSAGE])
    assert result["invalid_citations"] == 1
    assert sentences(result)[0]["inferred"] is True
    assert result["used_sources"] == [1]


def test_evidence_points_at_the_supporting_sentence():
    result = attribute("She was born in Warsaw in 1867 [1].", [PASSAGE])
    evidence = sentences(result)[0]["evidence"][0]
    assert PASSAGE[evidence["start"] : evidence["end"]] == "She was born in Warsaw in 1867."


def test_uncited_sentence_is_attached_when_a_passage_clearly_supports_it():
    parsed = sentences(attribute("Curie was born in Warsaw in 1867.", [PASSAGE]))[0]
    assert parsed["sources"] == [1]
    assert parsed["inferred"] is True


def test_unsupported_uncited_sentence_is_reported():
    result = attribute("Bananas grow in tropical climates.", [PASSAGE])
    assert result["uncited_sentences"] == 1
    assert result["used_sources"] == []


def test_evidence_prefers_the_focused_sentence_over_a_long_one():
    passage = (
        "Göreme became a Christian monastic centre in 300 AD. "
        "Many churches in Göreme and houses carved into rocks in the valleys all evidence Cappadocia "
        "as a centre of early Christian learning and a place of pilgrimage for visitors."
    )
    evidence = best_evidence("Göreme is a Christian monastic centre in Cappadocia.", passage)
    assert passage[evidence["start"] : evidence["end"]].startswith("Göreme became")


def test_citation_with_no_matching_sentence_keeps_the_source_without_a_span():
    parsed = sentences(attribute("Bananas grow in tropical climates [1].", [PASSAGE]))[0]
    assert parsed["sources"] == [1]
    assert parsed["evidence"] == []


def test_paragraphs_are_preserved():
    result = attribute("Curie was born in Warsaw [1].\n\nShe won two Nobel Prizes [1].", [PASSAGE])
    assert len(result["paragraphs"]) == 2


def test_support_score_weights_numbers_and_names():
    assert support_score("born in 1867", PASSAGE) > support_score("born in 1999", PASSAGE)
    assert support_score("", PASSAGE) == 0.0


def test_best_evidence_returns_none_when_nothing_matches():
    assert best_evidence("Bananas grow in tropical climates.", PASSAGE) is None


def test_turkish_answer_is_matched_through_shared_names_and_numbers():
    parsed = sentences(attribute("Marie Curie 1867 yılında Varşova'da doğdu.", [PASSAGE], "tr"))[0]
    assert parsed["sources"] == [1]
    evidence = parsed["evidence"][0]
    assert "1867" in PASSAGE[evidence["start"] : evidence["end"]]


def test_refusals_are_recognised_in_both_languages():
    assert is_refusal(REFUSALS["en"])
    assert is_refusal(REFUSALS["tr"])
    assert is_refusal("I do not know the answer.")
    assert is_refusal("The passage does not mention who composed it [1].")
    assert is_refusal("Mars'ın başkanı yoktur, çünkü bu soruda Mars'tan bahsedilmemektedir [1].")
    assert not is_refusal("Marie Curie was born in Warsaw [1].")


def test_a_late_mention_of_not_knowing_is_not_a_refusal():
    answer = "Marie Curie was born in Warsaw, and she won two Nobel Prizes for her research on radioactivity in Paris. " * 2
    assert not is_refusal(answer + "I don't know more.")


def test_strip_citations():
    assert strip_citations("Warsaw [1], Poland [2, 3].") == "Warsaw, Poland."


def test_prompt_contains_numbered_passages_and_the_refusal_sentence():
    chunks = [{"document": PASSAGE, "metadata": {"entity": "Marie Curie", "section": "Life"}}]
    messages = build_messages("Where was she born?", chunks, "tr")
    assert REFUSALS["tr"] in messages[0]["content"]
    assert "Turkish" in messages[0]["content"]
    assert "[1] (Marie Curie, Life)" in messages[-1]["content"]


def test_history_is_included_without_citations_and_limited():
    history = [{"question": f"q{i}", "answer": f"a{i} [1]."} for i in range(5)]
    messages = build_messages("next", [], "en", history)
    assert [m["content"] for m in messages[1:5]] == ["q3", "a3.", "q4", "a4."]
