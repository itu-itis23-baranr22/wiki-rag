import pytest

from eval.metrics import load_golden
from src.catalog import Catalog
from src.router import Router, is_followup


@pytest.fixture
def router(catalog):
    return Router(catalog)


@pytest.mark.parametrize(
    "question, expected",
    [
        ("What jobs did Marie Curie have?", ["Marie Curie"]),
        ("How swift was the construction of the Great Wall?", ["Great Wall of China"]),
        ("What did Steve Jobs do at Pixar?", ["Steve Jobs"]),
        ("Which albums did Swift release in 2020?", ["Taylor Swift"]),
        ("What did Jobs do at Pixar?", ["Steve Jobs"]),
        ("Compare Einstein and the Eiffel Tower.", ["Albert Einstein", "Eiffel Tower"]),
        ("Who designed the Great Pyramid?", ["Giza pyramid complex"]),
        ("Ayasofya'nın mimarı kim?", ["Hagia Sophia"]),
        ("Teslanın en ünlü icadı ne?", ["Nikola Tesla"]),
        ("Çin Seddi nerede bulunur?", ["Great Wall of China"]),
        ("Who was ATATURK?", ["Mustafa Kemal Atatürk"]),
    ],
)
def test_names_are_recognised(router, question, expected):
    assert router.classify(question).entity_names == expected


@pytest.mark.parametrize(
    "question, expected",
    [
        ("Who was Einstien?", "Albert Einstein"),
        ("Where is the Eifel Tower?", "Eiffel Tower"),
        ("What did Marie Curei discover?", "Marie Curie"),
        ("Who built the Collosseum?", "Colosseum"),
        ("Where is Machu Pichu?", "Machu Picchu"),
        ("Who was Shakespear?", "William Shakespeare"),
        ("What did Isaac Newtn discover?", "Isaac Newton"),
    ],
)
def test_misspellings_are_recognised(router, question, expected):
    decision = router.classify(question)
    assert decision.entity_names == [expected]
    assert decision.reason == "fuzzy"


def test_typo_alongside_a_correct_name_finds_both(router):
    decision = router.classify("Compare Einstien and the Eiffel Tower.")
    assert set(decision.entity_names) == {"Albert Einstein", "Eiffel Tower"}
    assert decision.target == "both"


def test_questions_naming_no_page_match_nothing(router):
    named = {"refuse-06", "refuse-07", "refuse-08", "refuse-12"}  # these do name a real page
    items = [i for i in load_golden() if i["category"] in {"anonymous", "list", "refusal"} and i["id"] not in named]
    wrongly_matched = {i["id"]: router.classify(i["question"]).entity_names for i in items}
    assert {k: v for k, v in wrongly_matched.items() if v} == {}


@pytest.mark.parametrize(
    "question, previous, expected",
    [
        ("Where was she born?", ["Marie Curie"], ["Marie Curie"]),
        ("How tall is it?", ["Eiffel Tower"], ["Eiffel Tower"]),
        ("And when did he die?", ["Nikola Tesla"], ["Nikola Tesla"]),
        ("Peki ne zaman öldü?", ["Mustafa Kemal Atatürk"], ["Mustafa Kemal Atatürk"]),
        ("Peki nerede?", ["Hagia Sophia"], ["Hagia Sophia"]),
    ],
)
def test_followups_reuse_the_previous_pages(router, question, previous, expected):
    decision = router.classify(question, previous_entities=previous)
    assert decision.entity_names == expected
    assert decision.reason == "followup"


def test_new_question_does_not_inherit_previous_pages(router):
    assert router.classify("Who painted the Mona Lisa?", previous_entities=["Marie Curie"]).entity_names == []


def test_naming_a_new_page_replaces_the_previous_one(router):
    decision = router.classify("Where was Einstein born?", previous_entities=["Marie Curie"])
    assert decision.entity_names == ["Albert Einstein"]


def test_followup_without_history_stays_general(router):
    assert router.classify("Where was she born?").entity_names == []


def test_translation_helps_routing(router):
    decision = router.classify("Fransa'daki kule", translated="What is the Eiffel Tower?")
    assert decision.entity_names == ["Eiffel Tower"]


def test_keyword_cues_suggest_a_type(router):
    assert router.classify("Who discovered penicillin?").target == "person"
    assert router.classify("Where is the tallest building?").target == "place"
    assert router.classify("Tell me something").target == "both"


def test_is_followup_ignores_plain_questions():
    assert not is_followup("Who painted the Mona Lisa?")
    assert is_followup("what about her husband?")


def test_custom_pages_are_recognised_after_refresh(tmp_path):
    catalog = Catalog(tmp_path / "custom.json")
    router = Router(catalog)
    assert router.classify("Who was Ada Yonath?").entity_names != ["Ada Yonath"]
    catalog.add("Ada Yonath", "person")
    router.refresh()
    assert router.classify("Who was Ada Yonath?").entity_names == ["Ada Yonath"]
    assert router.classify("What did Yonath study?").entity_names == ["Ada Yonath"]


def test_custom_pages_persist(tmp_path):
    path = tmp_path / "custom.json"
    Catalog(path).add("Pelé", "person")
    assert Catalog(path).get("Pelé").custom
    catalog = Catalog(path)
    catalog.remove("Pelé")
    assert Catalog(path).get("Pelé") is None


def test_default_pages_cannot_be_removed(tmp_path):
    with pytest.raises(ValueError):
        Catalog(tmp_path / "custom.json").remove("Marie Curie")
