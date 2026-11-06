import asyncio

import pytest
from fastapi.testclient import TestClient

from src.ollama_client import OllamaUnavailable
from src.server import _stream, create_app
from src.wiki import AmbiguousPage, PageNotFound
from tests.conftest import make_page, parse_sse


@pytest.fixture
def client(rag):
    return TestClient(create_app(rag_factory=lambda: rag))


def test_status_reports_ready_state(client):
    report = client.get("/api/status").json()
    assert report["ready"] is True
    assert report["ollama"]["reachable"] is True
    assert report["index"]["pages"] == 8
    assert report["defaults"]["model"] == "llama3.2:3b"


def test_status_when_ollama_is_down(client, fake_ollama):
    fake_ollama.available = False
    report = client.get("/api/status").json()
    assert report["ready"] is False
    assert report["ollama"]["reachable"] is False


def test_library_lists_pages_with_passage_counts(client):
    library = client.get("/api/library").json()
    pages = {p["title"]: p for p in library["pages"]}
    assert pages["Marie Curie"]["passages"] >= 1
    assert pages["Marie Curie"]["url"].startswith("https://")
    assert pages["Mozart"]["passages"] == 0
    assert library["total_passages"] > 0


def test_chat_streams_events(client):
    response = client.post("/api/chat", json={"question": "Where was Marie Curie born?"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(response.text)
    assert [e["event"] for e in events[:2]] == ["route", "sources"]
    assert events[-1]["event"] == "done"
    assert events[1]["sources"][0]["entity"] == "Marie Curie"


def test_chat_accepts_history(client):
    body = {
        "question": "Where was she born?",
        "history": [{"question": "Who was Marie Curie?", "answer": "A physicist.", "entities": ["Marie Curie"]}],
    }
    events = parse_sse(client.post("/api/chat", json=body).text)
    assert events[0]["reason"] == "followup"


@pytest.mark.parametrize("body", [{"question": ""}, {"question": "x", "top_k": 0}, {"question": "x", "top_k": 99}, {}])
def test_chat_rejects_bad_input(client, body):
    assert client.post("/api/chat", json=body).status_code == 422


def test_ollama_failure_during_chat_becomes_an_error_event(client, rag, monkeypatch):
    def fail(*args, **kwargs):
        raise OllamaUnavailable("Cannot reach Ollama")

    monkeypatch.setattr(rag.embedder, "embed_query", fail)
    events = parse_sse(client.post("/api/chat", json={"question": "Who was Marie Curie?"}).text)
    assert events[-1]["event"] == "error"
    assert events[-1]["code"] == "ollama"


def test_requests_to_other_hosts_are_rejected(client):
    assert client.get("/api/status", headers={"host": "evil.example.com"}).status_code == 403
    assert client.get("/api/status", headers={"host": "127.0.0.1:8000"}).status_code == 200


def test_adding_a_page(client, rag, monkeypatch, tmp_path):
    from tests.conftest import PAGES

    PAGES["Pelé"] = [("", "Pelé was a Brazilian footballer who won three World Cups.")]
    monkeypatch.setattr("src.server.load_page", lambda title, refresh: make_page("Pelé"))
    try:
        events = parse_sse(client.post("/api/pages", json={"title": "Pelé", "type": "person"}).text)
        assert events[-1]["event"] == "done"
        assert events[-1]["passages"] == 1
        result = rag.ask("Who was Pelé?")
        assert result.route["entities"] == ["Pelé"]
        assert {s["entity"] for s in result.sources} == {"Pelé"}

        assert client.delete("/api/pages/Pelé").json() == {"removed": "Pelé"}
        assert rag.router.classify("Who was Pelé?").entity_names == []
        assert not [r for r in rag.store.all_records() if r["metadata"]["entity"] == "Pelé"]
    finally:
        del PAGES["Pelé"]


@pytest.mark.parametrize("error, code", [(PageNotFound("x"), "not_found"), (AmbiguousPage("x"), "ambiguous")])
def test_adding_a_bad_page_reports_why(client, rag, monkeypatch, error, code):
    def fail(title, refresh):
        raise error

    monkeypatch.setattr("src.server.load_page", fail)
    events = parse_sse(client.post("/api/pages", json={"title": "Mercury", "type": "other"}).text)
    assert events[-1] == {"event": "error", "code": code, "message": "Mercury"}
    assert rag.catalog.get("Mercury") is None


def test_unknown_page_type_is_rejected(client):
    assert client.post("/api/pages", json={"title": "X", "type": "planet"}).status_code == 422


def test_default_pages_cannot_be_deleted(client):
    assert client.delete("/api/pages/Marie Curie").status_code == 403


def test_interface_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Wiki-RAG" in response.text


class FakeRequest:
    def __init__(self, disconnect_after=None):
        self.checks = 0
        self.disconnect_after = disconnect_after

    async def is_disconnected(self):
        self.checks += 1
        return self.disconnect_after is not None and self.checks > self.disconnect_after


def endless_events(closed):
    try:
        while True:
            yield {"event": "token", "token": "word "}
    finally:
        closed.append(True)


def test_stream_closes_the_model_request_when_the_browser_disconnects():
    closed = []

    async def consume():
        return [chunk async for chunk in _stream(FakeRequest(disconnect_after=2), endless_events(closed))]

    assert len(asyncio.run(consume())) == 2
    assert closed == [True]


def test_stream_closes_the_model_request_when_it_is_cancelled():
    """Starlette cancels the response on disconnect, which must still reach the model request."""
    closed = []

    async def main():
        stream = _stream(FakeRequest(), endless_events(closed))

        async def consume():
            async for _ in stream:
                await asyncio.sleep(0)

        task = asyncio.create_task(consume())
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(main())
    assert closed == [True]


def test_interface_assets_are_served(client):
    for path in ("/app.js", "/app.css", "/i18n.js", "/api.js", "/fonts/AnekLatin-Variable.woff2"):
        assert client.get(path).status_code == 200, path


def test_interface_loads_nothing_from_other_hosts(client):
    for path in ("/", "/app.css", "/app.js", "/api.js"):
        text = client.get(path).text
        assert "cdn." not in text
        assert "fonts.googleapis" not in text
