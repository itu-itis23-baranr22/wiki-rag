import json

import pytest
import requests

from src.ollama_client import ModelMissing, OllamaClient, OllamaError, OllamaUnavailable


class FakeResponse:
    def __init__(self, status=200, lines=(), failure=None, body=None):
        self.status_code = status
        self.ok = status < 400
        self.text = "details"
        self._lines = lines
        self._failure = failure
        self._body = body
        self.closed = False

    def iter_lines(self):
        yield from self._lines
        if self._failure:
            raise self._failure

    def json(self):
        return self._body

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self, response=None, failure=None):
        self.response = response
        self.failure = failure

    def post(self, *args, **kwargs):
        if self.failure:
            raise self.failure
        return self.response


def client_for(response=None, failure=None):
    return OllamaClient(host="http://127.0.0.1:11434", session=FakeSession(response, failure))


@pytest.mark.parametrize("failure", [requests.ConnectionError(), requests.Timeout()])
def test_unreachable_server_is_reported(failure):
    with pytest.raises(OllamaUnavailable):
        client_for(failure=failure).embed("model", ["text"])


def test_missing_model_is_reported_with_the_install_command():
    with pytest.raises(ModelMissing, match="ollama pull model"):
        client_for(FakeResponse(status=404)).embed("model", ["text"])


def test_other_server_errors_are_reported():
    with pytest.raises(OllamaError, match="HTTP 500"):
        client_for(FakeResponse(status=500)).embed("model", ["text"])


def test_embedding_count_is_checked():
    with pytest.raises(OllamaError, match="Expected 2"):
        client_for(FakeResponse(body={"embeddings": [[0.1]]})).embed("model", ["a", "b"])


def test_connection_lost_during_an_answer_is_an_ollama_error():
    line = json.dumps({"message": {"content": "Hello"}}).encode()
    response = FakeResponse(lines=[line], failure=requests.exceptions.ChunkedEncodingError())
    stream = client_for(response).chat_stream("model", [])
    assert next(stream)["message"]["content"] == "Hello"
    with pytest.raises(OllamaUnavailable, match="Lost the connection"):
        next(stream)
    assert response.closed


def test_closing_the_stream_closes_the_request():
    line = json.dumps({"message": {"content": "Hello"}}).encode()
    response = FakeResponse(lines=[line, line])
    stream = client_for(response).chat_stream("model", [])
    next(stream)
    stream.close()
    assert response.closed
