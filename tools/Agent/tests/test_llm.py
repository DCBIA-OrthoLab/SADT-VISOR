"""The HTTP edge: where the model is, what a failure says, and what is refused.

Three of these tests assert an ABSENCE, and each absence is a whole upstream
code path: installing Ollama, pulling a model, and starting a server.
"""

import io
import json
import logging
import socket
import urllib.error

import pytest

from sadt_agent import llm
from sadt_agent.errors import ModelAnswerError, ToolUnavailableError


def response(payload):
    return io.BytesIO(json.dumps(payload).encode("utf-8"))


def answer(content):
    return {"message": {"role": "assistant", "content": content}}


# ---------------------------------------------------------------------------
# Where the endpoint is
# ---------------------------------------------------------------------------

def test_the_argument_wins(monkeypatch):
    monkeypatch.setenv(llm.ENDPOINT_ENV, "http://elsewhere:1234")
    assert llm.resolve_endpoint("http://asked:9999") == "http://asked:9999"


def test_the_environment_is_used_when_nothing_is_asked(monkeypatch):
    monkeypatch.setenv(llm.ENDPOINT_ENV, "http://elsewhere:1234")
    assert llm.resolve_endpoint("") == "http://elsewhere:1234"


def test_loopback_is_the_last_resort(monkeypatch):
    monkeypatch.delenv(llm.ENDPOINT_ENV, raising=False)
    assert llm.resolve_endpoint("") == llm.DEFAULT_ENDPOINT


def test_a_host_and_port_with_no_scheme_is_accepted(monkeypatch):
    """`OLLAMA_HOST=127.0.0.1:11434` is how Ollama's own documentation spells
    it, and urllib will not open that."""
    monkeypatch.setenv(llm.ENDPOINT_ENV, "127.0.0.1:11434")
    assert llm.resolve_endpoint("") == "http://127.0.0.1:11434"


def test_a_trailing_slash_never_doubles(monkeypatch):
    assert llm.resolve_endpoint("http://host:1/") == "http://host:1"


# ---------------------------------------------------------------------------
# The call
# ---------------------------------------------------------------------------

def test_the_request_carries_the_model_the_messages_and_no_stream(monkeypatch):
    sent = {}

    def fake_urlopen(request, timeout=None):
        sent["url"] = request.full_url
        sent["body"] = json.loads(request.data.decode("utf-8"))
        sent["timeout"] = timeout
        return response(answer("hello"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    content = llm.chat(
        "http://h:1", "qwen3:8b", [{"role": "user", "content": "hi"}],
        json_format=True, temperature=0.25, seed=7, timeout_seconds=11,
    )

    assert content == "hello"
    assert sent["url"] == "http://h:1/api/chat"
    assert sent["timeout"] == 11
    assert sent["body"]["model"] == "qwen3:8b"
    assert sent["body"]["stream"] is False
    assert sent["body"]["format"] == "json"
    assert sent["body"]["options"] == {"temperature": 0.25, "seed": 7}


def test_json_format_is_only_asked_for_when_wanted(monkeypatch):
    sent = {}

    def fake_urlopen(request, timeout=None):
        sent["body"] = json.loads(request.data.decode("utf-8"))
        return response(answer("prose"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    llm.chat("http://h:1", "m", [], json_format=False)
    assert "format" not in sent["body"]


def test_no_authorization_header_is_ever_sent(monkeypatch):
    """The server strips `API_TOKEN` from a tool's environment on purpose; a
    tool that sent one anywhere would be leaking a credential it is not given."""
    sent = {}

    def fake_urlopen(request, timeout=None):
        sent["headers"] = dict(request.header_items())
        return response(answer("ok"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    llm.chat("http://h:1", "m", [])
    assert not any("auth" in key.lower() for key in sent["headers"])


# ---------------------------------------------------------------------------
# Failures
# ---------------------------------------------------------------------------

def test_an_unreachable_endpoint_is_a_deployment_failure_naming_the_url(monkeypatch):
    """`ToolUnavailableError` maps to 503: no argument the caller changes makes
    an absent server appear."""

    def refuse(request, timeout=None):
        raise urllib.error.URLError("Connection refused")

    monkeypatch.setattr("urllib.request.urlopen", refuse)
    with pytest.raises(ToolUnavailableError) as raised:
        llm.chat("http://127.0.0.1:9", "m", [])
    message = str(raised.value)
    assert "http://127.0.0.1:9" in message
    assert "does not install" in message


@pytest.mark.parametrize("error", [
    TimeoutError("timed out"),
    socket.timeout("timed out"),
    urllib.error.URLError(socket.timeout("timed out")),
    urllib.error.URLError(TimeoutError("timed out")),
])
def test_a_timeout_names_the_call_and_the_limit_not_a_missing_server(
    monkeypatch, error
):
    """`TimeoutError` is an `OSError`, so a server that was up but slow used to
    be reported as "No Ollama server answered" -- sending the operator to start
    a service that was already running."""

    def stall(request, timeout=None):
        raise error

    monkeypatch.setattr("urllib.request.urlopen", stall)
    with pytest.raises(ToolUnavailableError) as raised:
        llm.chat("http://h:1", "qwen3:8b", [], timeout_seconds=42, call="router call")
    message = str(raised.value)
    assert message.startswith("router call exceeded timeout_seconds=42")
    assert "No Ollama server" not in message
    assert raised.value.__cause__ is error


def test_a_refused_connection_is_still_reported_as_no_server(monkeypatch):
    """The timeout branch must not swallow the case it was split from."""

    def refuse(request, timeout=None):
        raise urllib.error.URLError(ConnectionRefusedError(111, "Connection refused"))

    monkeypatch.setattr("urllib.request.urlopen", refuse)
    with pytest.raises(ToolUnavailableError) as raised:
        llm.chat("http://h:1", "m", [])
    assert "No Ollama server answered" in str(raised.value)
    assert raised.value.__cause__ is not None


def test_a_missing_model_is_named_with_its_whole_tag(monkeypatch):
    """THE defect. `Agent/Agent.py:915` did `model_name.split(':')[0]`, so a
    missing `qwen3:8b` was reported as "run: ollama pull qwen3" -- a different
    model, several gigabytes, and still not the one the tool asks for."""

    def not_found(request, timeout=None):
        raise urllib.error.HTTPError(
            request.full_url, 404, "Not Found", {},
            io.BytesIO(b'{"error": "model qwen3:8b not found, try pulling it"}'),
        )

    monkeypatch.setattr("urllib.request.urlopen", not_found)
    with pytest.raises(ToolUnavailableError) as raised:
        llm.chat("http://h:1", "qwen3:8b", [])
    message = str(raised.value)
    assert "ollama pull qwen3:8b" in message
    assert "ollama pull qwen3\n" not in message and "pull qwen3 " not in message


def test_a_missing_model_is_never_pulled(monkeypatch):
    """`chat_with_auto_pull` treated a 404 as a cue to download ~5 GB from
    inside a request handler. There is exactly one HTTP call here, and it is
    the one that failed."""
    calls = []

    def not_found(request, timeout=None):
        calls.append(request.full_url)
        raise urllib.error.HTTPError(
            request.full_url, 404, "Not Found", {}, io.BytesIO(b"not found")
        )

    monkeypatch.setattr("urllib.request.urlopen", not_found)
    with pytest.raises(ToolUnavailableError):
        llm.chat("http://h:1", "qwen3:8b", [])
    assert calls == ["http://h:1/api/chat"]


def test_another_http_status_is_reported_with_its_body(monkeypatch):
    def server_error(request, timeout=None):
        raise urllib.error.HTTPError(
            request.full_url, 500, "Internal Server Error", {},
            io.BytesIO(b"out of memory"),
        )

    monkeypatch.setattr("urllib.request.urlopen", server_error)
    with pytest.raises(ToolUnavailableError) as raised:
        llm.chat("http://h:1", "m", [])
    assert "500" in str(raised.value) and "out of memory" in str(raised.value)


def test_an_answer_with_no_content_field_is_reported(monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda request, timeout=None: response({"done": True})
    )
    with pytest.raises(ToolUnavailableError):
        llm.chat("http://h:1", "m", [])


def test_a_non_json_answer_is_reported(monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout=None: io.BytesIO(b"<html>proxy error</html>"),
    )
    with pytest.raises(ToolUnavailableError):
        llm.chat("http://h:1", "m", [])


def test_no_module_named_ollama_is_imported():
    """Two POSTs do not need a client library, and the tool's whole dependency
    list is empty because of it."""
    import pathlib

    source = pathlib.Path(llm.__file__).read_text(encoding="utf-8")
    assert "import ollama" not in source


# ---------------------------------------------------------------------------
# Reading the answer back
# ---------------------------------------------------------------------------

def test_a_plain_json_object_parses():
    assert llm.parse_json_object('{"tool": "X"}', "routing") == {"tool": "X"}


def test_a_reasoning_block_is_stripped():
    """qwen3 emits `<think>...</think>` around its answer even under
    `format="json"`."""
    text = '<think>Let me consider the options.</think>\n{"tool": "X"}'
    assert llm.parse_json_object(text, "routing") == {"tool": "X"}


def test_a_markdown_fence_is_stripped():
    assert llm.parse_json_object('```json\n{"tool": "X"}\n```', "routing") == {"tool": "X"}


def test_an_object_buried_in_prose_is_found():
    text = 'Sure! Here you go: {"tool": "X", "confidence": 0.9} -- hope that helps.'
    assert llm.parse_json_object(text, "routing")["tool"] == "X"


def test_a_brace_inside_a_string_does_not_end_the_object():
    text = '{"reason": "it matches {this} pattern", "tool": "X"}'
    assert llm.parse_json_object(text, "routing")["tool"] == "X"


def test_an_answer_with_no_object_is_a_server_fault_that_does_not_quote_it(caplog):
    """Not a 422: no argument the caller changes makes the model answer in
    JSON. And the model's text stays out of the message -- it can quote the
    request -- with its length there and its opening at DEBUG."""
    text = "I am not going to answer that."
    with caplog.at_level(logging.DEBUG, logger="Agent"):
        with pytest.raises(ModelAnswerError) as raised:
            llm.parse_json_object(text, "routing")
    message = str(raised.value)
    assert not isinstance(raised.value, ValueError)
    assert "routing answer was not a JSON object" in message
    assert "({} characters)".format(len(text)) in message
    assert "not going to answer" not in message
    debug = [r for r in caplog.records if r.levelno == logging.DEBUG]
    assert any("not going to answer" in r.getMessage() for r in debug)


def test_a_json_array_is_not_an_object():
    with pytest.raises(ModelAnswerError):
        llm.parse_json_object('["X"]', "routing")
