"""The tool as the SERVER runs it: another interpreter, a subprocess, JSON in.

`sadt_testkit.run_tool` is the same contract `server/execution/runner.py`
implements -- same coercion by annotation, same result file, same
error-class-name convention -- so these check the things an in-process test
cannot: that `folders` survives as `list[Path]`, that `catalog_file` arrives as
a `Path`, and that a failure reaches the caller under the exception name the
server maps to a status code.

They need no Ollama. Every one of them fails on purpose, before or at the model
call, which is what makes them runnable anywhere. The successful round trip is
in `test_live.py`, marked `models`.
"""

import json

import pytest

from sadt_testkit import ToolFailed, is_built, run_tool

from conftest import CATALOG

pytestmark = pytest.mark.skipif(
    not is_built("Agent"), reason="run `uv sync` in tools/Agent"
)

# Nothing listens here, and the tool must say so rather than hang.
CLOSED_PORT = "http://127.0.0.1:9"


@pytest.fixture
def catalog_path(tmp_path):
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(CATALOG), encoding="utf-8")
    return path


def test_an_unreachable_endpoint_reaches_the_caller_as_ToolUnavailableError(
    tmp_path, catalog_path
):
    """The server maps that class NAME to 503 -- a deployment problem, not the
    caller's. Through a real subprocess, so it is the name the runner writes
    into `result.json` that is being checked."""
    with pytest.raises(ToolFailed) as raised:
        run_tool(
            "Agent",
            prompt="segment the bone on my CBCTs",
            output_dir=tmp_path / "out",
            catalog_file=catalog_path,
            endpoint=CLOSED_PORT,
            timeout_seconds=10,
        )
    message = str(raised.value)
    assert "ToolUnavailableError" in message
    assert CLOSED_PORT in message


def test_a_list_of_folders_survives_the_json_round_trip(tmp_path, catalog_path):
    """`folders` is `list[Path]`; JSON has no path type, so the runner coerces
    it back. A folder with a comma in its name is the case that would have been
    split into two by upstream, and it has to survive this hop intact."""
    with pytest.raises(ToolFailed) as raised:
        run_tool(
            "Agent",
            prompt="segment the bone",
            output_dir=tmp_path / "out",
            catalog_file=catalog_path,
            folders=[tmp_path / "Smith, John", tmp_path / "out"],
            endpoint=CLOSED_PORT,
            timeout_seconds=10,
        )
    # It got past argument handling and failed at the model call, which is the
    # only thing that can fail with no endpoint.
    assert "ToolUnavailableError" in str(raised.value)


def test_a_missing_catalogue_reaches_the_caller_as_a_caller_facing_error(tmp_path):
    """Raised as `ToolInputError` itself, which the server maps to 422 with the
    message passed through. The server maps the EXACT class name, so a subclass
    of it -- as `CatalogError` used to be -- would have reached the caller as an
    opaque 500."""
    with pytest.raises(ToolFailed) as raised:
        run_tool(
            "Agent",
            prompt="anything",
            output_dir=tmp_path / "out",
            catalog_file=tmp_path / "nowhere.json",
            endpoint=CLOSED_PORT,
        )
    assert "ToolInputError" in str(raised.value)
    assert "'catalog_file' could not be read" in str(raised.value)


def test_an_empty_prompt_is_refused_in_the_subprocess_too(tmp_path, catalog_path):
    with pytest.raises(ToolFailed) as raised:
        run_tool(
            "Agent",
            prompt="   ",
            output_dir=tmp_path / "out",
            catalog_file=catalog_path,
            endpoint=CLOSED_PORT,
        )
    assert "ToolInputError" in str(raised.value)


def test_an_unset_catalog_file_stays_an_empty_string_not_the_cwd(tmp_path, monkeypatch):
    """`Path("")` is `PosixPath(".")` -- the current directory, and truthy. The
    runner keeps an empty string as a string precisely so an optional path's
    "not supplied" default is not read as a real directory. Here that means the
    live registry is tried, not `./` opened as a catalogue."""
    monkeypatch.delenv("SADT_API", raising=False)
    with pytest.raises(ToolFailed) as raised:
        run_tool(
            "Agent",
            prompt="anything",
            output_dir=tmp_path / "out",
            endpoint=CLOSED_PORT,
        )
    message = str(raised.value)
    assert "SADT_API" in message
    assert "could not be read" not in message


def test_the_live_registry_is_read_when_SADT_API_is_set(tmp_path, monkeypatch):
    """No server is listening, so this checks the URL it goes to and the way it
    reports not finding one -- not that it succeeds."""
    monkeypatch.setenv("SADT_API", CLOSED_PORT)
    with pytest.raises(ToolFailed) as raised:
        run_tool(
            "Agent",
            prompt="anything",
            output_dir=tmp_path / "out",
            endpoint=CLOSED_PORT,
        )
    assert CLOSED_PORT + "/tools" in str(raised.value)
