"""The catalogue: where it comes from, and what shapes are refused.

This is the port's one design change, so it gets the strictest tests. The
catalogue is an INPUT whose default source is the live registry, and every way
of getting one wrong has to say which way.
"""

import io
import json
import socket
import urllib.error

import pytest

from sadt_agent import catalog
from sadt_agent.errors import CatalogError, ToolInputError, ToolUnavailableError

from conftest import CATALOG


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def test_a_catalog_file_is_read_and_its_source_named(catalog_file):
    tools, source = catalog.load_catalog(catalog_file)
    assert [tool["name"] for tool in tools] == [tool["name"] for tool in CATALOG]
    assert source.startswith("file:")


def test_a_missing_catalog_file_is_the_callers_and_names_the_argument(tmp_path):
    """Raised as `ToolInputError` itself: the server maps the EXACT class name,
    so a subclass would have reached the caller as a 500. The argument is named
    rather than the file, which redaction would blank anyway."""
    with pytest.raises(ToolInputError) as raised:
        catalog.load_catalog(tmp_path / "nowhere.json")
    assert type(raised.value) is ToolInputError
    assert "'catalog_file' could not be read" in str(raised.value)
    assert "FileNotFoundError" in str(raised.value)


def test_a_catalog_file_that_is_not_json_says_so(tmp_path):
    path = tmp_path / "catalog.json"
    path.write_text("scripts:\n  - name: ali_cbct\n", encoding="utf-8")
    with pytest.raises(ToolInputError) as raised:
        catalog.load_catalog(path)
    assert "not valid JSON" in str(raised.value)


def test_with_no_catalog_and_no_registry_both_ways_forward_are_named(monkeypatch):
    """The refusal has to say what to do, because there are exactly two options
    and a caller outside a server only has one of them."""
    monkeypatch.delenv(catalog.API_ENV, raising=False)
    with pytest.raises(ToolUnavailableError) as raised:
        catalog.load_catalog("")
    message = str(raised.value)
    assert "catalog_file" in message and catalog.API_ENV in message


def test_the_live_registry_is_the_default_source(monkeypatch):
    """The whole design change: with no `catalog` argument the tool reads the
    server's own GET /tools, which is generated from each tool's signature and
    cannot drift from it."""
    monkeypatch.setenv(catalog.API_ENV, "http://127.0.0.1:8000")
    seen = {}

    def fake_urlopen(url, timeout=None):
        seen["url"] = url
        seen["timeout"] = timeout
        return io.BytesIO(json.dumps(CATALOG).encode("utf-8"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    tools, source = catalog.load_catalog("", timeout_seconds=7)

    assert seen["url"] == "http://127.0.0.1:8000/tools"
    assert seen["timeout"] == 7
    assert len(tools) == len(CATALOG)
    assert source == "registry:http://127.0.0.1:8000"


def test_a_catalog_file_wins_over_the_registry(monkeypatch, catalog_file):
    """Explicit beats ambient, so a test or an offline run is never at the mercy
    of whatever the environment happens to point at."""
    monkeypatch.setenv(catalog.API_ENV, "http://127.0.0.1:8000")
    _tools, source = catalog.load_catalog(catalog_file)
    assert source.startswith("file:")


def test_an_unreachable_registry_names_the_url_and_the_alternative(monkeypatch):
    monkeypatch.setenv(catalog.API_ENV, "http://127.0.0.1:9")

    def refuse(url, timeout=None):
        raise OSError("Connection refused")

    monkeypatch.setattr("urllib.request.urlopen", refuse)
    with pytest.raises(ToolUnavailableError) as raised:
        catalog.load_catalog("")
    message = str(raised.value)
    assert "http://127.0.0.1:9/tools" in message
    assert "catalog_file" in message


@pytest.mark.parametrize("timeout", [
    TimeoutError("timed out"),
    urllib.error.URLError(socket.timeout("timed out")),
])
def test_a_slow_registry_is_reported_as_a_timeout_not_as_unreachable(
    monkeypatch, timeout
):
    monkeypatch.setenv(catalog.API_ENV, "http://127.0.0.1:8000")

    def stall(url, **_kwargs):
        raise timeout

    monkeypatch.setattr("urllib.request.urlopen", stall)
    with pytest.raises(ToolUnavailableError) as raised:
        catalog.load_catalog("", timeout_seconds=7)
    assert str(raised.value).startswith(
        "reading the live registry exceeded timeout_seconds=7"
    )


def test_a_malformed_registry_is_the_servers_fault_not_the_callers(monkeypatch):
    """The same shape checks as a caller's file, but nobody but the server
    wrote this payload, so it must not come back as a 422."""
    monkeypatch.setenv(catalog.API_ENV, "http://127.0.0.1:8000")
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda url, timeout=None: io.BytesIO(b'[{"description": "x"}]'),
    )
    with pytest.raises(CatalogError) as raised:
        catalog.load_catalog("")
    assert not isinstance(raised.value, ValueError)
    assert "no 'name'" in str(raised.value)


def test_no_token_is_sent_to_the_registry(monkeypatch):
    """GET /tools needs none, and the server strips API_TOKEN from a tool's
    environment on purpose. A tool that sent one would be relying on something
    it is not given."""
    monkeypatch.setenv(catalog.API_ENV, "http://127.0.0.1:8000")
    captured = {}

    def fake_urlopen(url, timeout=None):
        captured["url"] = url
        return io.BytesIO(b"[]")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(CatalogError):
        catalog.load_catalog("")
    assert isinstance(captured["url"], str)  # a plain URL, not a Request with headers


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------

def test_both_published_shapes_of_get_tools_are_accepted():
    assert len(catalog.normalise(CATALOG)) == len(CATALOG)
    assert len(catalog.normalise({"tools": CATALOG})) == len(CATALOG)


def test_an_empty_catalog_is_refused_rather_than_routed_against():
    with pytest.raises(ToolInputError) as raised:
        catalog.normalise([])
    assert "no tools" in str(raised.value)


def test_a_catalog_that_is_not_a_list_says_what_it_got():
    with pytest.raises(ToolInputError) as raised:
        catalog.normalise(42)
    assert "int" in str(raised.value)


def test_a_tool_with_no_name_is_refused():
    with pytest.raises(ToolInputError) as raised:
        catalog.normalise([{"description": "x", "arguments": {}}])
    assert "no 'name'" in str(raised.value)


def test_two_spellings_of_one_tool_name_are_refused():
    """The same rule the server's registry applies at startup. Routing to the
    wrong spelling picks the wrong schema and fills the wrong parameters."""
    entries = [
        {"name": "Batch_Dental_Seg", "arguments": {}},
        {"name": "BatchDentalSeg", "arguments": {}},
    ]
    with pytest.raises(ToolInputError) as raised:
        catalog.normalise(entries)
    assert "same tool name" in str(raised.value)


def test_an_argument_with_no_name_is_refused_rather_than_becoming_one_phantom():
    """Upstream's `complete_with_defaults` keyed defaults on `p.get("name", "")`
    over a LIST of parameter objects, so every parameter that happened to lack a
    name collapsed into a single `""` entry and was then injected into the
    proposal as an argument no tool declares. A mapping keyed by name cannot
    express that shape, and an empty key is refused outright."""
    entries = [{"name": "T", "arguments": {"": {"type": "str"}}}]
    with pytest.raises(ToolInputError) as raised:
        catalog.normalise(entries)
    assert "empty name" in str(raised.value)


def test_arguments_as_a_list_is_refused_and_says_what_is_wanted():
    """Upstream's manifest shape. Accepting it would reintroduce the nameless
    parameter the test above closes."""
    entries = [{"name": "T", "arguments": [{"name": "scans", "type": "path"}]}]
    with pytest.raises(ToolInputError) as raised:
        catalog.normalise(entries)
    assert "keyed by argument name" in str(raised.value)


def test_an_argument_with_no_type_is_refused():
    with pytest.raises(ToolInputError) as raised:
        catalog.normalise([{"name": "T", "arguments": {"a": {}}}])
    assert "no 'type'" in str(raised.value)


def test_an_unknown_type_is_kept_opaque_rather_than_refusing_the_catalogue():
    """A type this agent cannot fill is never proposed a value, but the rest of
    the catalogue stays routable. See test_registry_shape.py."""
    entries = [{"name": "T", "arguments": {"a": {"type": "dict[str, path]"}}}]
    tools = catalog.normalise(entries)
    assert tools[0]["arguments"]["a"]["opaque"] is True
    assert catalog.fillable_arguments(tools[0]) == {}


def test_every_type_describe_py_can_emit_is_accepted():
    arguments = {
        name.replace("[", "_").replace("]", ""): {"type": name}
        for name in catalog.KNOWN_TYPES
    }
    tools = catalog.normalise([{"name": "T", "arguments": arguments}])
    assert len(tools[0]["arguments"]) == len(catalog.KNOWN_TYPES)


def test_the_supervisor_flag_travels():
    tools = catalog.normalise([{"name": "T", "arguments": {}, "supervisor": True}])
    assert tools[0]["supervisor"] is True


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def test_a_tool_is_found_by_its_exact_name():
    tools = catalog.normalise(CATALOG)
    assert catalog.find_tool(tools, "Bone_Seg")["name"] == "Bone_Seg"


def test_a_tool_is_found_through_separators_and_case():
    """A model answers `boneseg` or `bone seg` often enough that refusing it
    would throw away good routings; the comparison is the server's own."""
    tools = catalog.normalise(CATALOG)
    for spelling in ("boneseg", "BONE_SEG", "bone-seg", "Bone Seg"):
        assert catalog.find_tool(tools, spelling)["name"] == "Bone_Seg"


def test_a_name_no_tool_carries_resolves_to_nothing():
    tools = catalog.normalise(CATALOG)
    assert catalog.find_tool(tools, "Definitely_Not_A_Tool") is None
    assert catalog.find_tool(tools, "") is None
    assert catalog.find_tool(tools, None) is None


def test_output_dir_is_never_offered_to_the_model():
    """It is filled at dispatch with the job's own output folder, and a model
    proposing one is proposing a write outside the job."""
    tools = catalog.normalise(CATALOG)
    for tool in tools:
        assert "output_dir" not in catalog.fillable_arguments(tool)


def test_a_hidden_argument_is_never_offered_to_the_model():
    """`device` is a deployment knob the tool's author kept off a clinician's
    panel; a model has no better basis for setting it than the default does."""
    tools = catalog.normalise(CATALOG)
    bone = catalog.find_tool(tools, "Bone_Seg")
    assert "device" in bone["arguments"]
    assert "device" not in catalog.fillable_arguments(bone)


def test_missing_required_lists_only_what_the_caller_can_supply():
    tools = catalog.normalise(CATALOG)
    bone = catalog.find_tool(tools, "Bone_Seg")
    assert catalog.missing_required(bone, {}) == ["scans", "model"]
    assert catalog.missing_required(bone, {"scans": "/a", "model": "/b"}) == []


def test_missing_required_keeps_signature_order():
    """Argument order is the signature's, and a client renders forms in it."""
    tools = catalog.normalise(CATALOG)
    reg = catalog.find_tool(tools, "Timepoint_Reg")
    assert catalog.missing_required(reg, {}) == ["baseline", "followup"]
