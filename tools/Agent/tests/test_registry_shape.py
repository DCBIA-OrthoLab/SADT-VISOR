"""The catalogue as the live registry actually publishes it.

`GET /tools` is not the schema `scripts/describe.py` writes: the server folds
each schema into its own argument vocabulary before publishing it. A `list[str]`
with choices goes out as "multichoice" with `choices` as an
{option: selected by default} mapping; a `str` with choices as "choice"; a
scalar's default as `initial`; and every argument carries every presentation
key, null when unset. `test_catalog.py` covers the schema shape a
`catalog_file` holds; this file covers the wire shape, which is what broke the
agent in production when ALI_CBCT's `regions` became a multichoice.
"""

import io
import json
import logging

from sadt_agent import catalog, validation

# Every key GET /tools publishes for an argument, as the server writes them.
WIRE_KEYS = (
    "server_selectable", "selectable_scope", "choices", "initial", "extensions",
    "label", "section", "visible_when", "options_when", "ui", "x_range",
    "y_range", "x_labels", "section_columns", "cell", "x_label", "y_label",
    "y_labels", "groups",
)


def wire(kind, required=False, description="", hidden=False, **extra):
    spec = {key: None for key in WIRE_KEYS}
    spec.update({
        "type": kind, "types": [kind], "required": required,
        "description": description, "hidden": hidden,
    })
    spec.update(extra)
    return spec


REGIONS = ["Cranial base", "Upper", "Lower", "Impacted canine"]

#: Shaped like the live registry's answer today: no tool-level description,
#: `output_kind` rather than `returns`, and ALI_CBCT's `regions` as published.
REGISTRY = [
    {
        "name": "ALI_CBCT",
        "arguments": {
            "input": wire("path", True, "One CBCT scan, or a folder of them.",
                          label="Scan or Folder", section="Inputs"),
            # A server-selectable model bundle is published as a name.
            "model": wire("str", False, "The model bundle.", hidden=True,
                          server_selectable="model"),
            "regions": wire(
                "multichoice", False, "Anatomical regions to predict.",
                hidden=True, ui="inline", section="Landmarks", label="Regions",
                choices={region: True for region in REGIONS},
            ),
            "landmarks": wire(
                "multichoice", False, "Landmarks to place.",
                ui="tabs", section="Landmarks",
                choices={"Ba": False, "S": False, "N": False},
                groups={"Cranial base": ["Ba", "S", "N"]},
            ),
            "device": wire("choice", False, "Where to run.",
                           choices={"cuda": True, "cpu": False}),
            "search_steps": wire("int", False, "Extra search steps.", initial=0),
        },
        "output_kind": "files",
    },
    {
        "name": "FlexReg",
        "arguments": {
            "scans": wire("path", True, "The intraoral scans."),
            "anterior_right": wire(
                "vec2", False, "Anterior right corner of the palate patch.",
                x_range=[0.0, 1.0], y_range=[-5.0, 5.0], ui="joystick",
                initial=[0.5, 0.0],
            ),
            "teeth": wire("list[str]", False, "Teeth bounding the patch."),
        },
        "output_kind": "files",
    },
    {
        # A tool registered in-process declares the server's own file types.
        "name": "Example_Tool",
        "arguments": {
            "label": wire("str", True, "Free-text label for this run"),
            "table": dict(wire("csv_file", True, "A table or a folder of them"),
                          types=["csv_file", "folder"]),
            "threshold": wire("float", True, "Numeric threshold parameter"),
        },
        "output_kind": "text",
    },
]


def test_the_live_registry_shape_is_a_usable_catalogue(monkeypatch):
    """The production failure, end to end: GET /tools as served today used to
    be refused whole, on ALI_CBCT's `regions`."""
    monkeypatch.setenv(catalog.API_ENV, "http://127.0.0.1:8000")
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda url, timeout=None: io.BytesIO(json.dumps(REGISTRY).encode("utf-8")),
    )
    tools, source = catalog.load_catalog("")
    assert [tool["name"] for tool in tools] == ["ALI_CBCT", "FlexReg", "Example_Tool"]
    assert source.startswith("registry:")
    assert catalog.opaque_arguments(tools) == []


def test_a_multichoice_is_a_list_of_its_options_with_the_ticked_ones_as_default():
    ali = catalog.find_tool(catalog.normalise(REGISTRY), "ALI_CBCT")
    regions = ali["arguments"]["regions"]
    assert regions["type"] == "list[str]"
    assert regions["choices"] == REGIONS
    assert regions["default"] == REGIONS
    assert ali["arguments"]["landmarks"]["default"] == []


def test_a_choice_is_a_string_with_the_selected_option_as_default():
    ali = catalog.find_tool(catalog.normalise(REGISTRY), "ALI_CBCT")
    device = ali["arguments"]["device"]
    assert device["type"] == "str"
    assert device["choices"] == ["cuda", "cpu"]
    assert device["default"] == "cuda"


def test_a_published_initial_value_becomes_the_default():
    ali = catalog.find_tool(catalog.normalise(REGISTRY), "ALI_CBCT")
    assert ali["arguments"]["search_steps"]["default"] == 0
    assert "default" not in ali["arguments"]["input"]


def test_server_file_types_are_paths():
    example = catalog.find_tool(catalog.normalise(REGISTRY), "Example_Tool")
    assert example["arguments"]["table"]["type"] == "path"
    for kind in ("file", "folder", "nifti_file", "surface_or_zip_file"):
        tools = catalog.normalise([{"name": "T", "arguments": {"a": wire(kind)}}])
        assert tools[0]["arguments"]["a"]["type"] == "path"


def test_multichoice_values_are_checked_against_the_published_options():
    ali = catalog.find_tool(catalog.normalise(REGISTRY), "ALI_CBCT")
    spec = ali["arguments"]["landmarks"]
    assert validation.coerce("landmarks", spec, "Ba, N") == ["Ba", "N"]
    values, errors, _ = validation.validate(ali, ali["arguments"], {"landmarks": ["Xx"]})
    assert values == {} and "must be one of" in errors[0]


def test_a_vec2_is_two_numbers_inside_its_ranges():
    flex = catalog.find_tool(catalog.normalise(REGISTRY), "FlexReg")
    spec = flex["arguments"]["anterior_right"]
    assert spec["type"] == "vec2"
    assert spec["default"] == [0.5, 0.0]
    assert validation.coerce("anterior_right", spec, "0.4, -2") == [0.4, -2.0]
    for bad in ([0.5], [1.5, 0.0], [0.5, 9.0], "a, b"):
        _, errors, _ = validation.validate(flex, flex["arguments"], {"anterior_right": bad})
        assert errors, bad


def test_the_router_is_offered_what_it_can_fill_and_nothing_hidden():
    ali = catalog.find_tool(catalog.normalise(REGISTRY), "ALI_CBCT")
    assert list(catalog.fillable_arguments(ali)) == [
        "input", "landmarks", "device", "search_steps",
    ]


def test_an_unknown_type_is_opaque_and_does_not_break_the_catalogue(caplog):
    """One argument the agent has never seen must not take every other tool
    down with it. It is logged, never offered to the model, and still counted
    as missing when required, so the tool is not run without it."""
    entries = REGISTRY + [{
        "name": "Future_Tool",
        "arguments": {
            "scans": wire("path", True),
            "transform": wire("matrix4x4", True, "A rigid transform."),
            "spare": wire("tensor", False),
        },
    }]
    with caplog.at_level(logging.WARNING, logger="Agent"):
        tools = catalog.normalise(entries)
    future = catalog.find_tool(tools, "Future_Tool")
    assert len(tools) == 4
    assert "matrix4x4" in caplog.text
    assert future["arguments"]["transform"] == {
        "type": "matrix4x4", "required": True, "description": "A rigid transform.",
        "hidden": False, "label": None, "section": None, "opaque": True,
    }
    assert list(catalog.fillable_arguments(future)) == ["scans"]
    assert catalog.missing_required(future, {"scans": "/a"}) == ["transform"]
    assert catalog.opaque_arguments(tools) == [
        "Future_Tool.transform (matrix4x4)", "Future_Tool.spare (tensor)",
    ]
