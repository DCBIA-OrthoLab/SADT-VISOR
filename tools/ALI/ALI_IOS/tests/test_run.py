"""ALI_IOS: the catalog it publishes, and the weights it recognises.

These moved out of ALI_CBCT's test file when ALI became two tools. They were
still there, importing `sadt_ali.ios`, which no virtualenv has provided since
the split -- so ALI_CBCT's whole suite failed to collect and ALI_IOS had no
tests at all. Found by running each tool's suite in its own interpreter.
"""

import os
import typing

import pytest

from sadt_ali_ios import catalog, engine
from sadt_ali_ios.errors import ToolInputError
from sadt_ali_ios import run


def _choices(argument):
    """The `Literal` options `run()` publishes for one argument."""
    hint = typing.get_type_hints(run)[argument]
    if typing.get_origin(hint) is list:
        hint = typing.get_args(hint)[0]
    return list(typing.get_args(hint))


# Copied from ALI_CBCT's suite rather than shared: the two tools are
# separate packages with separate virtualenvs, and CONTRIBUTING.md says a
# test helper is duplicated rather than given a package of its own.
def write_surface(path, labelled=True):
    """A minimal .vtk polydata, optionally carrying a tooth-label array."""
    vtk = pytest.importorskip("vtk")

    points = vtk.vtkPoints()
    for coordinates in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 1)):
        points.InsertNextPoint(*coordinates)

    polys = vtk.vtkCellArray()
    for triangle in ((0, 1, 2), (0, 1, 3), (1, 2, 3), (0, 2, 3)):
        polys.InsertNextCell(3)
        for point_id in triangle:
            polys.InsertCellPoint(point_id)

    surface = vtk.vtkPolyData()
    surface.SetPoints(points)
    surface.SetPolys(polys)

    if labelled:
        labels = vtk.vtkIntArray()
        labels.SetName("Universal_ID")
        for value in (8, 8, 8, 8):
            labels.InsertNextValue(value)
        surface.GetPointData().AddArray(labels)

    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(surface)
    writer.Write()
    return str(path)

def write_ios_bundle(root, names):
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        (root / name).write_bytes(b"fake checkpoint")
    return str(root)

def test_ios_offers_only_landmark_types_a_model_predicts():
    """R, RIP and OIP were selectable in the Slicer UI and predicted by
    nothing: no network produced them and no label table contained them.
    Ticking them did literally nothing."""
    offered = {lm_type for types in catalog.NETWORKS.values() for lm_type in types}
    assert offered == {"O", "MB", "DB", "CL", "CB", "MG"}
    assert not offered & {"R", "RIP", "OIP"}


def test_ios_tooth_numbering_matches_the_shipped_label_tables():
    assert catalog.UNIVERSAL_NUMBERS["Upper"]["UL7"] == 15
    assert catalog.UNIVERSAL_NUMBERS["Upper"]["UR7"] == 2
    assert catalog.UNIVERSAL_NUMBERS["Lower"]["LL7"] == 18
    assert catalog.UNIVERSAL_NUMBERS["Lower"]["LR7"] == 31
    # Tooth 8 is UR1: the occlusal network's three channels, in channel order.
    assert catalog.LABELS["O"]["8"] == ["UR1O", "UR1MB", "UR1DB"]
    assert catalog.LABELS["C"]["8"] == ["UR1CL", "UR1CB"]


def test_ios_network_codes_from_a_selection():
    assert catalog.network_codes(["Occlusal"]) == ("O",)
    assert catalog.network_codes(["O"]) == ("O",)
    assert catalog.network_codes(None) == catalog.NETWORK_CODES
    with pytest.raises(ValueError, match="Occlusal"):
        catalog.network_codes(["Buccal"])


def test_the_published_networks_are_the_catalogs_own():
    assert _choices("networks") == list(catalog.NETWORK_NAMES)


def test_ios_weight_discovery_reads_the_published_names(tmp_path):
    """The real ALIDDM bundle: Upper_O_model.pth, Lower_C_model.pth, ..."""
    bundle = write_ios_bundle(
        tmp_path / "bundle",
        ["Upper_O_model.pth", "Lower_O_model.pth", "Upper_C_model.pth", "Lower_C_model.pth"],
    )
    weights, unrecognized = engine.discover_weights(bundle)

    assert weights["O"].keys() == {"Upper", "Lower"}
    assert weights["C"].keys() == {"Upper", "Lower"}
    assert unrecognized == []


def test_an_ios_checkpoint_with_no_jaw_token_is_reported_not_assumed_upper(tmp_path):
    """The original treated every file not containing "Lower" as upper-jaw
    weights, so a bundle missing its mandibular model quietly predicted the
    lower arch with the maxillary one."""
    bundle = write_ios_bundle(tmp_path / "bundle", ["model_O.pth", "Upper_C_model.pth"])
    weights, unrecognized = engine.discover_weights(bundle)

    assert unrecognized == ["model_O.pth"]
    assert "O" not in weights
    assert weights["C"] == {"Upper": str(tmp_path / "bundle" / "Upper_C_model.pth")}


def test_a_mesh_without_tooth_labels_names_the_tool_that_makes_them(tmp_path):
    """The handoff `ALILogic.ensure_segmented()` used to make in-process.

    Tools do not call each other any more, so this cannot segment the mesh
    itself -- but it can say exactly what to run, which is the difference
    between a fixable request and "no known tooth number is present".
    """
    labelled = write_surface(tmp_path / "in" / "good.vtk", labelled=True)
    raw = write_surface(tmp_path / "in" / "raw.vtk", labelled=False)

    with pytest.raises(ToolInputError) as raised:
        engine.require_labels([(labelled, "good.vtk"), (raw, "raw.vtk")])

    message = str(raised.value)
    assert "Crown_Seg" in message
    assert "1 of 2" in message
    # The array names it looked for, so the fix is actionable without reading
    # the source.
    assert "Universal_ID" in message


def test_a_fully_labelled_batch_passes_the_check(tmp_path):
    mesh = write_surface(tmp_path / "in" / "arch.vtk", labelled=True)
    assert engine.require_labels([(mesh, "arch.vtk")]) is None


def test_mucogingival_is_offered_but_not_on_by_default():
    """One point per lower tooth on the gingival margin, wanted by a mandible
    registration and by nobody asking for crown landmarks. On by default would
    add a third pass over every mesh of every existing request."""
    import inspect

    assert catalog.NETWORK_NAMES["Mucogingival"] == "MG"
    assert "Mucogingival" in _choices("networks")
    assert inspect.signature(run).parameters["networks"].default == [
        "Occlusal", "Cervical"
    ]


def test_mucogingival_runs_on_the_mandible_only():
    """It was trained on the mandible alone, so a maxilla is not a missing
    model -- it is a question the network cannot be asked."""
    assert catalog.NETWORK_JAWS["MG"] == ("Lower",)
    # The other two are unrestricted, and must stay that way.
    assert "O" not in catalog.NETWORK_JAWS
    assert "C" not in catalog.NETWORK_JAWS


def test_the_mucogingival_names_are_positional_not_derived():
    """Six MG output names collide with the TRAINING name of a DIFFERENT tooth
    -- LR1MG is the training name of tooth 25 and the output name of tooth 26 --
    because tooth 25 carries the midline name L0MG and shifts the right side by
    one. Deriving `<tooth><type>` here would mislabel half the arch."""
    labels = catalog.LABELS["MG"]

    assert labels["19"] == ["LL6MG"]      # first trained tooth
    assert labels["25"] == ["L0MG"]       # the midline, not "LR1MG"
    assert labels["26"] == ["LR1MG"]      # shifted by one against the numbers
    assert labels["31"] == ["LR6MG"]
    # Tooth 18 was excluded from training and has no MG label at all.
    assert "18" not in labels
    assert len(labels) == 13 == len(catalog.MG_TEETH)


def test_every_mucogingival_tooth_has_an_aim_offset():
    """The cameras aim at the landmark's expected position rather than at a
    flat drop below the tooth centre, which only ever matched the incisors: on
    the molars the landmark is ~0.15 further buccal and fell outside the render
    entirely. A tooth with no offset would be back to that."""
    assert set(catalog.MG_AIM_OFFSET) == set(catalog.MG_TEETH)
    for tooth, offset in catalog.MG_AIM_OFFSET.items():
        assert len(offset) == 3, tooth
        # Below the crown, always: the gingival margin is under it.
        assert offset[2] < 0, tooth


# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------

def test_progress_writes_one_line_per_event_and_nothing_when_unset(tmp_path, monkeypatch):
    """The engine's own loop cannot run here -- it needs pytorch3d, which
    publishes no usable wheel -- so what is tested is this package's copy of
    the helper the loop calls, on the contract every tool's copy holds to.
    """
    import json

    from sadt_ali_ios import progress

    monkeypatch.delenv(progress.VARIABLE, raising=False)
    progress.report(1, 4, "mesh")  # a no-op, and above all not an exception

    events_file = tmp_path / "events.jsonl"
    monkeypatch.setenv(progress.VARIABLE, str(events_file))
    progress.report(3, 4, "mesh")

    assert json.loads(events_file.read_text()) == {
        "fraction": 0.5, "message": "mesh 3 of 4",
    }


def test_the_engine_reports_the_position_it_already_logs():
    """The counter was there and went only to the server's log. The rule the
    log line states -- position in the batch, never the mesh's name -- is why
    the progress call could take it unchanged."""
    source = open(engine.__file__, encoding="utf-8").read()
    assert 'progress.report(mesh_index, len(meshes), "mesh")' in source
# The published schema and the catalog cannot drift apart
#
# `Literal` takes literals only, so it cannot be built from the catalog. That
# makes the signature a SECOND declaration of the same sets, and these are what
# keep the two honest: an option added to one and not the other is either
# unselectable from the client, or offered and then refused.
# ---------------------------------------------------------------------------

def test_every_published_default_is_one_of_its_own_options():
    """A default outside its option list gives the client a picker that cannot
    produce the value the tool starts from."""
    import inspect

    for argument in ("networks", "device"):
        default = inspect.signature(run).parameters[argument].default
        options = _choices(argument)
        for value in (default if isinstance(default, list) else [default]):
            assert value in options, (argument, value)


def test_the_published_devices_are_the_two_the_engine_resolves():
    """`resolve_device` falls back to CPU when no card is visible, so both
    values are always runnable -- but a third would reach torch verbatim."""
    assert _choices("device") == ["cuda", "cpu"]


def test_the_marker_is_constant_and_no_caller_can_move_it():
    """It was an argument, `prediction_ID`, defaulting to "Pred", and that
    made the marker a property of the REQUEST: nothing downstream could
    predict it, while pairing a scan with its landmarks and grouping results
    by patient both have to strip a marker they can predict.

    Both halves are asserted. The argument is gone, so no caller can move it;
    and what the tool publishes is what it writes, so the two cannot drift.
    """
    import inspect

    from sadt_ali_ios import OUTPUT_SUFFIXES, PREDICTION_ID

    assert "prediction_ID" not in inspect.signature(run).parameters
    assert PREDICTION_ID == "Pred"
    assert "_lm_" + PREDICTION_ID in OUTPUT_SUFFIXES


# What the SERVER adds to a tool that calls another, so the tool never declares
# it and `run()` has no such parameter. Restated rather than imported:
# `scripts/describe.py` is not on a tool venv's path, and this is the one fact
# the two files have to agree on. Its own list is `describe.INJECTED_ARGUMENTS`.
INJECTED_ARGUMENTS = ("keep_intermediate",)


def test_the_layout_only_names_arguments_the_signature_offers():
    """`describe.py` refuses a hint naming an argument `run()` does not take,
    so a stale entry here is a tool that will not publish its schema at all.

    Its rule has two halves and this used to state one: an INJECTED argument
    may be named too. `keep_intermediate` is generic, but "Keep the labelled
    meshes" is not -- only the tool knows what its chain produces, which is why
    it gets a layout without a parameter.
    """
    import inspect

    from sadt_ali_ios.layout import LAYOUT

    offered = set(inspect.signature(run).parameters) | set(INJECTED_ARGUMENTS)
    assert set(LAYOUT) <= offered


def test_an_injected_argument_is_the_only_hint_without_a_parameter():
    """The exception is narrow on purpose: a typo in a layout key would
    otherwise pass as "probably injected" and the hint would silently do
    nothing."""
    import inspect

    from sadt_ali_ios.layout import LAYOUT

    parameters = set(inspect.signature(run).parameters)
    assert set(LAYOUT) - parameters <= set(INJECTED_ARGUMENTS)


def test_the_network_selection_is_returned_in_declaration_order():
    """Not the caller's order: the report, the passes and the log lines all
    read from it, and a run that reordered itself per request would make two
    identical batches unreadable side by side."""
    assert catalog.network_codes(["Cervical", "Occlusal"]) == ("O", "C")
    assert catalog.network_codes(["Mucogingival", "Occlusal"]) == ("O", "MG")


def test_a_repeated_network_is_asked_for_once():
    """A client rendering check boxes can send the same family twice; running
    it twice doubles a pass over every mesh for an identical answer."""
    assert catalog.network_codes(["Occlusal", "O", "Occlusal"]) == ("O",)


def test_an_empty_selection_stays_empty_rather_than_meaning_everything():
    """`None` means "the argument was omitted" and falls back to every
    network; `[]` means "the user cleared the box" and has to be refused by
    name, not silently turned into a full run."""
    assert catalog.network_codes([]) == ()
    assert catalog.network_codes(None) == catalog.NETWORK_CODES


def test_an_unknown_network_names_the_value_it_was_given():
    """The message reaches the user as a 422, so it has to say WHICH name was
    wrong as well as which are right."""
    with pytest.raises(ValueError) as raised:
        catalog.network_codes(["Buccal"])
    message = str(raised.value)
    assert "'Buccal'" in message
    for known in catalog.NETWORK_NAMES:
        assert known in message


def test_the_display_names_round_trip_to_the_codes_and_back():
    """The report publishes display names and the engine keys on codes; a
    one-way table would report a network under a name no client offers."""
    for display, code in catalog.NETWORK_NAMES.items():
        assert catalog.NETWORK_DISPLAY_NAMES[code] == display
    assert set(catalog.NETWORK_DISPLAY_NAMES) == set(catalog.NETWORK_CODES)


def test_every_network_has_a_label_table_of_the_right_width():
    """The channels of a network's output are read positionally, so a label
    table one entry short renames every landmark after the gap."""
    for code, types in catalog.NETWORKS.items():
        for number, names in catalog.LABELS[code].items():
            assert len(names) == len(types), (code, number)


def test_the_two_jaws_partition_the_universal_numbering():
    """A tooth in both jaws would be predicted twice, once against the wrong
    weights."""
    upper = set(catalog.UNIVERSAL_NUMBERS["Upper"].values())
    lower = set(catalog.UNIVERSAL_NUMBERS["Lower"].values())
    assert upper == set(range(2, 16))
    assert lower == set(range(18, 32))
    assert not upper & lower
    assert set(catalog.JAW_OF_NUMBER) == upper | lower


def test_the_crown_networks_cover_every_tooth_of_both_jaws():
    """A tooth missing from a label table is one the engine walks past in
    silence."""
    every_number = {str(number) for number in catalog.JAW_OF_NUMBER}
    assert set(catalog.LABELS["O"]) == every_number
    assert set(catalog.LABELS["C"]) == every_number


def test_the_mucogingival_teeth_are_the_thirteen_that_were_trained():
    """Tooth 18 was excluded from training, so asking for it is asking a
    question the network was never shown."""
    assert catalog.MG_TEETH == tuple(range(19, 32))
    assert 18 not in catalog.MG_TEETH
    assert len(catalog.MG_OUTPUT_NAME) == len(catalog.MG_TEETH)
