"""Choosing IOS landmarks one at a time, the way ALI_CBCT already lets you.

The panel used to offer a "Landmark family" selection and nothing finer, so a
caller wanting the three points ASO registers on took a whole family -- 84 of
them -- to use three. `run()` now publishes the 153 landmarks themselves, the
family selection is hidden behind them, and this file is what keeps the three
declarations of that set (the catalog, `run()`'s `Literal`, and the layout's
tabs) from drifting apart.
"""

import inspect
import json
import os
import typing

import pytest

from sadt_ali_common import markups

from sadt_ali_ios import catalog, dispatch, layout, run
from sadt_ali_ios.errors import ToolInputError


def _choices(argument):
    """The `Literal` options `run()` publishes for one argument."""
    hint = typing.get_type_hints(run)[argument]
    if typing.get_origin(hint) is list:
        hint = typing.get_args(hint)[0]
    return list(typing.get_args(hint))


# ---------------------------------------------------------------------------
# The published options, the catalog, and the tabs cannot drift apart
# ---------------------------------------------------------------------------

def test_the_published_landmarks_are_the_catalogs_own():
    """`Literal` takes literals only, so it cannot be built from the catalog.
    That makes the signature a second declaration of the same 153 names, and
    this is what keeps the two honest: a landmark added to one and not the
    other would be unselectable from the client, or offered and then refused."""
    assert _choices("landmarks") == list(catalog.LANDMARKS)


def test_the_catalog_holds_the_landmarks_the_networks_actually_predict():
    """84 occlusal (14 teeth x 3 types x 2 jaws), 56 cervical, 13
    mucogingival -- the mandible alone, and one tooth short of 14."""
    assert len(catalog.LANDMARKS) == 153
    assert len(set(catalog.LANDMARKS)) == 153
    counts = {name: len(labels) for name, labels in catalog.LANDMARK_GROUPS.items()}
    assert counts == {
        "Occlusal Upper": 42,
        "Occlusal Lower": 42,
        "Cervical Upper": 28,
        "Cervical Lower": 28,
        "Mucogingival Lower": 13,
    }


def test_one_landmark_names_exactly_one_network():
    """`networks_for` reads this mapping to decide which passes to run, so a
    name appearing in two label tables would make a selection ambiguous."""
    assert set(catalog.LANDMARK_NETWORK) == set(catalog.LANDMARKS)
    assert catalog.LANDMARK_NETWORK["UR1O"] == "O"
    assert catalog.LANDMARK_NETWORK["UR1CL"] == "C"
    assert catalog.LANDMARK_NETWORK["L0MG"] == "MG"


def test_every_published_landmark_is_reachable_through_a_tab():
    """The failure the old hand-written tables produced: a landmark the schema
    offered and no tab could reach, so it was invisible from the client."""
    groups = layout.LAYOUT["landmarks"]["groups"]
    covered = [label for labels in groups.values() for label in labels]

    assert sorted(covered) == sorted(_choices("landmarks"))
    # And no landmark in two tabs, which would render it twice and let the two
    # check boxes disagree.
    assert len(covered) == len(set(covered))


def test_mucogingival_has_no_upper_tab():
    """It was trained on the mandible alone, so a maxillary tab is not an empty
    tab -- it is a question the network cannot be asked. The group is dropped
    rather than shown with nothing in it."""
    groups = layout.LAYOUT["landmarks"]["groups"]

    assert catalog.NETWORK_JAWS["MG"] == ("Lower",)
    assert "Mucogingival Lower" in groups
    assert [name for name in groups if name.startswith("Mucogingival")] == [
        "Mucogingival Lower"
    ]


def test_no_landmark_name_is_written_by_hand_in_the_layout():
    """The tabs are computed from the catalog. Spelling one name here would be
    the start of the drift the derivation exists to prevent."""
    source = open(layout.__file__, encoding="utf-8").read()
    assert [label for label in catalog.LANDMARKS if '"{}"'.format(label) in source] == []


def test_the_family_selection_is_no_longer_rendered():
    """What the landmark table replaces. It stays in `run()`: AREG asks this
    tool for the mucogingival pass alone, by family, and never sees a panel."""
    assert layout.LAYOUT["networks"]["hidden"] is True
    assert inspect.signature(run).parameters["networks"].default == [
        "Occlusal", "Cervical"
    ]
    assert inspect.signature(run).parameters["landmarks"].default == []


# ---------------------------------------------------------------------------
# A selection decides which networks run
# ---------------------------------------------------------------------------

def test_a_landmark_selection_implies_the_networks_that_produce_it():
    """This is the half of the selection that is not cosmetic: asking for the
    13 mucogingival points runs the MG pass alone, instead of the occlusal and
    cervical passes the default would have run over every mesh."""
    assert catalog.networks_for(["L0MG"]) == ("MG",)
    assert catalog.networks_for(["UR1O", "UR1CL"]) == ("O", "C")
    # Declaration order, not the caller's, so the report reads the same way
    # however the request was assembled.
    assert catalog.networks_for(["L0MG", "UR1CL", "UR1O"]) == ("O", "C", "MG")
    assert catalog.networks_for([]) == ()


def test_recognised_landmarks_come_back_in_declaration_order():
    recognised, unknown = catalog.resolve_landmarks(["L0MG", "UR1O"])
    assert recognised == ("UR1O", "L0MG")
    assert unknown == ()


def test_a_name_the_catalog_does_not_know_is_reported_not_dropped():
    """A network emits a fixed set of channels: unlike ALI_CBCT, there is no
    bundle-provided extra that could make an unknown name runnable. It is
    carried out so the run report can say what the selection did not buy."""
    recognised, unknown = catalog.resolve_landmarks(["UR1O", "UR3RIP"])
    assert recognised == ("UR1O",)
    assert unknown == ("UR3RIP",)


def test_an_empty_selection_is_not_a_selection():
    assert catalog.resolve_landmarks([]) == ((), ())
    assert catalog.resolve_landmarks(None) == ((), ())


# ---------------------------------------------------------------------------
# What a run actually writes
# ---------------------------------------------------------------------------

# Everything one occlusal pass and one mucogingival pass emit for two teeth.
# The two MG names are the shifted pair: L0MG is the midline, on tooth 25, and
# LR1MG sits on tooth 26 -- so a filter deriving `<tooth><type>` would keep the
# wrong one of them.
PREDICTED = {
    "UR1O": (1.0, 0.0, 0.0),
    "UR1MB": (2.0, 0.0, 0.0),
    "UR1DB": (3.0, 0.0, 0.0),
    "L0MG": (4.0, 0.0, 0.0),
    "LR1MG": (5.0, 0.0, 0.0),
}

DEGRADED = {"L0MG": "cameras aimed from an arch fit, tooth not segmented"}


@pytest.fixture
def stub_engine(monkeypatch):
    """Stand in for `engine.predict_landmarks`, which needs pytorch3d.

    It writes what a real pass would write -- every channel of every network it
    was asked for -- through the same writer the engine uses, which is exactly
    what the filtering has to work on.
    """
    from sadt_ali_ios import engine

    calls = {}

    def predict_landmarks(meshes, model_path, networks, prediction_ID, output_dir,
                          device):
        calls["networks"] = tuple(networks)
        scans = {}
        for mesh_path, key in meshes:
            destination = os.path.join(
                output_dir,
                os.path.dirname(key),
                "{}_lm_{}{}".format(
                    os.path.splitext(os.path.basename(mesh_path))[0],
                    prediction_ID,
                    markups.MARKUPS_EXTENSION,
                ),
            )
            markups.write(PREDICTED, destination, DEGRADED)
            scans[key] = {
                "input": os.path.basename(mesh_path),
                "status": "ok",
                "landmarks_found": sorted(PREDICTED),
                "landmarks_failed": {},
                "landmarks_degraded": dict(DEGRADED),
                "jaws_without_model": {},
                "produced": [destination],
            }
        return {
            "mode": "IOS",
            "networks": list(networks),
            "cases": scans,
            "summary": {"total": len(scans), "processed": len(scans), "failed": 0},
        }

    monkeypatch.setattr(engine, "predict_landmarks", predict_landmarks)
    return calls


def _mesh(tmp_path):
    """A real labelled polydata, small enough to write by hand.

    It used to be a one-line header, on the grounds that its contents never
    reached the stubbed engine. They do now: `dispatch` reads every mesh's
    point-data array NAMES to tell the ones that need `Crown_Seg` from the ones
    that do not, and that read happens before the engine is called at all. A
    fixture only valid because everything reading it was stubbed is a fixture
    that stops testing the moment the code stops stubbing.
    """
    path = tmp_path / "in" / "arch.vtk"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# vtk DataFile Version 3.0\n"
        "labelled arch\n"
        "ASCII\n"
        "DATASET POLYDATA\n"
        "POINTS 3 float\n"
        "0 0 0  1 0 0  0 1 0\n"
        "POLYGONS 1 4\n"
        "3 0 1 2\n"
        "POINT_DATA 3\n"
        "SCALARS PredictedID int 1\n"
        "LOOKUP_TABLE default\n"
        "2 2 2\n",
        encoding="utf-8",
    )
    return str(path)


def _written_labels(output_dir):
    path = os.path.join(output_dir, "arch_lm_Pred.mrk.json")
    with open(path, encoding="utf-8") as handle:
        content = json.load(handle)
    return [point["label"] for point in content["markups"][0]["controlPoints"]]


def test_a_selection_filters_what_is_written_not_what_is_computed(tmp_path, stub_engine):
    """One forward pass emits every channel of its network, so UR1O arrives
    with UR1MB and UR1DB whether or not they were asked for. What the selection
    buys is whole passes -- the cervical one never runs here -- and a file
    holding the points the caller asked for and nothing else."""
    output_dir = str(tmp_path / "out")
    report = dispatch.identify(
        input_path=_mesh(tmp_path),
        model_path=str(tmp_path / "bundle"),
        output_dir=output_dir,
        landmarks=["UR1O", "L0MG"],
    )

    assert stub_engine["networks"] == ("O", "MG")
    assert _written_labels(output_dir) == ["UR1O", "L0MG"]
    assert report["landmarks_selected"] == ["UR1O", "L0MG"]
    assert report["cases"]["arch.vtk"]["landmarks_found"] == ["L0MG", "UR1O"]


def test_the_shifted_mucogingival_label_survives_the_filter(tmp_path, stub_engine):
    """L0MG is the midline name, carried by tooth 25, which shifts the right
    side by one: LR1MG sits on tooth 26. A filter rebuilding `<tooth><type>`
    from the tooth number would keep LR1MG here and drop the point asked
    for."""
    output_dir = str(tmp_path / "out")
    dispatch.identify(
        input_path=_mesh(tmp_path),
        model_path=str(tmp_path / "bundle"),
        output_dir=output_dir,
        landmarks=["L0MG"],
    )

    assert _written_labels(output_dir) == ["L0MG"]


def test_a_kept_landmark_keeps_the_caveat_it_was_written_with(tmp_path, stub_engine):
    """A degraded point looks exactly like a good one; dropping the neighbours
    it was written beside must not quietly make it look clean."""
    output_dir = str(tmp_path / "out")
    report = dispatch.identify(
        input_path=_mesh(tmp_path),
        model_path=str(tmp_path / "bundle"),
        output_dir=output_dir,
        landmarks=["L0MG"],
    )

    path = os.path.join(output_dir, "arch_lm_Pred.mrk.json")
    with open(path, encoding="utf-8") as handle:
        point = json.load(handle)["markups"][0]["controlPoints"][0]
    assert point["description"] == DEGRADED["L0MG"]
    assert report["cases"]["arch.vtk"]["landmarks_degraded"] == DEGRADED


def test_a_scan_left_with_nothing_loses_its_file_rather_than_gaining_an_empty_one(
    tmp_path, stub_engine
):
    """An upper-arch selection against a mandible. An empty markups file opens
    in Slicer as an empty node, which reads as a run that went wrong."""
    output_dir = str(tmp_path / "out")
    report = dispatch.identify(
        input_path=_mesh(tmp_path),
        model_path=str(tmp_path / "bundle"),
        output_dir=output_dir,
        landmarks=["UL7CL"],
    )

    assert report["cases"]["arch.vtk"]["produced"] == []
    assert report["cases"]["arch.vtk"]["landmarks_found"] == []
    assert not os.path.exists(os.path.join(output_dir, "arch_lm_Pred.mrk.json"))


def test_an_empty_selection_changes_nothing(tmp_path, stub_engine):
    """The ordinary case, and what every request written before this argument
    existed relies on: the families decide and nothing is filtered."""
    output_dir = str(tmp_path / "out")
    report = dispatch.identify(
        input_path=_mesh(tmp_path),
        model_path=str(tmp_path / "bundle"),
        output_dir=output_dir,
        ios_networks=["Occlusal"],
        landmarks=[],
    )

    assert stub_engine["networks"] == ("O",)
    assert _written_labels(output_dir) == list(PREDICTED)
    assert report["landmarks_selected"] == []
    assert "landmarks_unknown" not in report


def test_naming_landmarks_replaces_the_family_selection(tmp_path, stub_engine):
    """Not narrows it: a caller naming the points it needs must not also have
    to leave the right families ticked. Mucogingival is off by default, and
    asking for L0MG alone still runs it."""
    dispatch.identify(
        input_path=_mesh(tmp_path),
        model_path=str(tmp_path / "bundle"),
        output_dir=str(tmp_path / "out"),
        ios_networks=["Occlusal", "Cervical"],
        landmarks=["L0MG"],
    )

    assert stub_engine["networks"] == ("MG",)


def test_a_selection_the_catalog_half_knows_runs_and_says_what_it_dropped(
    tmp_path, stub_engine
):
    output_dir = str(tmp_path / "out")
    report = dispatch.identify(
        input_path=_mesh(tmp_path),
        model_path=str(tmp_path / "bundle"),
        output_dir=output_dir,
        landmarks=["UR1O", "UR3RIP"],
    )

    assert _written_labels(output_dir) == ["UR1O"]
    assert report["landmarks_unknown"] == ["UR3RIP"]


def test_a_selection_naming_nothing_that_exists_is_refused(tmp_path, stub_engine):
    """A 422, not a run of everything: silently ignoring the selection would
    return 153 landmarks to a caller that asked for one."""
    with pytest.raises(ToolInputError) as raised:
        dispatch.identify(
            input_path=_mesh(tmp_path),
            model_path=str(tmp_path / "bundle"),
            output_dir=str(tmp_path / "out"),
            landmarks=["UR3RIP"],
        )

    assert "UR3RIP" in str(raised.value)


def test_every_landmark_says_what_it_is():
    """A landmark is a code. `UR1MB` names no anatomy a clinician can read off
    it, so the schema publishes one line per option."""
    from sadt_ali_ios import catalog

    assert set(catalog.DESCRIPTIONS) == set(catalog.LANDMARKS)
    assert all(text.strip() for text in catalog.DESCRIPTIONS.values())


def test_a_description_names_the_tooth_the_ENGINE_uses_not_the_one_the_name_suggests():
    """The trap this exists to avoid. Mucogingival names are assigned
    POSITIONALLY and the midline name shifts the right side by one, so `LR1MG`
    sits on tooth 26 -- reading it as "LR1" would put it on 25 and name the
    wrong tooth in a tooltip a clinician is about to trust."""
    from sadt_ali_ios import catalog

    assert catalog.DESCRIPTIONS["LR1MG"].startswith("lower right lateral incisor")
    assert "universal 26" in catalog.DESCRIPTIONS["LR1MG"]
    # The midline name itself, which no quadrant spelling could produce.
    assert catalog.DESCRIPTIONS["L0MG"].startswith("lower right central incisor")


def test_the_words_travel_with_the_argument():
    """Declared by the tool, published by the schema: a landmark gains its line
    with no client release."""
    from sadt_ali_ios import catalog, layout

    assert layout.LAYOUT["landmarks"]["option_help"] is catalog.DESCRIPTIONS
