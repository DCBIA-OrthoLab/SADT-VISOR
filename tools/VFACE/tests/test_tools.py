"""The six calls VFACE makes, and the parameters each one sends.

Asserted on a fake supervisor, which is all a tool can ever see of a real one.
Nothing is imported across virtualenvs here, any more than it is in the server:
`sup.run("ASO", ...)` starts that tool in its own interpreter and blocks.

What these tests pin is the SEAM -- that each call names the tool it means and
sends the arguments that tool publishes. `test_schema_seam.py` is the other
half: it reads the six real schemas out of process and checks every argument
sent exists.
"""

import os

import pytest

from conftest import FakeSup
from sadt_vface import catalogs, tools
from sadt_vface.errors import SupervisorRequired


@pytest.fixture
def sup(tmp_path):
    return FakeSup(tmp_path, {
        name: (lambda params: params["output_dir"])
        for name in ("ASO", "AMASSS", "AutoMatrix", "AREG_CBCT", "ALI_CBCT",
                     "Batch_Dental_Seg")
    })


# ---------------------------------------------------------------------------
# Refusing at the door
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tool", [
    "ASO", "AMASSS", "AutoMatrix", "AREG_CBCT", "ALI_CBCT", "Batch_Dental_Seg",
])
def test_a_mode_that_needs_a_tool_refuses_without_a_supervisor(tool):
    with pytest.raises(SupervisorRequired) as raised:
        tools.require(None, tool, "This mode")
    assert tool in str(raised.value)


@pytest.mark.parametrize("tool", list(tools._ADVICE))
def test_the_refusal_names_a_mode_that_works_instead(tool):
    """"Deploy a tool" is not an answer a clinician can act on. Every one of
    these has a mode that does not need it, and the refusal says which."""
    with pytest.raises(SupervisorRequired) as raised:
        tools.require(None, tool, "This mode")
    assert tools._ADVICE[tool] in str(raised.value)


def test_every_tool_called_carries_advice():
    """Derived from the source, not listed: a call added without advice would
    refuse by naming a deployment problem and nothing else."""
    import re

    called = set(re.findall(r'sup\.run\(\s*"([A-Za-z_]+)"', open(
        os.path.join(os.path.dirname(tools.__file__), "tools.py")
    ).read()))
    assert called, "no sup.run call found -- has the call graph moved?"
    assert called <= set(tools._ADVICE)


def test_a_supervisor_that_is_there_is_not_refused(sup):
    tools.require(sup, "ASO", "This mode")


# ---------------------------------------------------------------------------
# ASO -- the orientation, once per frame
# ---------------------------------------------------------------------------

def test_the_orientation_is_fully_automated_on_cbct(sup, tmp_path):
    frame = catalogs.FRAMES[catalogs.FRAME_CRANIAL_BASE]
    tools.orient_scans(
        sup, str(tmp_path / "scans"), str(tmp_path / "ref"),
        frame["landmarks"], frame["suffix"], str(tmp_path / "bundle"),
        label=catalogs.FRAME_CRANIAL_BASE,
    )
    asked = sup.asked("ASO")
    assert asked["modality"] == "CBCT"
    assert asked["automation"] == "Fully-Automated"
    assert asked["cbct_landmarks"] == ["Ba", "LPo", "N", "RPo", "S", "LOr", "ROr"]
    assert asked["output_suffix"] == "CB_Or"
    assert asked["landmark_model"] == str(tmp_path / "bundle")


def test_each_frame_is_fitted_on_its_own_landmarks(sup, tmp_path):
    """The cranial base frame is the Frankfort horizontal and the mid-sagittal
    plane; the maxillary frame is the occlusal and mid-sagittal plane. Each
    reference is built on its own set of points, and sending one frame's
    landmarks against the other's reference orients the scan onto nothing."""
    for frame_name, frame in catalogs.FRAMES.items():
        tools.orient_scans(
            sup, str(tmp_path / "scans"), str(tmp_path / frame["reference"]),
            frame["landmarks"], frame["suffix"], label=frame_name,
        )
    sent = [params for name, params in sup.calls if name == "ASO"]
    assert [params["cbct_landmarks"] for params in sent] == [
        catalogs.FRAMES[name]["landmarks"] for name in catalogs.FRAMES
    ]
    assert len({params["reference"] for params in sent}) == len(catalogs.FRAMES)


def test_the_two_orientations_do_not_write_into_one_folder(sup, tmp_path):
    """Two calls to the same tool. Sharing an output directory leaves the
    second reading the first one's scans as its own, and a cohort then comes
    back oriented into the wrong frame with nothing raised."""
    for frame_name, frame in catalogs.FRAMES.items():
        tools.orient_scans(sup, str(tmp_path / "scans"), str(tmp_path / "ref"),
                           frame["landmarks"], frame["suffix"], label=frame_name)
    sent = [params["output_dir"] for name, params in sup.calls if name == "ASO"]
    assert len(set(sent)) == len(sent)


def test_an_orientation_with_no_bundle_named_does_not_send_an_empty_one(sup, tmp_path):
    """ASO resolves nothing itself. An empty path is not "use the default", it
    is a path that does not exist."""
    frame = catalogs.FRAMES[catalogs.FRAME_MAXILLA]
    tools.orient_scans(sup, str(tmp_path / "scans"), str(tmp_path / "ref"),
                       frame["landmarks"], frame["suffix"])
    assert "landmark_model" not in sup.asked("ASO")


# ---------------------------------------------------------------------------
# AMASSS -- the masks
# ---------------------------------------------------------------------------

def test_the_masks_are_asked_for_one_file_per_structure(sup, tmp_path):
    """A region's mask is looked up by name. A merged multi-label volume makes
    every region resolve to the same file."""
    tools.segment_masks(sup, str(tmp_path / "oriented"), str(tmp_path / "bundle"),
                        ["CB", "MAND"], label="CB")
    asked = sup.asked("AMASSS")
    assert asked["merge"] == ["SEPARATE"]
    assert asked["structures"] == ["CB", "MAND"]
    assert asked["generate_surface"] is False


def test_the_structures_of_a_frame_are_asked_for_in_one_call(sup, tmp_path):
    """AMASSS loads a network per structure and a cohort per call, so both of
    the cranial base frame's structures in one call is one pass over the scans
    rather than two."""
    structures = catalogs.structures_for(catalogs.FRAME_CRANIAL_BASE, catalogs.REGIONS)
    tools.segment_masks(sup, str(tmp_path / "oriented"), str(tmp_path / "bundle"),
                        structures, label="CB")
    assert sup.asked("AMASSS")["structures"] == ["CB", "MAND"]


# ---------------------------------------------------------------------------
# AutoMatrix -- the mirror
# ---------------------------------------------------------------------------

def test_the_mirror_uses_one_transform_for_every_patient(sup, tmp_path):
    """It is a reflection of the frame, not something fitted per patient."""
    tools.mirror(sup, str(tmp_path / "oriented"), str(tmp_path / "mirror.tfm"))
    asked = sup.asked("AutoMatrix")
    assert asked["same_transform_for_every_patient"] is True
    assert asked["output_suffix"] == "mir"
    assert asked["transforms"] == str(tmp_path / "mirror.tfm")


def test_a_mask_is_mirrored_as_a_segmentation_not_as_a_scan(sup, tmp_path):
    """Interpolating a label map linearly invents labels that are neither of
    the two it sits between."""
    tools.mirror(sup, str(tmp_path / "masks"), str(tmp_path / "mirror.tfm"),
                 content="Segmentation", label="masks")
    assert sup.asked("AutoMatrix")["content"] == "Segmentation"


# ---------------------------------------------------------------------------
# AREG_CBCT -- the registration
# ---------------------------------------------------------------------------

def test_the_registration_is_semi_automated_on_masks_vface_already_has(sup, tmp_path):
    """Asking AREG to segment and orient again would redo two steps and --
    worse -- orient the mirror independently of the original. A mirror is only
    a mirror while it shares the original's frame."""
    tools.register(sup, str(tmp_path / "t1"), str(tmp_path / "t2"),
                   catalogs.REGION_MANDIBLE, str(tmp_path / "masks"))
    asked = sup.asked("AREG_CBCT")
    assert asked["automation"] == "Semi-Automated"
    assert asked["regions"] == ["Mandible"]
    assert asked["t1_masks"] == str(tmp_path / "masks")


def test_each_region_is_registered_into_its_own_folder(sup, tmp_path):
    for region in catalogs.REGIONS:
        tools.register(sup, str(tmp_path / "t1"), str(tmp_path / "t2"),
                       catalogs.REGION_TABLE[region]["areg"], str(tmp_path / "masks"))
    sent = [params["output_dir"] for name, params in sup.calls if name == "AREG_CBCT"]
    assert len(set(sent)) == len(catalogs.REGIONS)


def test_the_region_names_sent_are_the_ones_areg_publishes():
    """`REGION_TABLE` holds VFACE's names on one side and AREG's on the other,
    and they are not the same vocabulary. The `areg` column is what travels."""
    assert sorted(entry["areg"] for entry in catalogs.REGION_TABLE.values()) == [
        "Cranial base", "Mandible", "Maxilla"
    ]


# ---------------------------------------------------------------------------
# ALI_CBCT -- the landmarks
# ---------------------------------------------------------------------------

def test_the_landmarks_are_asked_for_by_name_not_by_region(sup, tmp_path):
    """ALI spawns one search agent per landmark, at about a minute each.
    Asking for a region's whole catalogue searches dozens of points no
    measurement reads."""
    tools.predict_landmarks(sup, str(tmp_path / "registered"), ["Ba", "S", "N"],
                            str(tmp_path / "bundle"))
    asked = sup.asked("ALI_CBCT")
    assert asked["landmarks"] == ["Ba", "S", "N"]
    assert asked["model"] == str(tmp_path / "bundle")


# ---------------------------------------------------------------------------
# Batch_Dental_Seg -- the surfaces
# ---------------------------------------------------------------------------

def test_the_surfaces_are_asked_for_with_the_bundle_named(sup, tmp_path):
    """`Batch_Dental_Seg.run` takes `model` with no default, so an omitted one
    is a TypeError one tool down rather than a refusal here."""
    tools.segment_surfaces(sup, str(tmp_path / "registered"), str(tmp_path / "bundle"))
    assert sup.asked("Batch_Dental_Seg")["model"] == str(tmp_path / "bundle")
