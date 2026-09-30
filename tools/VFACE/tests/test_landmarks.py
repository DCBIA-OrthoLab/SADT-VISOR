"""Whose landmarks these are, and how they move between the two frames.

All of it is VFACE's own: the padding that lets the search reach a landmark at
the border, the rigid derivation that saves searching a second time, and the
naming rule that says which file belongs to which patient. Nothing here needs
a supervisor, a card or a checkpoint.
"""

import json
import os

import numpy as np
import pytest

from conftest import read_volume, tree_of, write_volume
from sadt_vface import landmarks
from sadt_vface.errors import ToolInputError


ARCH = {
    "Ba": [0.0, -30.0, -20.0], "S": [0.0, -10.0, 0.0], "N": [0.0, 40.0, 10.0],
    "RPo": [-35.0, -25.0, -12.0], "LPo": [35.0, -25.0, -12.0],
    "ANS": [0.0, 45.0, -25.0], "Me": [0.0, 25.0, -70.0],
}


def write_transform(path, matrix=None, translation=(0.0, 0.0, 0.0)):
    """An affine `.tfm`, the shape ASO writes beside an oriented scan."""
    import SimpleITK as sitk

    transform = sitk.AffineTransform(3)
    if matrix is not None:
        transform.SetMatrix([float(v) for v in np.asarray(matrix).reshape(9)])
    transform.SetTranslation([float(v) for v in translation])
    os.makedirs(os.path.dirname(str(path)) or ".", exist_ok=True)
    sitk.WriteTransform(transform, str(path))
    return str(path)


def rotation_about_z(angle):
    return np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])


# ---------------------------------------------------------------------------
# patient_of
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(("name", "patient"), [
    ("P_0001_T1_CB_Or_lm_Pred_CB.mrk.json", "P_0001"),
    ("P_0001_T1_CB_Or.nii.gz", "P_0001"),
    ("P_0001_T1_MAX_Or.tfm", "P_0001"),
    ("P_0001_T1_CB_Or_mir_CB_reg.mrk.json", "P_0001"),
    ("Dupont_003_T1_Scan.nrrd", "Dupont_003"),
])
def test_every_step_of_the_pipeline_leaves_the_patient_readable(name, patient):
    """This is the whole contract by which a landmark file finds its scan and a
    scan finds the transform that oriented it. Each step appends what it did,
    and the identifier is everything before the first of those."""
    assert landmarks.patient_of(name) == patient


def test_a_decoration_is_a_whole_token_not_a_substring():
    """Upstream cuts with `basename.split("_CB")[0]`, which also cuts
    `P1_CBrown` -- the same substring defect this repository has fixed three
    times over."""
    assert landmarks.patient_of("P1_CBrown_T1.nii.gz") == "P1_CBrown"
    assert landmarks.patient_of("P1_Origami_T1.nii.gz") == "P1_Origami"
    assert landmarks.patient_of("P1_CB_T1.nii.gz") == "P1"


def test_a_name_that_is_all_decoration_keeps_its_stem():
    """Rather than reducing to the empty string, which would merge every such
    file into one patient."""
    assert landmarks.patient_of("CB_Or.nii.gz") == "CB_Or"


# ---------------------------------------------------------------------------
# Reading and writing
# ---------------------------------------------------------------------------

def test_one_patients_groups_are_merged_into_one_set(tmp_path):
    """ALI writes one file per group, so a patient's cranial base and lower
    points arrive separately -- and a measurement needs both."""
    landmarks.write_markups({"Ba": ARCH["Ba"], "S": ARCH["S"]},
                            str(tmp_path / "lm" / "P1_T1_lm_Pred_CB.mrk.json"))
    landmarks.write_markups({"Me": ARCH["Me"]},
                            str(tmp_path / "lm" / "P1_T1_lm_Pred_L.mrk.json"))

    found = landmarks.read_cohort(str(tmp_path / "lm"))
    assert set(found) == {"P1"}
    assert set(found["P1"]) == {"Ba", "S", "Me"}


def test_two_patients_are_not_merged(tmp_path):
    landmarks.write_markups({"Ba": ARCH["Ba"]}, str(tmp_path / "lm" / "P1_T1_lm_Pred_CB.mrk.json"))
    landmarks.write_markups({"S": ARCH["S"]}, str(tmp_path / "lm" / "P2_T1_lm_Pred_CB.mrk.json"))

    found = landmarks.read_cohort(str(tmp_path / "lm"))
    assert sorted(found) == ["P1", "P2"]


def test_the_positions_survive_a_round_trip(tmp_path):
    path = landmarks.write_markups(ARCH, str(tmp_path / "P1_lm_Pred_CB.mrk.json"))
    assert landmarks.read_markups(path) == {
        label: [float(v) for v in position] for label, position in ARCH.items()
    }


def test_what_is_written_actually_draws_when_it_is_opened(tmp_path):
    """A markups file written with `visibility` False loads into Slicer, builds
    the node and draws NOTHING. It was invisible in both of ALI's engines until
    somebody opened a result outside the module."""
    path = landmarks.write_markups(ARCH, str(tmp_path / "P1_lm_Pred_CB.mrk.json"))
    with open(path, encoding="utf-8") as handle:
        document = json.load(handle)
    control_points = document["markups"][0]["controlPoints"]
    assert control_points
    assert all(point["visibility"] is True for point in control_points)


def test_a_file_that_is_not_markups_does_not_cost_the_cohort(tmp_path):
    landmarks.write_markups({"Ba": ARCH["Ba"]}, str(tmp_path / "lm" / "P1_lm_Pred_CB.mrk.json"))
    (tmp_path / "lm" / "P2_lm_Pred_CB.mrk.json").write_text("{not json")

    assert sorted(landmarks.read_cohort(str(tmp_path / "lm"))) == ["P1"]


def test_an_empty_folder_reads_as_no_patients(tmp_path):
    (tmp_path / "lm").mkdir()
    assert landmarks.read_cohort(str(tmp_path / "lm")) == {}


# ---------------------------------------------------------------------------
# The groups
# ---------------------------------------------------------------------------

def test_a_landmark_is_filed_under_the_group_ali_files_it_under():
    assert landmarks.group_of("Ba") == "CB"
    assert landmarks.group_of("ANS") == "U"
    assert landmarks.group_of("Me") == "L"


def test_a_landmark_no_group_names_still_gets_one():
    """Rather than a KeyError, or a file named after nothing."""
    assert landmarks.group_of("NotALandmark") == "U"


def test_no_landmark_is_filed_under_two_groups():
    """The table is inverted to answer `group_of`, and a label in two groups
    would answer whichever came last."""
    seen = [label for labels in landmarks.GROUP_LABELS.values() for label in labels]
    assert len(seen) == len(set(seen))


# ---------------------------------------------------------------------------
# Padding
# ---------------------------------------------------------------------------

def test_a_padded_scan_is_larger_and_sits_in_the_same_place(tmp_path):
    """Physical coordinates are preserved -- the origin moves with the padding
    -- so the landmarks come back in the original scan's space."""
    write_volume(tmp_path / "in" / "P1_T1_CB_Or.nii.gz", size=(24, 20, 16),
                 spacing=(0.5, 0.5, 0.5))
    before = read_volume(tmp_path / "in" / "P1_T1_CB_Or.nii.gz")

    landmarks.pad_scans(str(tmp_path / "in"), str(tmp_path / "out"), margin_mm=5)
    after = read_volume(tmp_path / "out" / "P1_T1_CB_Or.nii.gz")

    # 5 mm at 0.5 mm spacing is 10 voxels on each side.
    assert after.GetSize() == tuple(n + 20 for n in before.GetSize())
    assert after.GetOrigin() == pytest.approx(
        tuple(o - 5.0 for o in before.GetOrigin()), abs=1e-6
    )
    # The same physical point reads the same value in both.
    point = before.TransformIndexToPhysicalPoint((5, 5, 5))
    assert after.GetPixel(after.TransformPhysicalPointToIndex(point)) == pytest.approx(
        before.GetPixel((5, 5, 5))
    )


def test_the_padding_value_is_the_scans_own_minimum(tmp_path):
    """An air value that already exists in the volume keeps the intensity
    rescaling the landmark tool applies unchanged by the room it was given."""
    import SimpleITK as sitk

    write_volume(tmp_path / "in" / "P1_T1.nii.gz")
    source = sitk.GetArrayFromImage(read_volume(tmp_path / "in" / "P1_T1.nii.gz"))

    landmarks.pad_scans(str(tmp_path / "in"), str(tmp_path / "out"), margin_mm=5)
    padded = sitk.GetArrayFromImage(read_volume(tmp_path / "out" / "P1_T1.nii.gz"))

    assert float(padded[0, 0, 0]) == pytest.approx(float(source.min()), abs=1e-3)
    assert float(padded.min()) == pytest.approx(float(source.min()), abs=1e-3)


def test_a_scan_that_cannot_be_padded_is_copied_and_reported(tmp_path):
    """An unpadded scan is still worth landmarking; losing it is not."""
    write_volume(tmp_path / "in" / "P1_T1.nii.gz")
    (tmp_path / "in" / "P2_T1.nii.gz").write_bytes(b"\x1f\x8b not a volume")

    report = {}
    landmarks.pad_scans(str(tmp_path / "in"), str(tmp_path / "out"), report=report)

    assert tree_of(tmp_path / "out") == ["P1_T1.nii.gz", "P2_T1.nii.gz"]
    assert list(report["not_padded"]) == ["P2_T1.nii.gz"]


def test_nothing_to_pad_is_refused_by_name(tmp_path):
    (tmp_path / "in").mkdir()
    with pytest.raises(ToolInputError, match="No scan to give room to"):
        landmarks.pad_scans(str(tmp_path / "in"), str(tmp_path / "out"))


# ---------------------------------------------------------------------------
# Deriving one frame's landmarks from the other's
# ---------------------------------------------------------------------------

def _two_frames(tmp_path, angle=0.3, shift=(4.0, -2.0, 1.0)):
    """A patient oriented into two frames, and its landmarks in the first.

    Both transforms map the same centred scan into their own frame, so the
    landmarks in the second frame are `inv(target) . source` applied to the
    ones in the first -- which is exactly what the derivation computes, and
    what this fixture can therefore check against.
    """
    landmarks.write_markups(ARCH, str(tmp_path / "lm_cb" / "P1_T1_CB_Or_lm_Pred_CB.mrk.json"))
    write_transform(tmp_path / "cb" / "P1_T1_CB_Or.tfm")
    write_transform(tmp_path / "max" / "P1_T1_MAX_Or.tfm",
                    matrix=rotation_about_z(angle), translation=shift)
    write_volume(tmp_path / "max" / "P1_T1_MAX_Or.nii.gz")
    return rotation_about_z(angle), np.asarray(shift, dtype=float)


def test_the_derived_position_is_the_one_the_two_transforms_imply(tmp_path):
    rotation, shift = _two_frames(tmp_path)

    landmarks.derive_into_frame(
        str(tmp_path / "lm_cb"), str(tmp_path / "cb"), str(tmp_path / "max"),
        str(tmp_path / "max"), ["Ba", "S", "ANS"], str(tmp_path / "lm_max"),
    )
    derived = landmarks.read_cohort(str(tmp_path / "lm_max"))["P1"]

    for label in ("Ba", "S", "ANS"):
        # The source transform is the identity here, so the answer is the
        # target transform's inverse applied to the original point.
        expected = np.linalg.inv(rotation) @ (np.asarray(ARCH[label]) - shift)
        assert derived[label] == pytest.approx(expected, abs=1e-6)


def test_only_the_landmarks_asked_for_are_written(tmp_path):
    """The maxillary frame wants the occlusal points, not the whole catalogue."""
    _two_frames(tmp_path)
    landmarks.derive_into_frame(
        str(tmp_path / "lm_cb"), str(tmp_path / "cb"), str(tmp_path / "max"),
        str(tmp_path / "max"), ["ANS"], str(tmp_path / "lm_max"),
    )
    assert set(landmarks.read_cohort(str(tmp_path / "lm_max"))["P1"]) == {"ANS"}


def test_the_derived_files_are_named_after_the_target_frames_scan(tmp_path):
    """So the next step pairs them with the scan they belong to, by the same
    rule everything else pairs by."""
    _two_frames(tmp_path)
    landmarks.derive_into_frame(
        str(tmp_path / "lm_cb"), str(tmp_path / "cb"), str(tmp_path / "max"),
        str(tmp_path / "max"), ["Ba", "ANS", "Me"], str(tmp_path / "lm_max"),
    )
    written = tree_of(tmp_path / "lm_max")
    assert all(name.startswith("P1_T1_MAX_Or_lm_Pred_") for name in written)
    # One file per group, as ALI writes them.
    assert sorted(written) == [
        "P1_T1_MAX_Or_lm_Pred_CB.mrk.json",
        "P1_T1_MAX_Or_lm_Pred_L.mrk.json",
        "P1_T1_MAX_Or_lm_Pred_U.mrk.json",
    ]


def test_a_patient_missing_a_transform_is_reported_and_the_cohort_goes_on(tmp_path):
    _two_frames(tmp_path)
    landmarks.write_markups(ARCH, str(tmp_path / "lm_cb" / "P2_T1_CB_Or_lm_Pred_CB.mrk.json"))

    report = {}
    landmarks.derive_into_frame(
        str(tmp_path / "lm_cb"), str(tmp_path / "cb"), str(tmp_path / "max"),
        str(tmp_path / "max"), ["Ba"], str(tmp_path / "lm_max"), report=report,
    )
    assert "P2" in report["not_derived"]
    assert landmarks.read_cohort(str(tmp_path / "lm_max"))["P1"]


def test_nothing_to_derive_from_is_refused_by_name(tmp_path):
    (tmp_path / "lm_cb").mkdir()
    with pytest.raises(ToolInputError, match="No landmark file to derive from"):
        landmarks.derive_into_frame(
            str(tmp_path / "lm_cb"), str(tmp_path / "cb"), str(tmp_path / "max"),
            str(tmp_path / "max"), ["Ba"], str(tmp_path / "lm_max"),
        )


def test_a_cohort_where_no_patient_could_be_derived_is_refused(tmp_path):
    """The guard counts what was WRITTEN. One counting the patients it walked
    past would pass on a cohort where not one of them could be derived."""
    landmarks.write_markups(ARCH, str(tmp_path / "lm_cb" / "P1_T1_CB_Or_lm_Pred_CB.mrk.json"))
    (tmp_path / "cb").mkdir()
    (tmp_path / "max").mkdir()

    with pytest.raises(ToolInputError, match="No patient's landmarks could be"):
        landmarks.derive_into_frame(
            str(tmp_path / "lm_cb"), str(tmp_path / "cb"), str(tmp_path / "max"),
            str(tmp_path / "max"), ["Ba"], str(tmp_path / "lm_max"),
        )
