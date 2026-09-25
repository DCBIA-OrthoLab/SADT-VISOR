"""Which intraoral scan belongs with which CBCT.

This tool is NOT longitudinal: it takes one timepoint imaged two ways, so
there is no timepoint to strip and `pairing.pair()` is not what pairs it. The
two modalities are named by different conventions -- the meshes by the
scanner, the volumes by the acquisition -- and the digits are what they
genuinely share.

Getting this wrong does not fail: it registers one patient's mesh into another
patient's skull and reports "ok".
"""

import os

import pytest

from conftest import write_mesh, write_volume
from sadt_areg_ioscbct import pipeline
from sadt_areg_common.errors import ToolInputError


# ---------------------------------------------------------------------------
# patient_key
# ---------------------------------------------------------------------------

def test_the_two_naming_conventions_reduce_to_the_same_patient():
    """Upstream's own test set: `P001_T2_U.vtk` beside `P_0001_T2.nii.gz`."""
    assert pipeline.patient_key("P001_T2_U.vtk") == "1"
    assert pipeline.patient_key("P_0001_T2.nii.gz") == "1"


def test_a_landmark_file_keys_to_the_patient_its_scan_keys_to():
    """ALI appends `_lm_Pred` to the scan's own name, so the landmark file has
    to reduce to the same patient -- that is what lets a mesh find its own
    points instead of the next patient's."""
    assert pipeline.patient_key("P001_T2_U_lm_Pred.mrk.json") == "1"
    assert pipeline.patient_key("P_0001_T2_lm_Pred.mrk.json") == "1"


def test_the_timepoint_is_not_part_of_the_patient():
    """One timepoint imaged two ways: T1 and T2 are the same subject here, and
    a trailing `2` swallowed into the digits would make `P1_T2` patient 12."""
    assert pipeline.patient_key("P1_T1.nii.gz") == "1"
    assert pipeline.patient_key("P1_T2.nii.gz") == "1"
    assert pipeline.patient_key("P1_T0.nii.gz") == "1"


def test_a_jaw_run_together_with_the_timepoint_is_still_split():
    """`A2_UpperT1.vtk` is one token, `uppert1`, matching neither table. Left
    unsplit the timepoint stays in and the two timepoints become two
    patients."""
    assert pipeline.patient_key("P1_UpperT2.vtk") == "1"


def test_patient_1_and_patient_10_are_different_patients():
    """The collision this repository keeps finding: a substring test made
    patient 1 match patient 10."""
    assert pipeline.patient_key("P1_T2_U.vtk") != pipeline.patient_key("P10_T2_U.vtk")
    assert pipeline.patient_key("P10_T2_U.vtk") == "10"


def test_leading_zeros_do_not_make_a_second_patient():
    for name in ("P1.vtk", "P01.vtk", "P_0001.vtk", "0001_scan.nii.gz"):
        assert pipeline.patient_key(name) == "1", name


def test_a_name_with_no_digits_keeps_its_stem():
    """The published reference files are called `Upper_gold.vtk`. Reducing them
    to the empty string would make every one of them the same patient."""
    assert pipeline.patient_key("Upper_gold.vtk") == "Upper_gold"
    assert pipeline.patient_key("Lower_gold.vtk") == "Lower_gold"


def test_an_all_zero_identifier_survives_as_itself():
    """`lstrip("0")` on `000` leaves nothing, and an empty key merges every
    such patient into one."""
    assert pipeline.patient_key("P000_T2.nii.gz") == "000"


def test_the_compound_extension_is_split_as_one_unit():
    """`.split(".")[0]` truncated a name at its first dot, which is how
    `P1.2_scan.nii.gz` became patient `P1`."""
    assert pipeline.patient_key("P7.nii.gz") == pipeline.patient_key("P7.vtk") == "7"


# ---------------------------------------------------------------------------
# discover
# ---------------------------------------------------------------------------

def test_a_patient_present_in_both_modalities_is_paired(tmp_path):
    mesh = write_mesh(tmp_path / "ios" / "P001_T2_U.vtk")
    volume = write_volume(tmp_path / "cbct" / "P_0001_T2.nii.gz")

    paired, unpaired = pipeline.discover(str(tmp_path / "ios"), str(tmp_path / "cbct"))
    assert paired == {"1": {"ios": [mesh], "cbct": volume}}
    assert unpaired == {}


def test_both_arches_of_one_patient_are_one_patient(tmp_path):
    """Upper and lower are registered separately but belong to one subject, and
    they are listed in a stable order."""
    write_mesh(tmp_path / "ios" / "P001_T2_U.vtk")
    write_mesh(tmp_path / "ios" / "P001_T2_L.vtk")
    write_volume(tmp_path / "cbct" / "P_0001_T2.nii.gz")

    paired, _unpaired = pipeline.discover(str(tmp_path / "ios"), str(tmp_path / "cbct"))
    assert [os.path.basename(path) for path in paired["1"]["ios"]] == [
        "P001_T2_L.vtk", "P001_T2_U.vtk"
    ]


def test_a_patient_with_only_one_modality_is_reported_not_dropped(tmp_path):
    """A batch that registered half of what was sent and said nothing is the
    failure this repository keeps finding."""
    write_mesh(tmp_path / "ios" / "P001_T2_U.vtk")
    write_mesh(tmp_path / "ios" / "P002_T2_U.vtk")
    write_volume(tmp_path / "cbct" / "P_0001_T2.nii.gz")
    write_volume(tmp_path / "cbct" / "P_0003_T2.nii.gz")

    paired, unpaired = pipeline.discover(str(tmp_path / "ios"), str(tmp_path / "cbct"))
    assert set(paired) == {"1"}
    assert unpaired == {"2": "no CBCT", "3": "no intraoral scan"}


def test_no_pair_at_all_names_both_counts_and_what_was_left_over(tmp_path):
    write_mesh(tmp_path / "ios" / "alpha.vtk")
    write_volume(tmp_path / "cbct" / "P_0001_T2.nii.gz")

    with pytest.raises(ToolInputError) as raised:
        pipeline.discover(str(tmp_path / "ios"), str(tmp_path / "cbct"))
    message = str(raised.value)
    assert "1 intraoral key(s)" in message
    assert "1 CBCT key(s)" in message
    assert "no CBCT" in message


def test_discovery_is_recursive_on_both_sides(tmp_path):
    write_mesh(tmp_path / "ios" / "siteA" / "nested" / "P001_T2_U.vtk")
    write_volume(tmp_path / "cbct" / "siteB" / "P_0001_T2.nii.gz")

    paired, _unpaired = pipeline.discover(str(tmp_path / "ios"), str(tmp_path / "cbct"))
    assert set(paired) == {"1"}


def test_an_stl_is_an_intraoral_scan_too(tmp_path):
    """Advertised by the schema, so it has to be discovered: the trap `.stl`
    fell into upstream was being counted by the UI and globbed past by the CLI."""
    write_mesh(tmp_path / "ios" / "P001_T2_U.vtk")
    (tmp_path / "ios" / "P002_T2_U.stl").write_bytes(b"solid x\nendsolid x\n")
    write_volume(tmp_path / "cbct" / "P_0001_T2.nii.gz")
    write_volume(tmp_path / "cbct" / "P_0002_T2.nii.gz")

    paired, _unpaired = pipeline.discover(str(tmp_path / "ios"), str(tmp_path / "cbct"))
    assert set(paired) == {"1", "2"}


@pytest.mark.parametrize(
    "name", ["P_0001.nii", "P_0001.nrrd", "P_0001.gipl", "P_0001.nii.gz",
             "P_0001.nrrd.gz", "P_0001.gipl.gz"],
)
def test_every_volume_spelling_counts_as_a_cbct(tmp_path, name):
    write_mesh(tmp_path / "ios" / "P001_T2_U.vtk")
    write_volume(tmp_path / "cbct" / name)
    paired, _unpaired = pipeline.discover(str(tmp_path / "ios"), str(tmp_path / "cbct"))
    assert set(paired) == {"1"}


def test_files_of_neither_kind_are_ignored(tmp_path):
    write_mesh(tmp_path / "ios" / "P001_T2_U.vtk")
    write_volume(tmp_path / "cbct" / "P_0001_T2.nii.gz")
    (tmp_path / "ios" / "notes.txt").write_text("hello")
    (tmp_path / "cbct" / "P_0002_T2.dcm").write_bytes(b"x")

    paired, unpaired = pipeline.discover(str(tmp_path / "ios"), str(tmp_path / "cbct"))
    assert set(paired) == {"1"}
    assert unpaired == {}


def test_a_patient_with_two_cbcts_keeps_the_first_in_sorted_order(tmp_path):
    """It used to keep whichever `os.walk` reached last, so which volume a
    patient was registered onto depended on the order the filesystem returned
    directories in -- the same request answering differently on two machines."""
    write_mesh(tmp_path / "ios" / "P001_T2_U.vtk")
    write_volume(tmp_path / "cbct" / "P_0001_T2_a.nii.gz")
    write_volume(tmp_path / "cbct" / "P_0001_T2_b.nii.gz")

    paired, _unpaired = pipeline.discover(str(tmp_path / "ios"), str(tmp_path / "cbct"))
    assert os.path.basename(paired["1"]["cbct"]) == "P_0001_T2_a.nii.gz"


def test_the_choice_does_not_depend_on_the_order_the_filesystem_returns(tmp_path):
    """Same two volumes, in two subdirectories this time. `os.walk` visits
    those in `os.listdir` order, which is arbitrary -- so the walk sorts them."""
    write_mesh(tmp_path / "ios" / "P001_T2_U.vtk")
    write_volume(tmp_path / "cbct" / "zeta" / "P_0001_T2.nii.gz")
    write_volume(tmp_path / "cbct" / "alpha" / "P_0001_T2.nii.gz")

    paired, _unpaired = pipeline.discover(str(tmp_path / "ios"), str(tmp_path / "cbct"))
    assert os.path.basename(os.path.dirname(paired["1"]["cbct"])) == "alpha"


def test_the_patients_come_back_in_a_stable_order(tmp_path):
    for patient in ("P003", "P001", "P002"):
        write_mesh(tmp_path / "ios" / f"{patient}_T2_U.vtk")
        write_volume(tmp_path / "cbct" / f"P_0{patient[1:]}_T2.nii.gz")

    paired, _unpaired = pipeline.discover(str(tmp_path / "ios"), str(tmp_path / "cbct"))
    assert list(paired) == ["1", "2", "3"]
