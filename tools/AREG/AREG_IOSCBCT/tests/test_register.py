"""A registration, end to end, in the mode that needs no other tool at all.

Registration mode takes both landmark sets and predicts nothing, so everything
below runs for real: the pairing, the matching, the fit, the mesh that comes
back out and the report that describes it. What the other two modes add is
where the landmarks came from, which `test_tools.py` covers.
"""

import os

import numpy as np
import pytest

from conftest import (
    LOWER_LABELS, UPPER_LABELS, cohort, moved, read_mesh_points, read_report,
    run_registration, tree_of, write_contourable_volume, write_markups,
    write_mesh, write_plane, write_volume,
)
from sadt_areg_ioscbct import dispatch, geometry, run
from sadt_areg_common.errors import ToolInputError


# ---------------------------------------------------------------------------
# What lands on disk
# ---------------------------------------------------------------------------

def test_a_run_writes_the_mesh_its_matrix_and_the_report(tmp_path):
    cohort(tmp_path)
    run_registration(tmp_path)
    assert tree_of(tmp_path / "out") == sorted([
        "AREG_report.json",
        os.path.join("1", "P001_T2_U_Reg.vtk"),
        os.path.join("1", "P001_T2_U_Reg_matrix.npy"),
    ])


def test_the_written_mesh_is_the_input_moved_by_the_written_matrix(tmp_path):
    """The matrix is the one thing a downstream tool reuses, so the two have to
    describe the same motion. A matrix written for a different space still
    loads and still transforms."""
    cohort(tmp_path)
    run_registration(tmp_path)

    before = read_mesh_points(tmp_path / "ios" / "P001_T2_U.vtk")
    after = read_mesh_points(tmp_path / "out" / "1" / "P001_T2_U_Reg.vtk")
    matrix = np.load(str(tmp_path / "out" / "1" / "P001_T2_U_Reg_matrix.npy"))

    assert geometry.apply(before, matrix) == pytest.approx(after, abs=1e-4)


def test_the_matrix_is_the_one_the_landmarks_asked_for(tmp_path):
    """The intraoral landmarks are `UPPER_LABELS` and the CBCT's are those
    shifted by a known vector, so the fit is a pure translation by it."""
    cohort(tmp_path)
    run_registration(tmp_path)

    matrix = np.load(str(tmp_path / "out" / "1" / "P001_T2_U_Reg_matrix.npy"))
    assert matrix[:3, :3] == pytest.approx(np.eye(3), abs=1e-6)
    assert matrix[:3, 3] == pytest.approx([3.0, -2.0, 1.0], abs=1e-6)


def test_each_patient_gets_a_folder_of_its_own(tmp_path):
    cohort(tmp_path, patients=("P001", "P002"))
    run_registration(tmp_path)
    assert tree_of(tmp_path / "out") == sorted([
        "AREG_report.json",
        os.path.join("1", "P001_T2_U_Reg.vtk"),
        os.path.join("1", "P001_T2_U_Reg_matrix.npy"),
        os.path.join("2", "P002_T2_U_Reg.vtk"),
        os.path.join("2", "P002_T2_U_Reg_matrix.npy"),
    ])


def test_both_arches_of_one_patient_are_registered_separately(tmp_path):
    cohort(tmp_path, jaws=("U", "L"), landmarks=UPPER_LABELS)
    write_markups(tmp_path / "ios_lm" / "P001_T2_L_lm_Pred.mrk.json", UPPER_LABELS)
    run_registration(tmp_path)

    report = read_report(tmp_path / "out")
    assert set(report["patients"]["1"]["meshes"]) == {"P001_T2_U.vtk", "P001_T2_L.vtk"}


def test_the_suffix_reaches_every_name_it_writes(tmp_path):
    cohort(tmp_path)
    run_registration(tmp_path, output_suffix="Study7")
    assert tree_of(tmp_path / "out") == sorted([
        "AREG_report.json",
        os.path.join("1", "P001_T2_U_Study7.vtk"),
        os.path.join("1", "P001_T2_U_Study7_matrix.npy"),
    ])


@pytest.mark.parametrize("given", ["", "   ", None])
def test_a_blank_suffix_falls_back_to_reg(tmp_path, given):
    cohort(tmp_path)
    run_registration(tmp_path, output_suffix=given)
    assert read_report(tmp_path / "out")["output_suffix"] == "Reg"
    assert os.path.join("1", "P001_T2_U_Reg.vtk") in tree_of(tmp_path / "out")


@pytest.mark.parametrize("given", ["../escape", "a/b", os.sep + "abs"])
def test_a_suffix_that_is_a_path_is_refused(tmp_path, given):
    """It is a name fragment. As a path it writes outside the output directory,
    which is the one guarantee this tool makes about where its results go."""
    cohort(tmp_path)
    with pytest.raises(ToolInputError, match="name fragment"):
        run_registration(tmp_path, output_suffix=given)


def test_a_mesh_whose_own_stem_carries_vtk_still_names_its_matrix_correctly(tmp_path):
    """`destination.replace(".vtk", "_matrix.npy")` rewrote EVERY occurrence, so
    a mesh called `P001_T2_U.vtk.vtk` produced a mangled matrix name beside a
    perfectly good mesh."""
    cohort(tmp_path)
    os.rename(
        str(tmp_path / "ios" / "P001_T2_U.vtk"), str(tmp_path / "ios" / "P001_T2_U.vtk.vtk")
    )
    run_registration(tmp_path)
    assert os.path.join("1", "P001_T2_U.vtk_Reg_matrix.npy") in tree_of(tmp_path / "out")


def test_the_output_directory_is_created_when_it_does_not_exist(tmp_path):
    cohort(tmp_path)
    run_registration(tmp_path, output_dir=str(tmp_path / "deep" / "nested" / "out"))
    assert os.path.isfile(str(tmp_path / "deep" / "nested" / "out" / "AREG_report.json"))


def test_nothing_is_written_outside_the_output_directory(tmp_path, monkeypatch):
    """The Slicer envelope wrote a log file, `<t2>_Center` copies and error
    files into the caller's own tree. None of that survives."""
    monkeypatch.chdir(tmp_path)
    cohort(tmp_path)
    before = {name: tree_of(tmp_path / name) for name in ("ios", "cbct", "ios_lm", "cbct_lm")}

    run_registration(tmp_path)

    assert {name: tree_of(tmp_path / name) for name in before} == before
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "cbct", "cbct_lm", "ios", "ios_lm", "out",
    ]


def test_the_working_directory_does_not_survive_a_successful_run(tmp_path):
    cohort(tmp_path)
    run_registration(tmp_path)
    assert not os.path.exists(str(tmp_path / "out" / dispatch.WORK_DIRNAME))


def test_the_working_directory_is_removed_when_the_run_fails(tmp_path):
    """A surviving `.areg_work/` means a run crashed -- and it would be shipped
    to the client inside the result archive."""
    cohort(tmp_path)
    with pytest.raises(ToolInputError):
        run_registration(tmp_path, ios_landmarks="")
    assert not os.path.exists(str(tmp_path / "out" / dispatch.WORK_DIRNAME))


def test_run_returns_the_output_directory(tmp_path):
    cohort(tmp_path)
    returned = run(
        ios=tmp_path / "ios",
        cbct=tmp_path / "cbct",
        output_dir=tmp_path / "out",
        automation="Registration",
        ios_landmarks=tmp_path / "ios_lm",
        cbct_landmarks=tmp_path / "cbct_lm",
    )
    assert str(returned) == str(tmp_path / "out")
    assert os.path.isfile(str(tmp_path / "out" / "AREG_report.json"))


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

def test_the_report_says_what_ran_and_on_what(tmp_path):
    cohort(tmp_path)
    run_registration(tmp_path)
    report = read_report(tmp_path / "out")

    assert report["modality"] == "IOSCBCT"
    assert report["automation"] == "Registration"
    assert report["output_suffix"] == "Reg"
    assert isinstance(report["duration_seconds"], float)

    patient = report["patients"]["1"]
    assert patient["status"] == "ok"
    assert patient["cbct"] == "P_0001_T2.nii.gz"
    mesh = patient["meshes"]["P001_T2_U.vtk"]
    assert mesh["status"] == "ok"
    assert mesh["landmarks_used"] == sorted(UPPER_LABELS)
    assert mesh["landmarks_dropped"] == []
    assert mesh["output"] == os.path.join("1", "P001_T2_U_Reg.vtk")


def test_the_report_carries_no_server_side_path(tmp_path):
    """The report travels to the client inside the archive. The job directory
    it was produced in does not."""
    cohort(tmp_path)
    run_registration(tmp_path)
    with open(str(tmp_path / "out" / "AREG_report.json"), encoding="utf-8") as handle:
        raw = handle.read()
    assert str(tmp_path) not in raw
    assert os.sep + "tmp" not in raw


def test_a_landmark_only_one_side_names_is_reported_as_dropped(tmp_path):
    cohort(tmp_path)
    extended = dict(moved(UPPER_LABELS), EXTRA=[1.0, 2.0, 3.0])
    write_markups(tmp_path / "cbct_lm" / "P_0001_T2_lm_Pred.mrk.json", extended)

    run_registration(tmp_path)
    mesh = read_report(tmp_path / "out")["patients"]["1"]["meshes"]["P001_T2_U.vtk"]
    assert mesh["landmarks_dropped"] == ["EXTRA"]


def test_a_patient_with_only_one_modality_is_named_in_the_report(tmp_path):
    cohort(tmp_path, patients=("P001",))
    write_mesh(tmp_path / "ios" / "P009_T2_U.vtk")

    run_registration(tmp_path)
    assert read_report(tmp_path / "out")["unpaired"] == {"9": "no CBCT"}


def test_the_icp_refines_on_the_cbct_the_run_was_given(tmp_path):
    """The refinement is reached from `register`, not merely implemented.

    It used to be unreachable: `register_one` refined only when handed points
    sampled from the CBCT and `register` sampled none, so every run stopped at
    the landmark fit -- six points placing an arch, with nothing able to notice
    that one of them was wrong. This is the test that would have said so.
    """
    cohort(tmp_path)
    # A flat patch rather than the default arch, and a box with a face in its
    # plane: the point-to-plane step measures along normals, so the moving
    # surface has to have normals that mean something.
    write_plane(tmp_path / "ios" / "P001_T2_U.vtk")
    # Wide enough to hold every landmark as well as the patch: a landmark
    # outside the bright box reads as not on a tooth, and the fit would drop it.
    write_contourable_volume(tmp_path / "cbct" / "P_0001_T2.nii.gz",
                             box=((3.0, -2.0, 1.0), (23.0, 12.0, 11.0)))

    run_registration(tmp_path)
    icp = read_report(tmp_path / "out")["patients"]["1"]["meshes"][
        "P001_T2_U.vtk"]["icp"]
    assert icp is not None
    assert icp["pairs"] >= 3 and icp["fitness"] > 0
    assert icp["used_normals"] is True


def test_a_cbct_no_reader_can_open_falls_back_to_the_landmarks_and_says_so(tmp_path):
    """Worse than refining, better than stopping -- and never silent.

    A volume that cannot be contoured leaves the landmark fit as the whole
    registration, which is what this tool did on every run before the ICP was
    wired in. It is a usable answer, so the run continues; what it must not do
    is come back looking like a run that simply had no ICP to do.
    """
    cohort(tmp_path)

    run_registration(tmp_path)
    patient = read_report(tmp_path / "out")["patients"]["1"]
    assert patient["meshes"]["P001_T2_U.vtk"]["icp"] is None
    assert "could not be contoured" in patient["cbct_surface_error"]
    assert str(tmp_path) not in patient["cbct_surface_error"]


# ---------------------------------------------------------------------------
# Failure, per mesh and per batch
# ---------------------------------------------------------------------------

def test_one_mesh_failing_does_not_cost_the_batch(tmp_path):
    """A cohort is the unit of work. One patient's landmarks being unusable
    must not throw away the ones that registered."""
    cohort(tmp_path, patients=("P001", "P002"))
    # Patient 2's CBCT landmarks share only one label with its intraoral set.
    write_markups(
        tmp_path / "cbct_lm" / "P_0002_T2_lm_Pred.mrk.json",
        {"UR1O": [0.0, 0.0, 0.0]},
    )

    run_registration(tmp_path)
    report = read_report(tmp_path / "out")

    assert report["patients"]["1"]["status"] == "ok"
    failed = report["patients"]["2"]
    assert failed["status"] == "failed"
    assert "share only 1 landmark(s)" in failed["meshes"]["P002_T2_U.vtk"]["error"]
    assert os.path.join("1", "P001_T2_U_Reg.vtk") in tree_of(tmp_path / "out")


def test_a_batch_where_nothing_registered_raises_and_says_where_to_look(tmp_path):
    cohort(tmp_path)
    write_markups(
        tmp_path / "cbct_lm" / "P_0001_T2_lm_Pred.mrk.json", {"UR1O": [0.0, 0.0, 0.0]}
    )
    with pytest.raises(RuntimeError, match="registered no mesh"):
        run_registration(tmp_path)


def test_a_mesh_with_no_landmarks_of_its_own_names_the_patient_and_the_side(tmp_path):
    cohort(tmp_path, patients=("P001", "P002"))
    os.remove(str(tmp_path / "ios_lm" / "P002_T2_U_lm_Pred.mrk.json"))

    run_registration(tmp_path)
    failed = read_report(tmp_path / "out")["patients"]["2"]["meshes"]["P002_T2_U.vtk"]
    assert failed["status"] == "failed"
    assert "patient '2'" in failed["error"]
    assert "the intraoral" in failed["error"]


def test_an_empty_landmark_folder_is_refused_before_any_mesh_is_read(tmp_path):
    cohort(tmp_path)
    os.remove(str(tmp_path / "cbct_lm" / "P_0001_T2_lm_Pred.mrk.json"))

    with pytest.raises(ToolInputError) as raised:
        run_registration(tmp_path)
    assert "1 intraoral and 0 CBCT" in str(raised.value)


# ---------------------------------------------------------------------------
# The defect that produced a confident wrong answer
# ---------------------------------------------------------------------------

def test_each_patient_is_registered_on_its_own_landmarks(tmp_path):
    """Matching on the jaw token alone put the SECOND patient's mesh onto the
    FIRST patient's landmarks: every ALI_IOS file carries a `_U` token,
    `sorted()` puts P1's first, and the first jaw match won. Nothing looked
    wrong -- a rigid transform was produced and the report said "ok"."""
    cohort(tmp_path, patients=("P001", "P002"))
    # Patient 2's CBCT sits somewhere else entirely, so borrowing patient 1's
    # points gives a visibly different transform.
    write_markups(
        tmp_path / "cbct_lm" / "P_0002_T2_lm_Pred.mrk.json",
        moved(UPPER_LABELS, shift=(100.0, 100.0, 100.0)),
    )

    run_registration(tmp_path)
    second = np.load(str(tmp_path / "out" / "2" / "P002_T2_U_Reg_matrix.npy"))
    assert second[:3, 3] == pytest.approx([100.0, 100.0, 100.0], abs=1e-5)


def test_a_patients_two_arches_do_not_borrow_each_others_points(tmp_path):
    """Within one patient the jaw token is what separates them, and it must
    beat sort order the same way."""
    cohort(tmp_path, jaws=("U", "L"))
    write_markups(tmp_path / "ios_lm" / "P001_T2_L_lm_Pred.mrk.json", LOWER_LABELS)
    write_markups(
        tmp_path / "cbct_lm" / "P_0001_T2_U_lm_Pred.mrk.json", moved(UPPER_LABELS)
    )
    write_markups(
        tmp_path / "cbct_lm" / "P_0001_T2_L_lm_Pred.mrk.json",
        moved(LOWER_LABELS, shift=(50.0, 0.0, 0.0)),
    )
    os.remove(str(tmp_path / "cbct_lm" / "P_0001_T2_lm_Pred.mrk.json"))

    run_registration(tmp_path)
    upper = np.load(str(tmp_path / "out" / "1" / "P001_T2_U_Reg_matrix.npy"))
    lower = np.load(str(tmp_path / "out" / "1" / "P001_T2_L_Reg_matrix.npy"))
    assert upper[:3, 3] == pytest.approx([3.0, -2.0, 1.0], abs=1e-5)
    assert lower[:3, 3] == pytest.approx([50.0, 0.0, 0.0], abs=1e-5)


def test_one_cbct_landmark_file_serves_both_arches(tmp_path):
    """A CBCT covers both arches in one volume, so its landmark file has no jaw
    to name. The labels carry it, and `shared_landmarks` intersects."""
    cohort(tmp_path, jaws=("U", "L"))
    write_markups(tmp_path / "ios_lm" / "P001_T2_L_lm_Pred.mrk.json", LOWER_LABELS)
    write_markups(
        tmp_path / "cbct_lm" / "P_0001_T2_lm_Pred.mrk.json",
        dict(moved(UPPER_LABELS), **moved(LOWER_LABELS, shift=(50.0, 0.0, 0.0))),
    )

    run_registration(tmp_path)
    report = read_report(tmp_path / "out")["patients"]["1"]["meshes"]
    assert report["P001_T2_U.vtk"]["landmarks_used"] == sorted(UPPER_LABELS)
    assert report["P001_T2_L.vtk"]["landmarks_used"] == sorted(LOWER_LABELS)


def test_two_sites_holding_a_patient_of_the_same_name_keep_their_own_files(tmp_path):
    """ALI mirrors the input tree, so the same base name really does occur
    twice. Keyed by base name the second file silently replaced the first."""
    cohort(tmp_path)
    write_mesh(tmp_path / "ios" / "siteB" / "P001_T2_U.vtk")
    write_markups(
        tmp_path / "ios_lm" / "siteB" / "P001_T2_U_lm_Pred.mrk.json", UPPER_LABELS
    )
    write_volume(tmp_path / "cbct" / "siteB" / "P_0001_T2.nii.gz")

    run_registration(tmp_path)
    assert len(dispatch._landmarks_by_jaw(str(tmp_path / "ios_lm"))) == 2


def test_an_arch_the_cbct_cannot_confirm_is_refused_rather_than_written(tmp_path):
    """An ICP that matched nothing still returns a matrix.

    Writing it put an untouched intraoral scan in the results folder under the
    name of a registered one -- the worst failure this tool can have, because
    nothing about the output says so. The floor is on the share of the arch that
    found a match, which is the one number that sees it.
    """
    cohort(tmp_path)
    write_contourable_volume(tmp_path / "cbct" / "P_0001_T2.nii.gz",
                             box=((3.0, -2.0, 1.0), (23.0, 12.0, 11.0)))
    # Far from the CBCT surface and staying there: the landmark fit is a
    # translation of a few millimetres, so it cannot bring this back.
    write_plane(tmp_path / "ios" / "P001_T2_U.vtk", centre=(500.0, 500.0, 500.0))

    with pytest.raises(RuntimeError, match="registered no mesh"):
        run_registration(tmp_path)
    # Nothing written is the whole point: the failure mode being guarded against
    # is a file that looks like a registered arch and is not one.
    assert list(tree_of(tmp_path / "out")) == []
