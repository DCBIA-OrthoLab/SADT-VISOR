"""A cohort kept as one folder per patient, sorted into one folder per type."""

import os

import pytest
from sadt_areg_common import catalogs, pairing, patient_folders
from sadt_areg_common.errors import ToolInputError


def touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("x")
    return path


def tree(root):
    return sorted(
        os.path.relpath(os.path.join(directory, name), root)
        for directory, _dirs, names in os.walk(root) for name in names
    )


_CBCT = patient_folders.by_timepoint(pairing.is_scan_file, skip=catalogs.MASK_TOKENS)


def _cbct_name(patient, role, relative, index):
    return f"{patient}_{role.upper()}{patient_folders.extension_of(relative)}"


# ---------------------------------------------------------------------------
# What a file is
# ---------------------------------------------------------------------------

def test_the_timepoint_comes_from_the_folder_or_the_name():
    assert patient_folders.timepoint_of("T1/scan.nii.gz") == "T1"
    assert patient_folders.timepoint_of("t2/scan.nii.gz") == "T2"
    assert patient_folders.timepoint_of("case_042_T2.nii.gz") == "T2"
    assert patient_folders.timepoint_of("scan.nii.gz") is None


def test_a_folder_and_a_name_that_disagree_give_no_timepoint():
    """Guessing between them is how a baseline gets registered onto itself."""
    assert patient_folders.timepoint_of("T1/scan_T2.nii.gz") is None


def test_the_jaw_comes_from_the_name():
    assert patient_folders.jaw_of("scan_U.stl") == "U"
    assert patient_folders.jaw_of("Lower.vtk") == "L"
    assert patient_folders.jaw_of("scan.vtk") is None


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------

def test_each_patients_files_are_filed_under_their_folders_name(tmp_path):
    touch(tmp_path / "in" / "case_042" / "T1" / "scan.nii.gz")
    touch(tmp_path / "in" / "case_042" / "T2" / "scan.nii.gz")
    touch(tmp_path / "in" / "case_043" / "case_043_T1.nii.gz")
    touch(tmp_path / "in" / "case_043" / "case_043_T2.nii.gz")

    folders = patient_folders.sort(
        str(tmp_path / "in"), str(tmp_path / "work"), ("t1", "t2"), _CBCT, _cbct_name
    )
    assert tree(folders["t1"]) == ["case_042_T1.nii.gz", "case_043_T1.nii.gz"]
    assert tree(folders["t2"]) == ["case_042_T2.nii.gz", "case_043_T2.nii.gz"]


def test_what_the_engine_does_not_read_is_left_out(tmp_path):
    """A segmentation beside its scan is not a second T1, and a note is not a
    scan at all."""
    touch(tmp_path / "in" / "case_042" / "T1" / "scan.nii.gz")
    touch(tmp_path / "in" / "case_042" / "T1" / "scan_seg.nii.gz")
    touch(tmp_path / "in" / "case_042" / "notes.txt")

    folders = patient_folders.sort(
        str(tmp_path / "in"), str(tmp_path / "work"), ("t1", "t2"), _CBCT, _cbct_name
    )
    assert tree(folders["t1"]) == ["case_042_T1.nii.gz"]
    assert tree(folders["t2"]) == []


def test_two_files_filed_under_one_name_are_refused(tmp_path):
    """One replacing the other silently would register whichever came last."""
    touch(tmp_path / "in" / "case_042" / "T1" / "a.nii.gz")
    touch(tmp_path / "in" / "case_042" / "T1" / "b.nii.gz")

    with pytest.raises(ToolInputError, match="same T1 input"):
        patient_folders.sort(
            str(tmp_path / "in"), str(tmp_path / "work"), ("t1", "t2"), _CBCT, _cbct_name
        )


def test_a_folder_with_no_patient_folder_is_refused(tmp_path):
    touch(tmp_path / "in" / "case_042_T1.nii.gz")
    with pytest.raises(ToolInputError, match="no sub-folder"):
        patient_folders.sort(
            str(tmp_path / "in"), str(tmp_path / "work"), ("t1", "t2"), _CBCT, _cbct_name
        )


def test_the_files_are_the_callers_own_not_copies_that_differ(tmp_path):
    source = touch(tmp_path / "in" / "case_042" / "T1" / "scan.nii.gz")
    folders = patient_folders.sort(
        str(tmp_path / "in"), str(tmp_path / "work"), ("t1", "t2"), _CBCT, _cbct_name
    )
    with open(os.path.join(folders["t1"], "case_042_T1.nii.gz"), encoding="utf-8") as handle:
        assert handle.read() == "x"
    assert os.path.exists(source)


# ---------------------------------------------------------------------------
# resolve
# ---------------------------------------------------------------------------

def test_the_per_type_layout_passes_its_folders_through(tmp_path):
    folders, scratch = patient_folders.resolve(
        catalogs.LAYOUT_BY_TYPE, "", {"t1": "/a", "t2": "/b"}, str(tmp_path / "out"),
        _CBCT, _cbct_name,
    )
    assert folders == {"t1": "/a", "t2": "/b"} and scratch is None


def test_the_per_type_layout_still_needs_both_folders(tmp_path):
    with pytest.raises(ToolInputError, match="'t2' is empty"):
        patient_folders.resolve(
            catalogs.LAYOUT_BY_TYPE, "", {"t1": "/a", "t2": ""}, str(tmp_path / "out"),
            _CBCT, _cbct_name,
        )


def test_the_per_patient_layout_sorts_beside_the_output_never_in_it(tmp_path):
    """In the output folder, the sorted copies would be handed back."""
    touch(tmp_path / "in" / "case_042" / "T1" / "scan.nii.gz")
    touch(tmp_path / "in" / "case_042" / "T2" / "scan.nii.gz")
    (tmp_path / "out").mkdir()

    folders, scratch = patient_folders.resolve(
        catalogs.LAYOUT_BY_PATIENT, str(tmp_path / "in"), {"t1": "", "t2": ""},
        str(tmp_path / "out"), _CBCT, _cbct_name,
    )
    assert os.path.dirname(scratch) == str(tmp_path)
    assert tree(folders["t1"]) == ["case_042_T1.nii.gz"]
    assert os.listdir(tmp_path / "out") == []


def test_a_refused_sort_leaves_no_scratch_folder_behind(tmp_path):
    (tmp_path / "in").mkdir()
    (tmp_path / "out").mkdir()
    with pytest.raises(ToolInputError):
        patient_folders.resolve(
            catalogs.LAYOUT_BY_PATIENT, str(tmp_path / "in"), {"t1": "", "t2": ""},
            str(tmp_path / "out"), _CBCT, _cbct_name,
        )
    assert sorted(os.listdir(tmp_path)) == ["in", "out"]
