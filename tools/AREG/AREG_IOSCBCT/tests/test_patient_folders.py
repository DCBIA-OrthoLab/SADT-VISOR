"""A cohort kept as one folder per patient registers like one kept per type."""

import inspect
import os
import shutil
import typing

import pytest
import numpy as np
from conftest import (
    UPPER_LABELS, cohort, moved, read_report, tree_of, write_markups, write_mesh,
    write_volume,
)
from sadt_areg_common import catalogs
from sadt_areg_common.errors import ToolInputError
from sadt_areg_ioscbct import run


def as_patient_folders(tmp_path, subfolders=False):
    """The cohort `cohort()` wrote, moved to `patients/<patient>/`."""
    root = tmp_path / "patients" / "P001"
    for side in ("ios", "cbct"):
        destination = root / side.upper() if subfolders else root
        destination.mkdir(parents=True, exist_ok=True)
        for name in os.listdir(tmp_path / side):
            shutil.move(str(tmp_path / side / name), str(destination / name))
    return tmp_path / "patients"


def registration(tmp_path, **arguments):
    return run(
        output_dir=tmp_path / "out", automation="Registration",
        ios_landmarks=tmp_path / "ios_lm", cbct_landmarks=tmp_path / "cbct_lm",
        **arguments,
    )


def test_the_layouts_are_the_catalogs():
    hint = inspect.signature(run).parameters["input_layout"].annotation
    assert list(typing.get_args(hint)) == list(catalogs.LAYOUT_CHOICES)


@pytest.mark.parametrize("subfolders", [False, True])
def test_a_patient_folder_registers_into_a_folder_of_its_name(tmp_path, subfolders):
    cohort(tmp_path)
    patients = as_patient_folders(tmp_path, subfolders=subfolders)

    registration(tmp_path, input_layout=catalogs.LAYOUT_BY_PATIENT, patients=patients)

    assert tree_of(tmp_path / "out") == sorted([
        "AREG_report.json",
        os.path.join("P001", "P001_U_Reg.vtk"),
        os.path.join("P001", "P001_U_Reg_matrix.npy"),
        os.path.join("P001", "scene.json"),
    ])
    assert read_report(tmp_path / "out")["patients"]["P001"]["status"] == "ok"


def test_the_sorted_copies_are_removed_and_never_handed_back(tmp_path):
    cohort(tmp_path)
    patients = as_patient_folders(tmp_path)
    registration(tmp_path, input_layout=catalogs.LAYOUT_BY_PATIENT, patients=patients)
    assert not [name for name in os.listdir(tmp_path) if name.startswith("areg-patients-")]


def test_a_patient_folder_with_no_number_is_refused(tmp_path):
    """Patients are matched on digits; a name without one cannot be."""
    cohort(tmp_path)
    patients = as_patient_folders(tmp_path)
    os.rename(patients / "P001", patients / "no_number")
    with pytest.raises(ToolInputError, match="carries no number"):
        registration(tmp_path, input_layout=catalogs.LAYOUT_BY_PATIENT, patients=patients)


def test_the_per_type_layout_still_needs_both_folders(tmp_path):
    cohort(tmp_path)
    with pytest.raises(ToolInputError, match="'cbct' is empty"):
        registration(tmp_path, ios=tmp_path / "ios")


@pytest.mark.parametrize("subfolders", [False, True])
def test_a_patient_folder_holding_a_t1_and_a_t2_registers_each_on_its_own_cbct(
        tmp_path, subfolders):
    """Renamed `<patient>_CBCT.nii.gz` both, a T1 and a T2 were one name and
    the run was refused. The timepoint is kept, in the name or the sub-folder."""
    shifts = {"T1": (3.0, -2.0, 1.0), "T2": (-4.0, 5.0, 2.0)}
    root = tmp_path / "patients" / "P007"
    for point, shift in shifts.items():
        where, stem = (root / point, "P007") if subfolders else (root, f"P007_{point}")
        write_volume(where / f"{stem}.nii.gz")
        write_mesh(where / f"{stem}_U.vtk")
        write_markups(tmp_path / "cbct_lm" / f"P007_{point}_CBCT_lm.mrk.json",
                      moved(UPPER_LABELS, shift))
        write_markups(tmp_path / "ios_lm" / f"P007_{point}_U_lm.mrk.json", UPPER_LABELS)

    registration(tmp_path, input_layout=catalogs.LAYOUT_BY_PATIENT,
                 patients=tmp_path / "patients")

    report = read_report(tmp_path / "out")
    assert report["meshes_failed"] == 0
    assert sorted(report["patients"]) == ["P007_T1", "P007_T2"]
    for point, shift in shifts.items():
        matrix = np.load(str(tmp_path / "out" / f"P007_{point}" / f"P007_{point}_U_Reg_matrix.npy"))
        assert np.allclose(matrix[:3, 3], shift, atol=1e-3), point
