"""A cohort kept as one folder per patient registers like one kept per type."""

import inspect
import os
import shutil
import typing

import pytest
from conftest import cohort, read_report, tree_of
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
