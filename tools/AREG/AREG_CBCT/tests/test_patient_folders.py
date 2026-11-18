"""A cohort kept as one folder per patient reaches the engine as T1 and T2."""

import inspect
import os
import typing

import pytest
import sadt_areg_cbct
from sadt_areg_common import catalogs
from sadt_areg_common.errors import ToolInputError


def touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("x")


@pytest.fixture
def seen(monkeypatch):
    """What the engine was handed, read before the sorted folders are removed."""
    found = {}

    def main(t1, t2, **_rest):
        found["t1"] = sorted(os.listdir(t1))
        found["t2"] = sorted(os.listdir(t2))
        found["scratch"] = os.path.dirname(t1)

    monkeypatch.setattr(sadt_areg_cbct, "main", main)
    return found


def test_the_layouts_are_the_catalogs():
    hint = inspect.signature(sadt_areg_cbct.run).parameters["input_layout"].annotation
    assert list(typing.get_args(hint)) == list(catalogs.LAYOUT_CHOICES)


def test_a_patient_folder_reaches_the_engine_as_its_two_timepoints(tmp_path, seen):
    root = tmp_path / "patients" / "case_042"
    touch(root / "T1" / "scan.nii.gz")
    touch(root / "T2" / "scan.nii.gz")
    touch(root / "T1" / "scan_seg.nii.gz")

    sadt_areg_cbct.run(
        input_layout=catalogs.LAYOUT_BY_PATIENT, patients=tmp_path / "patients",
        output_dir=tmp_path / "out",
    )

    assert seen["t1"] == ["case_042_T1.nii.gz"]
    assert seen["t2"] == ["case_042_T2.nii.gz"]
    assert not os.path.exists(seen["scratch"])


def test_the_per_type_layout_still_needs_both_folders(tmp_path, seen):
    with pytest.raises(ToolInputError, match="'t2' is empty"):
        sadt_areg_cbct.run(t1=tmp_path, output_dir=tmp_path / "out")
