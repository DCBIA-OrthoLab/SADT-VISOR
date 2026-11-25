"""A clinician's Windows account may not create symbolic links (WinError 1314).

A scan staged for ASO, Convert or AMASSS is then a hard link, or a copy, never
a failure; removing the stage leaves the scan in place.
"""

import os
import shutil

import pytest
from sadt_tmj_crop import dispatch


def _no_symlink(*_args, **_kwargs):
    raise OSError(1314, "A required privilege is not held by the client")


def _no_link(*_args, **_kwargs):
    raise OSError(17, "Hard links are not supported on this volume")


@pytest.mark.parametrize("broken", [[], ["symlink"], ["symlink", "link"]])
def test_a_scan_is_staged_whatever_links_the_account_may_make(tmp_path, monkeypatch, broken):
    scan = tmp_path / "scan.nii.gz"
    scan.write_bytes(b"cbct bytes")
    if "symlink" in broken:
        monkeypatch.setattr(os, "symlink", _no_symlink)
    if "link" in broken:
        monkeypatch.setattr(os, "link", _no_link)
    stage = tmp_path / "stage"
    stage.mkdir()

    dispatch._place(scan, stage / "case_Or.nii.gz")

    assert (stage / "case_Or.nii.gz").read_bytes() == b"cbct bytes"
    assert (stage / "case_Or.nii.gz").is_symlink() == (not broken)
    shutil.rmtree(stage)
    assert scan.read_bytes() == b"cbct bytes"
