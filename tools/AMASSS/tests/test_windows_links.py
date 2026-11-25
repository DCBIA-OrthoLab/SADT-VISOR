"""A clinician's Windows account may not create symbolic links (WinError 1314).

The scan handed to nnUNet is then a hard link, or a copy, never a failure.
"""

import os

import numpy as np
import pytest
import SimpleITK as sitk
from sadt_amasss import pipeline


def _no_symlink(*_args, **_kwargs):
    raise OSError(1314, "A required privilege is not held by the client")


def _no_link(*_args, **_kwargs):
    raise OSError(17, "Hard links are not supported on this volume")


@pytest.fixture
def scan(tmp_path):
    path = tmp_path / "scan.nii.gz"
    image = sitk.GetImageFromArray(np.arange(4 * 5 * 6, dtype=np.int16).reshape(4, 5, 6))
    image.SetSpacing((0.4, 0.4, 0.4))
    sitk.WriteImage(image, str(path))
    return path


@pytest.mark.parametrize("broken", [["symlink"], ["symlink", "link"]])
def test_a_scan_is_staged_without_the_privilege_to_link(tmp_path, scan, monkeypatch, broken):
    if "symlink" in broken:
        monkeypatch.setattr(os, "symlink", _no_symlink)
    if "link" in broken:
        monkeypatch.setattr(os, "link", _no_link)
    destination = tmp_path / "nnunet_input" / "p_000_0000.nii.gz"
    destination.parent.mkdir()

    pipeline._convert_to_nifti(str(scan), str(destination))

    assert not destination.is_symlink()
    assert np.array_equal(sitk.GetArrayFromImage(sitk.ReadImage(str(destination))),
                          sitk.GetArrayFromImage(sitk.ReadImage(str(scan))))
    assert scan.exists()
