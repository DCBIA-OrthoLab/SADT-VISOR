"""Reading the CBCT: the level to contour it at, and what it says about the
landmarks it was given.

Both come off the voxels, so both run for real here -- no card, no checkpoint,
no other tool. The volumes are written by `write_contourable_volume`, which
holds one bright box in a dark field: two intensities and nothing between them,
so every number below is one that can be worked out by hand.
"""

import numpy as np
import pytest

from conftest import write_contourable_volume, write_volume
from sadt_areg_ioscbct import volume


BOX = ((0.0, 0.0, 0.0), (10.0, 10.0, 10.0))
# On the box's faces, so each reads `bright` at its brightest.
ON_TEETH = {
    "UR1O": np.array([0.0, 0.0, 0.0]),
    "UR3O": np.array([10.0, 0.0, 0.0]),
    "UR6O": np.array([0.0, 10.0, 0.0]),
    "UL1O": np.array([10.0, 10.0, 10.0]),
}


def test_the_level_is_read_off_the_scan_rather_than_fixed(tmp_path):
    """The same anatomy images at different intensities depending on the
    machine and the reconstruction, so one number cannot serve every scan.

    Halfway between the enamel at the landmarks and the surround: here 1000 and
    -1000, so 0.
    """
    path = write_contourable_volume(tmp_path / "v.nii.gz", box=BOX)
    _surface, _on_enamel = volume.read(path, ON_TEETH)

    import SimpleITK as sitk
    image = sitk.ReadImage(path)
    ijk_to_lps = np.eye(4)
    ijk_to_lps[:3, :3] = (np.array(image.GetDirection()).reshape(3, 3)
                          @ np.diag(image.GetSpacing()))
    ijk_to_lps[:3, 3] = np.array(image.GetOrigin())
    array = sitk.GetArrayFromImage(image)

    assert volume.surface_threshold(
        array, ijk_to_lps, np.array(list(ON_TEETH.values()))) == pytest.approx(0.0)


def test_a_level_the_caller_named_wins_over_the_scan(tmp_path):
    """For the scan the rule cannot read."""
    path = write_contourable_volume(tmp_path / "v.nii.gz", box=BOX)
    surface, _on_enamel = volume.read(path, ON_TEETH, level=500.0)
    assert surface.n_points > 0


def test_a_scan_with_no_landmarks_to_probe_falls_back_to_a_known_number(tmp_path):
    """Not a guess: 400 is what every run used before the level was measured."""
    path = write_contourable_volume(tmp_path / "v.nii.gz", box=BOX)

    import SimpleITK as sitk
    array = sitk.GetArrayFromImage(sitk.ReadImage(path))
    assert volume.surface_threshold(
        array, np.eye(4), None) == volume.FALLBACK_LEVEL


def test_crowns_no_brighter_than_their_surround_fall_back_too(tmp_path):
    """A uniform scan cannot say where a surface is, and a level read off it
    would be an arbitrary number carrying the authority of a measurement."""
    path = write_contourable_volume(tmp_path / "v.nii.gz", box=BOX,
                                    bright=0.0, dark=0.0)

    import SimpleITK as sitk
    image = sitk.ReadImage(path)
    ijk_to_lps = np.eye(4)
    ijk_to_lps[:3, 3] = np.array(image.GetOrigin())
    assert volume.surface_threshold(
        sitk.GetArrayFromImage(image), ijk_to_lps,
        np.array(list(ON_TEETH.values()))) == volume.FALLBACK_LEVEL


def test_a_landmark_off_the_teeth_is_named_as_such(tmp_path):
    """The check the landmark fit cannot make. A point the network put on the
    opposing arch is perfectly consistent with itself, so no residual sees it;
    the voxels under it do.
    """
    path = write_contourable_volume(tmp_path / "v.nii.gz", box=BOX)
    off = dict(ON_TEETH, UL6O=np.array([40.0, 40.0, 40.0]))

    _surface, on_enamel = volume.read(path, off)
    assert on_enamel["UL6O"] is False
    assert all(on_enamel[label] for label in ON_TEETH)


def test_the_surface_comes_back_in_patient_coordinates(tmp_path):
    """Contoured on the voxel grid and then mapped through the scan's own
    ijk-to-patient matrix. Skipping that step gives a surface that looks
    plausible and sits nowhere near the landmarks it is meant to be registered
    against."""
    path = write_contourable_volume(tmp_path / "v.nii.gz", box=BOX)
    surface, _on_enamel = volume.read(path, ON_TEETH)

    low, high = np.array(BOX[0]), np.array(BOX[1])
    bounds = np.array(surface.bounds).reshape(3, 2)
    assert np.all(np.abs(bounds[:, 0] - low) < 1.0)
    assert np.all(np.abs(bounds[:, 1] - high) < 1.0)


def test_a_file_no_reader_can_open_raises_rather_than_returning_nothing(tmp_path):
    """The caller turns this into a landmark-only run and says so in the
    report. It can only do that if it is told."""
    path = write_volume(tmp_path / "not-a-volume.nii.gz")
    with pytest.raises(Exception):
        volume.read(path, ON_TEETH)
