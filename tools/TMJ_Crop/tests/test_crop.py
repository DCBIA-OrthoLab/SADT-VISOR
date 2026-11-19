"""The geometry, on a synthetic head: no GPU, no weights, no network."""

import json

import nibabel as nib
import numpy as np
import pytest

from conftest import CONDYLES, SHIFT, reference_ras, write_scene
from sadt_tmj_crop import crop

USED = ["Ba", "S", "N", "RPo", "LPo", "ROr", "LOr"]


def _crop(tmp_path, settings=None, shift=SHIFT):
    scan, markups, mask = write_scene(tmp_path / "in", "a", scene_shift=shift)
    out = tmp_path / "out"
    out.mkdir()
    result = crop.crop_scan(
        scan, markups, mask, reference_ras(), USED, {},
        settings or crop.CropSettings(), "a", str(out),
    )
    return result, out


def test_the_translation_is_recovered_from_the_landmarks_alone(tmp_path):
    result, _ = _crop(tmp_path)

    assert np.allclose(result.report["translation"], SHIFT, atol=1e-3)
    assert max(result.report["fit_residual_mm"].values()) < 0.01


def test_both_joints_are_found_and_written_at_the_documented_size(tmp_path):
    result, out = _crop(tmp_path)

    for side in ("R", "L"):
        assert result.report["sides"][side]["status"] == "ok"
        image = nib.load(out / f"a_{side}.nii.gz")
        assert image.shape == (96, 112, 80)
    assert result.report["box_mm"] == [28.8, 33.6, 24.0]


def test_the_condyle_top_is_where_the_synthetic_condyle_is(tmp_path):
    result, _ = _crop(tmp_path)

    for side, centre in CONDYLES.items():
        top = np.array(result.report["sides"][side]["condyle_top"])
        # The top of a 6 mm sphere sits 6 mm above its centre, within a voxel. Its flat top
        # is a ring of equally high voxels, so across the plane only "near the centre" is promised.
        assert abs(top[2] - (centre[2] + 6)) <= 1.5
        assert np.linalg.norm(top[:2] - centre[:2]) <= 4.5


def test_the_left_crop_is_the_mirror_image_of_the_right(tmp_path):
    _, out = _crop(tmp_path)

    right = np.asanyarray(nib.load(out / "a_R.nii.gz").dataobj)
    left = np.asanyarray(nib.load(out / "a_L.nii.gz").dataobj)

    # The scene is mirror-symmetric, so a mirrored left crop equals the right one.
    assert np.abs(right.astype(float) - left.astype(float)).max() < 20
    assert right.max() > 500, "the condyle is inside the box"


def test_a_scene_with_no_mandible_near_a_porion_says_so(tmp_path):
    scan, markups, _ = write_scene(tmp_path / "in", "a")
    empty = tmp_path / "empty.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((160,) * 3, np.uint8), nib.load(scan).affine), empty)
    out = tmp_path / "out"
    out.mkdir()

    with pytest.raises(ValueError):
        crop.crop_scan(scan, markups, empty, reference_ras(), USED, {}, crop.CropSettings(), "a", str(out))


def test_no_shared_landmark_is_refused_rather_than_guessed():
    with pytest.raises(ValueError, match="no landmark is shared"):
        crop.translation_to_reference({"X": np.zeros(3)}, {"Ba": np.zeros(3)}, ["Ba"])


def test_a_dropped_landmark_takes_no_part_in_the_fit(tmp_path):
    scan, markups, mask = write_scene(tmp_path / "in", "a")
    data = json.loads(markups.read_text())
    data["markups"][0]["controlPoints"][0]["position"] = [500.0, 500.0, 500.0]  # Ba, an outlier
    markups.write_text(json.dumps(data))
    out = tmp_path / "out"
    out.mkdir()

    kept = [name for name in USED if name != "Ba"]
    result = crop.crop_scan(scan, markups, mask, reference_ras(), kept, {"Ba": "outlier"},
                            crop.CropSettings(), "a", str(out))

    assert np.allclose(result.report["translation"], SHIFT, atol=1e-3)
    assert result.report["landmarks_dropped"] == {"Ba": "outlier"}


def test_a_fixed_scale_changes_the_voxel_size_and_not_the_voxel_count(tmp_path):
    result, out = _crop(tmp_path, crop.CropSettings(scale=1.5))

    assert result.report["spacing_mm"] == pytest.approx(0.45)
    assert nib.load(out / "a_R.nii.gz").shape == (96, 112, 80)


def test_the_mandible_volume_scale_follows_the_cube_root(tmp_path):
    result, _ = _crop(tmp_path, crop.CropSettings(scale_by="mandible-volume"))

    expected = (result.report["mandible_volume_ml"] / 54.6) ** (1 / 3)
    assert result.report["scale"] == pytest.approx(expected, rel=0.02)


def test_lps_markups_are_flipped_to_ras_and_ras_ones_are_left_alone(tmp_path):
    for system, written, expected in (("LPS", [10, 20, 30], [-10, -20, 30]), ("RAS", [10, 20, 30], [10, 20, 30])):
        path = tmp_path / f"{system}.mrk.json"
        path.write_text(json.dumps({"markups": [{"coordinateSystem": system, "controlPoints": [
            {"label": "N", "position": written}]}]}))
        assert crop.read_landmarks(path)["N"].tolist() == expected


@pytest.mark.parametrize("side", ["R", "L"])
def test_a_crop_opens_on_top_of_the_scan_it_was_cut_from(tmp_path, side):
    """Read through its own affine, a crop must match the scan at those world positions.

    The crop is computed in the oriented frame, which is the scan's frame shifted by
    the translation; the file has to be placed in the SCAN's frame, or it floats a few
    millimetres off the oriented scan this tool can return.
    """
    from scipy import ndimage

    _, out = _crop(tmp_path)
    scan = nib.load(tmp_path / "in" / "a_Or.nii.gz")
    cropped = nib.load(out / f"a_{side}.nii.gz")
    values = np.asanyarray(cropped.dataobj).astype(np.float32)

    voxels = np.indices(values.shape).reshape(3, -1).T
    world = voxels @ cropped.affine[:3, :3].T + cropped.affine[:3, 3]
    inverse = np.linalg.inv(scan.affine)
    sampled = ndimage.map_coordinates(
        np.asanyarray(scan.dataobj).astype(np.float32),
        (world @ inverse[:3, :3].T + inverse[:3, 3]).T, order=1, mode="nearest")

    assert np.corrcoef(sampled, values.ravel())[0, 1] > 0.99
    # And it would not, were it left in the oriented frame.
    shifted = world + SHIFT
    off = ndimage.map_coordinates(
        np.asanyarray(scan.dataobj).astype(np.float32),
        (shifted @ inverse[:3, :3].T + inverse[:3, 3]).T, order=1, mode="nearest")
    assert np.corrcoef(off, values.ravel())[0, 1] < 0.99
