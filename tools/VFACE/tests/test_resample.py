"""Putting a cohort on one voxel grid -- the step VFACE borrows from MRI2CBCT.

Vendored rather than called, because MRI2CBCT is not a tool in this repository
and what VFACE asks it for is not MRI2CBCT: every argument it sends turns off
the MRI branches, the mirroring, the target-size fit and the nearest-neighbour
interpolation. See `resample.py` for the full accounting.
"""

import os

import numpy as np
import pytest

from conftest import cohort, read_volume, tree_of, write_volume
from sadt_vface import resample
from sadt_vface.errors import ToolInputError


def test_the_spacing_is_the_one_asked_for(tmp_path):
    write_volume(tmp_path / "in" / "P001_T1.nii.gz")
    resample.resample_cohort(str(tmp_path / "in"), str(tmp_path / "out"))

    assert read_volume(tmp_path / "out" / "P001_T1.nii.gz").GetSpacing() == pytest.approx(
        resample.DEFAULT_SPACING
    )


def test_the_spacing_upstream_sends_is_isotropic_and_fine(tmp_path):
    """0.3 mm, upstream's value. Finer than most CBCTs are acquired at, so this
    is an upsample for nearly every cohort -- which is the point: the later
    steps compare volumes voxel for voxel and need them on one grid."""
    assert resample.DEFAULT_SPACING == (0.3, 0.3, 0.3)


def test_the_grid_size_is_kept_rather_than_recomputed(tmp_path):
    """Forcing a CBCT to a size derived from the new spacing crops the full
    head down to it and discards most of the volume. Upstream passes the
    file's own size through, and says so."""
    write_volume(tmp_path / "in" / "P001_T1.nii.gz", size=(24, 20, 16))
    resample.resample_cohort(str(tmp_path / "in"), str(tmp_path / "out"))

    assert read_volume(tmp_path / "out" / "P001_T1.nii.gz").GetSize() == (24, 20, 16)


def test_the_two_volumes_share_a_centre(tmp_path):
    """What `center=True` means: at a finer spacing and the same size the field
    of view shrinks, and it has to shrink around the anatomy rather than away
    from it."""
    import SimpleITK as sitk

    path = write_volume(tmp_path / "in" / "P001_T1.nii.gz")
    before = read_volume(path)
    resample.resample_cohort(str(tmp_path / "in"), str(tmp_path / "out"))
    after = read_volume(tmp_path / "out" / "P001_T1.nii.gz")

    def centre(image):
        size = np.array(image.GetSize(), dtype=float)
        spacing = np.array(image.GetSpacing(), dtype=float)
        return np.array(image.GetOrigin(), dtype=float) + 0.5 * size * spacing

    assert centre(after) == pytest.approx(centre(before), abs=1e-6)
    assert isinstance(sitk.ReadImage(str(path)), sitk.Image)


def test_centring_can_be_switched_off_and_then_the_origin_does_not_move(tmp_path):
    path = write_volume(tmp_path / "in" / "P001_T1.nii.gz")
    origin = read_volume(path).GetOrigin()

    resample.resample_cohort(str(tmp_path / "in"), str(tmp_path / "out"), centre=False)
    assert read_volume(tmp_path / "out" / "P001_T1.nii.gz").GetOrigin() == pytest.approx(origin)


def test_the_padding_value_is_the_volumes_own_minimum(tmp_path):
    """Not 0. CBCT air sits well below zero, so padding with 0 writes a shell
    of soft-tissue intensity around the head that every later threshold sees.

    Padding only happens when the output's field of view is the LARGER of the
    two, which -- the grid size being kept -- means a target spacing coarser
    than the volume's own. The source here is at 0.1 mm against a 0.3 mm
    target, so the output covers three times the extent and the border really
    is filled in. See the test below for the ordinary case, where the field of
    view shrinks instead and the default pixel value is never reached.
    """
    import SimpleITK as sitk

    write_volume(tmp_path / "in" / "P001_T1.nii.gz", size=(16, 16, 16), spacing=(0.1, 0.1, 0.1))
    # GetArrayFromImage, not GetArrayViewFromImage: a VIEW borrows the image's
    # buffer, so `GetArrayViewFromImage(read_volume(...))` reads freed memory
    # the moment the temporary image is collected. It returns nan when it does,
    # and nan compares unequal to everything -- including to itself -- so the
    # test failed claiming the padding was wrong.
    source = sitk.GetArrayFromImage(read_volume(tmp_path / "in" / "P001_T1.nii.gz"))

    resample.resample_cohort(str(tmp_path / "in"), str(tmp_path / "out"))
    resampled = sitk.GetArrayFromImage(read_volume(tmp_path / "out" / "P001_T1.nii.gz"))

    assert float(resampled.min()) == pytest.approx(float(source.min()), abs=1e-3)
    assert float(resampled.min()) < 0.0


def test_a_finer_target_spacing_narrows_the_field_of_view_around_the_centre(tmp_path):
    """The ordinary case, and the reason centring is not optional.

    The grid size is kept, so a target spacing finer than the volume's own
    covers LESS of the patient, not more -- 0.3 mm over a 24-voxel axis is
    7.2 mm where a 0.6 mm acquisition covered 14.4 mm. Nothing is padded; what
    the output holds is the middle of the input, and it is the middle only
    because the origin was moved to keep the two centres together.
    """
    import SimpleITK as sitk

    write_volume(tmp_path / "in" / "P001_T1.nii.gz", size=(24, 24, 24), spacing=(0.6, 0.6, 0.6))
    source = sitk.GetArrayFromImage(read_volume(tmp_path / "in" / "P001_T1.nii.gz"))

    resample.resample_cohort(str(tmp_path / "in"), str(tmp_path / "out"))
    resampled = sitk.GetArrayFromImage(read_volume(tmp_path / "out" / "P001_T1.nii.gz"))

    # Strictly inside the source's own range: every output voxel is an
    # interpolation of interior values, so neither extreme survives.
    assert float(resampled.min()) > float(source.min())
    assert float(resampled.max()) < float(source.max())


def test_the_interpolation_is_linear_rather_than_nearest_neighbour(tmp_path):
    """A gradient upsampled with nearest-neighbour reproduces the input's own
    values and nothing between them; linear fills in."""
    import SimpleITK as sitk

    write_volume(tmp_path / "in" / "P001_T1.nii.gz", size=(16, 16, 16), spacing=(1.2, 1.2, 1.2))
    source = set(np.unique(
        sitk.GetArrayFromImage(read_volume(tmp_path / "in" / "P001_T1.nii.gz"))
    ).tolist())

    resample.resample_cohort(str(tmp_path / "in"), str(tmp_path / "out"))
    resampled = np.unique(
        sitk.GetArrayFromImage(read_volume(tmp_path / "out" / "P001_T1.nii.gz"))
    )
    assert any(float(value) not in source for value in resampled)


def test_the_folder_tree_is_kept_rather_than_flattened(tmp_path):
    """Upstream derives each output path with
    `file_path.replace(os.path.dirname(file_path), output_folder)`, which puts
    every scan straight into the output folder: two sites holding a scan of the
    same name write to one file and the cohort silently loses a patient."""
    write_volume(tmp_path / "in" / "siteA" / "P001_T1.nii.gz")
    write_volume(tmp_path / "in" / "siteB" / "P001_T1.nii.gz")

    resample.resample_cohort(str(tmp_path / "in"), str(tmp_path / "out"))
    assert tree_of(tmp_path / "out") == [
        os.path.join("siteA", "P001_T1.nii.gz"),
        os.path.join("siteB", "P001_T1.nii.gz"),
    ]


def test_every_volume_spelling_is_resampled(tmp_path):
    """Restricting this to `.nii` dropped `.nrrd` cohorts without a word, and
    every later step then reported "0 file" on a folder that was not empty --
    naming the wrong step while doing it."""
    for name in ("a.nii", "b.nii.gz", "c.nrrd", "d.gipl"):
        write_volume(tmp_path / "in" / name)

    resample.resample_cohort(str(tmp_path / "in"), str(tmp_path / "out"))
    assert tree_of(tmp_path / "out") == ["a.nii", "b.nii.gz", "c.nrrd", "d.gipl"]


def test_a_cohort_with_nothing_in_it_is_refused_by_name(tmp_path):
    (tmp_path / "in").mkdir()
    (tmp_path / "in" / "notes.txt").write_text("hello")

    with pytest.raises(ToolInputError, match="No CBCT volume found"):
        resample.resample_cohort(str(tmp_path / "in"), str(tmp_path / "out"))


def test_one_unreadable_volume_is_reported_and_the_cohort_goes_on(tmp_path):
    cohort(tmp_path, patients=("P001",))
    (tmp_path / "t1" / "P002_T1.nii.gz").write_bytes(b"\x1f\x8b not a volume")

    report = {}
    resample.resample_cohort(str(tmp_path / "t1"), str(tmp_path / "out"), report=report)

    assert tree_of(tmp_path / "out") == ["P001_T1.nii.gz"]
    assert list(report["not_resampled"]) == ["P002_T1.nii.gz"]


def test_a_cohort_where_nothing_could_be_read_is_refused_not_reported_empty(tmp_path):
    """The guard counts what was WRITTEN, not what the walk found. One that
    counted the files it walked past would pass on a cohort where every single
    one failed to read."""
    (tmp_path / "in").mkdir()
    for name in ("P001_T1.nii.gz", "P002_T1.nii.gz"):
        (tmp_path / "in" / name).write_bytes(b"\x1f\x8b not a volume")

    # A RuntimeError, not a 422: SimpleITK's reader failing is not something
    # the error can pin on the caller. The cause travels in the message itself,
    # because the run report is deleted with the job when the run fails.
    with pytest.raises(RuntimeError, match=r"0 of 2 volume\(s\) resampled; most common "
                                           r"failure: RuntimeError: .* \(2 of 2\)"):
        resample.resample_cohort(str(tmp_path / "in"), str(tmp_path / "out"))


def test_an_unreadable_volume_is_logged_by_position_and_cause_never_by_name(tmp_path, caplog):
    cohort(tmp_path, patients=("P001",))
    (tmp_path / "t1" / "P002_T1.nii.gz").write_bytes(b"\x1f\x8b not a volume")

    with caplog.at_level("WARNING", logger="sadt_vface"):
        resample.resample_cohort(str(tmp_path / "t1"), str(tmp_path / "out"))

    failed = [r.getMessage() for r in caplog.records if "resample failed" in r.getMessage()]
    assert len(failed) == 1
    assert "scan 2 of 2" in failed[0]
    assert "RuntimeError" in failed[0]
    assert "P002" not in failed[0]
    assert any("1 of 2 volume(s) resampled" in r.getMessage() and r.levelname == "WARNING"
               for r in caplog.records)


def test_each_scan_is_announced_before_it_is_resampled(tmp_path):
    """So a failure in the middle of the step is reported with the scan it was on."""
    cohort(tmp_path, patients=("P001", "P002"))
    seen = []
    resample.resample_cohort(str(tmp_path / "t1"), str(tmp_path / "out"),
                             on_progress=lambda done, total, message: seen.append(
                                 (done, total, message)))
    assert seen == [(0, 2, "resampling scan 1 of 2"), (1, 2, "resampling scan 2 of 2")]
