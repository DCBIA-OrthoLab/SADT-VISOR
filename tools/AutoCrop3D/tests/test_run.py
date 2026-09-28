"""AutoCrop3D on real volumes and real markup files.

There is no model and no weight here -- a crop is arithmetic on an image grid --
so nothing is stubbed. Every test below builds a SimpleITK volume and a Slicer
`.mrk.json` in `tmp_path` and runs the real pipeline over them.

The tests are grouped by the upstream defect they pin. Each one is named after
the behaviour, not the function, because the value of this file is that it says
what the tool promises.
"""

import gzip
import inspect
import json
import os
import shutil
import sys
import threading
import typing

import numpy as np
import pytest
import SimpleITK as sitk
import vtk

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import sadt_autocrop3d
from sadt_autocrop3d import pipeline
from sadt_autocrop3d.pipeline import patient_key


# ---------------------------------------------------------------------------
# Fixtures built by hand, so every assertion below is against known geometry
# ---------------------------------------------------------------------------

def _volume(path, size=(20, 20, 20), spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0),
            direction=(1, 0, 0, 0, 1, 0, 0, 0, 1), fill=None):
    """A volume whose voxel value encodes its own index, so a crop is checkable.

    `array[z, y, x] = 100*z + 10*y + x` means the value of any voxel says where
    it came from: a crop that lost track of its bounds cannot produce the right
    numbers by accident.
    """
    path = _as_path(path)
    array = np.zeros(size[::-1], dtype=np.int16)
    if fill is None:
        for z in range(size[2]):
            for y in range(size[1]):
                for x in range(size[0]):
                    array[z, y, x] = 100 * z + 10 * y + x
    else:
        array[...] = fill
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    image.SetDirection(direction)
    sitk.WriteImage(image, str(path))
    return path


def _labelmap(path, blobs, size=(20, 20, 20), spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):
    """A label map: `blobs` is {label: (slice_z, slice_y, slice_x)}."""
    path = _as_path(path)
    array = np.zeros(size[::-1], dtype=np.int16)
    for label, box in blobs.items():
        array[box] = label
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    sitk.WriteImage(image, str(path))
    return path


def _roi(path, center, size, coordinate_system="LPS", orientation=None, markup_type="ROI"):
    """A Slicer ROI markup, in the shape Slicer actually writes."""
    path = _as_path(path)
    markup = {"type": markup_type, "center": list(center), "size": list(size)}
    if orientation is not None:
        markup["orientation"] = list(orientation)
    path.write_text(json.dumps({
        "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#",
        "coordinateSystem": coordinate_system,
        "markups": [markup],
    }))
    return path


def _as_path(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _read(path):
    return sitk.ReadImage(str(path))


def _outputs(directory):
    """Every volume the run wrote, by name."""
    found = {}
    for root, _subdirectories, names in os.walk(directory):
        for name in sorted(names):
            if name.endswith((".nii.gz", ".nii", ".nrrd", ".gipl", ".gipl.gz")):
                found[name] = os.path.join(root, name)
    return found


def _report(directory):
    return json.loads((directory / "AutoCrop3D_report.json").read_text())


# ===========================================================================
# Defect 1 -- the scan key and the ROI key were derived by two different rules
# ===========================================================================

def test_a_cohort_whose_identifiers_contain_an_underscore_is_not_silently_skipped(tmp_path):
    """The defect this port exists for.

    Upstream keyed a scan with fifteen chained splits (`PatientA_01`) and its
    ROI with `basename.split('_')[0]` (`PatientA`). The lookup missed, a bare
    `except` logged and continued, and the whole cohort came back as an empty
    output folder behind exit code 0.
    """
    scans, rois, out = tmp_path / "scans", tmp_path / "rois", tmp_path / "out"
    for subject in ("PatientA_01", "PatientA_02"):
        _volume(scans / f"{subject}_Scan.nii.gz")
        _roi(rois / f"{subject}_ROI.mrk.json", center=(5, 5, 5), size=(4, 4, 4))

    sadt_autocrop3d.run(scans=scans, roi=rois, output_dir=out)

    assert sorted(_outputs(out)) == [
        "PatientA_01_Scan_cropped.nii.gz",
        "PatientA_02_Scan_cropped.nii.gz",
    ]


def test_the_scan_and_its_roi_derive_the_same_patient_key():
    """One function, both sides. That equality IS the fix."""
    assert patient_key("PatientA_01_Scan.nii.gz") == patient_key("PatientA_01_ROI.mrk.json")
    assert patient_key("P1_MAND_seg.nii.gz") == patient_key("P1_ROI.mrk.json")
    assert patient_key("sub-004_T1_scan.nrrd") == patient_key("sub-004_T1_ROI.mrk.json")


def test_nothing_matched_raises_instead_of_writing_an_empty_folder(tmp_path):
    """Exit code 0 with an empty output folder is the worst possible answer."""
    scans, rois, out = tmp_path / "scans", tmp_path / "rois", tmp_path / "out"
    _volume(scans / "Alpha_Scan.nii.gz")
    _volume(scans / "Beta_Scan.nii.gz")
    _roi(rois / "Gamma_ROI.mrk.json", center=(5, 5, 5), size=(4, 4, 4))
    _roi(rois / "Delta_ROI.mrk.json", center=(5, 5, 5), size=(4, 4, 4))

    with pytest.raises(ValueError, match="cropped nothing"):
        sadt_autocrop3d.run(scans=scans, roi=rois, output_dir=out)


def test_the_no_match_error_names_the_keys_on_both_sides(tmp_path):
    """A message a user can act on: what the scans are called, what the ROIs
    are called, and therefore which of the two to rename."""
    scans, rois, out = tmp_path / "scans", tmp_path / "rois", tmp_path / "out"
    _volume(scans / "Alpha_Scan.nii.gz")
    _roi(rois / "Gamma_ROI.mrk.json", center=(5, 5, 5), size=(4, 4, 4))
    _roi(rois / "Delta_ROI.mrk.json", center=(5, 5, 5), size=(4, 4, 4))

    with pytest.raises(ValueError) as raised:
        sadt_autocrop3d.run(scans=scans, roi=rois, output_dir=out)

    message = str(raised.value)
    assert "Alpha" in message and "Gamma" in message and "Delta" in message


def test_a_scan_with_no_roi_is_reported_rather_than_logged_and_forgotten(tmp_path):
    scans, rois, out = tmp_path / "scans", tmp_path / "rois", tmp_path / "out"
    _volume(scans / "Alpha_Scan.nii.gz")
    _volume(scans / "Beta_Scan.nii.gz")
    _roi(rois / "Alpha_ROI.mrk.json", center=(5, 5, 5), size=(4, 4, 4))
    _roi(rois / "Zeta_ROI.mrk.json", center=(5, 5, 5), size=(4, 4, 4))

    sadt_autocrop3d.run(scans=scans, roi=rois, output_dir=out)

    without = _report(out)["without_a_roi"]
    assert [entry["scan"] for entry in without] == ["Beta_Scan.nii.gz"]
    assert without[0]["patient"] == "Beta"


# ===========================================================================
# Defect 2 -- the chained splits truncated identifiers and merged timepoints
# ===========================================================================

def test_two_timepoints_stay_two_patients():
    """`.split('_T1')[0]` collapsed both timepoints of a subject into one key,
    so each got whichever ROI the dict happened to end up holding."""
    assert patient_key("P01_T1_Scan.nii.gz") == "P01_T1"
    assert patient_key("P01_T2_Scan.nii.gz") == "P01_T2"
    assert patient_key("P01_T1_Scan.nii.gz") != patient_key("P01_T2_Scan.nii.gz")


def test_two_timepoints_are_cropped_with_their_own_box(tmp_path):
    """The end-to-end form of the same thing, and the one that costs anatomy:
    with one key for both, T1 was cropped with T2's ROI and looked fine."""
    scans, rois, out = tmp_path / "scans", tmp_path / "rois", tmp_path / "out"
    _volume(scans / "P01_T1_Scan.nii.gz")
    _volume(scans / "P01_T2_Scan.nii.gz")
    _roi(rois / "P01_T1_ROI.mrk.json", center=(3.5, 3.5, 3.5), size=(4, 4, 4))
    _roi(rois / "P01_T2_ROI.mrk.json", center=(12.5, 12.5, 12.5), size=(6, 6, 6))

    sadt_autocrop3d.run(scans=scans, roi=rois, output_dir=out)

    written = _outputs(out)
    assert _read(written["P01_T1_Scan_cropped.nii.gz"]).GetSize() == (4, 4, 4)
    assert _read(written["P01_T2_Scan_cropped.nii.gz"]).GetSize() == (6, 6, 6)


def test_an_identifier_containing_or_is_not_truncated():
    """`.split('_OR')[0]` turned `SMITH_ORTHO` into `SMITH`. Tokens are matched
    whole, so `ORTHO` is not the `_Or` suffix a previous run leaves."""
    assert patient_key("SMITH_ORTHO_Scan.nii.gz") == "SMITH_ORTHO"
    assert patient_key("SMITH_ORTHO_ROI.mrk.json") == "SMITH_ORTHO"


def test_the_or_suffix_a_previous_run_leaves_is_still_dropped():
    """The other half of the same rule: `_Or` as a WHOLE token still goes, so
    an oriented scan keys to the subject it belongs to."""
    assert patient_key("SMITH_Or.nii.gz") == "SMITH"
    assert patient_key("SMITH_Or_Scan.nii.gz") == "SMITH"


def test_a_dot_in_the_identifier_does_not_truncate_it():
    """`.split('.')[0]` cut `Patient.01_Scan.nii.gz` down to `Patient`."""
    assert patient_key("Patient.01_Scan.nii.gz") == "Patient.01"
    assert patient_key("Patient.01_ROI.mrk.json") == "Patient.01"


def test_a_leading_anatomy_token_is_kept_because_it_is_the_identifier():
    """A patient really can be called `MAX_01`. Dropping a leading `max` keys
    them as `01` -- upstream's failure reached from the other direction."""
    assert patient_key("MAX_01_scan.nii.gz") == "MAX_01"
    assert patient_key("MAX_01_ROI.mrk.json") == "MAX_01"


def test_an_anatomy_token_after_the_identifier_is_dropped():
    """So that a patient's mandible and maxilla segmentations both find the one
    ROI drawn for that patient."""
    assert patient_key("P1_MAND_seg.nii.gz") == "P1"
    assert patient_key("P1_MAX_seg.nii.gz") == "P1"


def test_a_compound_extension_is_split_off_whole():
    assert patient_key("P1.nii.gz") == "P1"
    assert patient_key("P1.gipl.gz") == "P1"
    assert patient_key("P1.nrrd") == "P1"


# ===========================================================================
# Defect 3 -- `result[patient] = file` overwrote in silence
# ===========================================================================

def test_two_rois_naming_one_patient_are_refused_rather_than_last_one_wins(tmp_path):
    """`P1_MAND_ROI` and `P1_MAX_ROI` both reduce to `P1`. Upstream kept
    whichever came last and cropped with it, which is a plausible-looking
    result of the wrong anatomy. An undecidable pairing is an error."""
    scans, rois, out = tmp_path / "scans", tmp_path / "rois", tmp_path / "out"
    _volume(scans / "P1_scan.nii.gz")
    _roi(rois / "P1_MAND_ROI.mrk.json", center=(5, 5, 5), size=(4, 4, 4))
    _roi(rois / "P1_MAX_ROI.mrk.json", center=(12, 12, 12), size=(6, 6, 6))

    with pytest.raises(ValueError, match="same patient"):
        sadt_autocrop3d.run(scans=scans, roi=rois, output_dir=out)


def test_the_collision_error_names_both_files(tmp_path):
    scans, rois, out = tmp_path / "scans", tmp_path / "rois", tmp_path / "out"
    _volume(scans / "P1_scan.nii.gz")
    _roi(rois / "P1_MAND_ROI.mrk.json", center=(5, 5, 5), size=(4, 4, 4))
    _roi(rois / "P1_MAX_ROI.mrk.json", center=(12, 12, 12), size=(6, 6, 6))

    with pytest.raises(ValueError) as raised:
        sadt_autocrop3d.run(scans=scans, roi=rois, output_dir=out)

    assert "P1_MAND_ROI.mrk.json" in str(raised.value)
    assert "P1_MAX_ROI.mrk.json" in str(raised.value)


def test_the_two_timepoints_of_one_subject_are_not_a_collision(tmp_path):
    """The exact pair upstream collapsed. They must remain two ROIs."""
    scans, rois, out = tmp_path / "scans", tmp_path / "rois", tmp_path / "out"
    _volume(scans / "P01_T1_Scan.nii.gz")
    _volume(scans / "P01_T2_Scan.nii.gz")
    _roi(rois / "P01_T1_ROI.mrk.json", center=(4, 4, 4), size=(4, 4, 4))
    _roi(rois / "P01_T2_ROI.mrk.json", center=(4, 4, 4), size=(4, 4, 4))

    sadt_autocrop3d.run(scans=scans, roi=rois, output_dir=out)

    assert len(_outputs(out)) == 2


# ===========================================================================
# Defect 4 -- the ROI table was built only `if len(ROIList) > 1`
# ===========================================================================

def test_a_folder_holding_one_roi_crops_every_scan(tmp_path):
    """Upstream built its lookup only for two or more ROIs, so a folder holding
    exactly one left `ROI_Path` pointing at the FOLDER and `open()` raised
    `IsADirectoryError` on the first patient."""
    scans, rois, out = tmp_path / "scans", tmp_path / "rois", tmp_path / "out"
    _volume(scans / "A_scan.nii.gz")
    _volume(scans / "B_scan.nii.gz")
    _roi(rois / "the_only_ROI.mrk.json", center=(5.5, 5.5, 5.5), size=(4, 4, 4))

    sadt_autocrop3d.run(scans=scans, roi=rois, output_dir=out)

    assert len(_outputs(out)) == 2


def test_a_single_roi_file_crops_every_scan(tmp_path):
    scans, out = tmp_path / "scans", tmp_path / "out"
    _volume(scans / "A_scan.nii.gz")
    _volume(scans / "B_scan.nii.gz")
    box = _roi(tmp_path / "box.mrk.json", center=(5.5, 5.5, 5.5), size=(4, 4, 4))

    sadt_autocrop3d.run(scans=scans, roi=box, output_dir=out)

    assert len(_outputs(out)) == 2


def test_an_roi_folder_holding_no_markup_is_refused(tmp_path):
    scans, rois, out = tmp_path / "scans", tmp_path / "rois", tmp_path / "out"
    _volume(scans / "A_scan.nii.gz")
    rois.mkdir()
    (rois / "notes.txt").write_text("nothing here")

    with pytest.raises(ValueError, match="No '.mrk.json' ROI found"):
        sadt_autocrop3d.run(scans=scans, roi=rois, output_dir=out)


def test_a_file_that_is_not_a_markup_is_refused_as_the_roi(tmp_path):
    scans, out = tmp_path / "scans", tmp_path / "out"
    _volume(scans / "A_scan.nii.gz")
    other = tmp_path / "box.json"
    other.write_text("{}")

    with pytest.raises(ValueError, match="not a Slicer ROI"):
        sadt_autocrop3d.run(scans=scans, roi=other, output_dir=out)


# ===========================================================================
# Defect 5 -- `.nrrd.gz` discovered but unreadable, and the panel's own filter
# ===========================================================================

def test_an_unreadable_nrrd_gz_costs_one_file_not_the_batch(tmp_path):
    """NRRD compresses inside the file, so no ITK reader opens a gzipped one.
    Upstream read the volume OUTSIDE its try block, so one such file ended the
    whole batch before a single scan had been cropped."""
    scans, out = tmp_path / "scans", tmp_path / "out"
    _volume(scans / "A_scan.nii.gz")
    plain = _volume(scans / "tmp.nrrd")
    with open(plain, "rb") as source, gzip.open(scans / "B_scan.nrrd.gz", "wb") as target:
        shutil.copyfileobj(source, target)
    plain.unlink()
    _roi(tmp_path / "box.mrk.json", center=(5.5, 5.5, 5.5), size=(4, 4, 4))

    sadt_autocrop3d.run(scans=scans, roi=tmp_path / "box.mrk.json", output_dir=out)

    assert "A_scan_cropped.nii.gz" in _outputs(out)


def test_the_nrrd_gz_failure_is_named_rather_than_counted_as_a_success(tmp_path):
    """Upstream's write sat in a bare `except:` and the progress index advanced
    unconditionally, so the user was shown 'Scan(s) cropped with success' for a
    run that wrote nothing for that patient."""
    scans, out = tmp_path / "scans", tmp_path / "out"
    _volume(scans / "A_scan.nii.gz")
    (scans / "B_scan.nrrd.gz").write_bytes(b"\x1f\x8b not a volume")
    _roi(tmp_path / "box.mrk.json", center=(5.5, 5.5, 5.5), size=(4, 4, 4))

    sadt_autocrop3d.run(scans=scans, roi=tmp_path / "box.mrk.json", output_dir=out)

    report = _report(out)
    assert report["summary"]["cropped"] == 1
    assert report["summary"]["failed"] == 1
    assert "B_scan.nrrd.gz" in report["failed"]


def test_a_folder_of_plain_nii_is_accepted(tmp_path):
    """The panel searched only `.nii.gz`, `.nrrd.gz` and `.gipl.gz` and REFUSED
    a folder of plain volumes that the CLI behind it reads perfectly well."""
    scans, out = tmp_path / "scans", tmp_path / "out"
    _volume(scans / "A_scan.nii")
    _roi(tmp_path / "box.mrk.json", center=(5.5, 5.5, 5.5), size=(4, 4, 4))

    sadt_autocrop3d.run(scans=scans, roi=tmp_path / "box.mrk.json", output_dir=out)

    assert "A_scan_cropped.nii" in _outputs(out)


def test_a_folder_of_plain_nrrd_is_accepted(tmp_path):
    scans, out = tmp_path / "scans", tmp_path / "out"
    _volume(scans / "A_scan.nrrd")
    _roi(tmp_path / "box.mrk.json", center=(5.5, 5.5, 5.5), size=(4, 4, 4))

    sadt_autocrop3d.run(scans=scans, roi=tmp_path / "box.mrk.json", output_dir=out)

    assert "A_scan_cropped.nrrd" in _outputs(out)


def test_a_gipl_volume_is_accepted(tmp_path):
    scans, out = tmp_path / "scans", tmp_path / "out"
    _volume(scans / "A_scan.gipl.gz")
    _roi(tmp_path / "box.mrk.json", center=(5.5, 5.5, 5.5), size=(4, 4, 4))

    sadt_autocrop3d.run(scans=scans, roi=tmp_path / "box.mrk.json", output_dir=out)

    assert "A_scan_cropped.gipl.gz" in _outputs(out)


def test_a_folder_with_no_volume_at_all_is_refused_by_name(tmp_path):
    scans, out = tmp_path / "scans", tmp_path / "out"
    scans.mkdir()
    (scans / "readme.txt").write_text("no volumes")
    _roi(tmp_path / "box.mrk.json", center=(5, 5, 5), size=(4, 4, 4))

    with pytest.raises(ValueError, match="No scan found"):
        sadt_autocrop3d.run(scans=scans, roi=tmp_path / "box.mrk.json", output_dir=out)


# ===========================================================================
# Defect 6 -- the output path was built with `str.replace`
# ===========================================================================

def test_a_single_file_input_writes_exactly_one_file(tmp_path):
    """In single-file mode `os.path.relpath(file, file)` is `"."`, and
    `os.path.basename(".")` is `"."`, so upstream's
    `.replace(basename, filename)` substituted the file name for EVERY DOT in
    the whole output path -- then `os.makedirs` created the resulting tree."""
    scan = _volume(tmp_path / "in" / "A_scan.nii.gz")
    box = _roi(tmp_path / "box.mrk.json", center=(5.5, 5.5, 5.5), size=(4, 4, 4))
    out = tmp_path / "out"

    sadt_autocrop3d.run(scans=scan, roi=box, output_dir=out)

    assert sorted(path.name for path in out.iterdir()) == [
        "A_scan_cropped.nii.gz", "AutoCrop3D_report.json",
    ]


def test_a_dotted_output_directory_is_not_rewritten(tmp_path):
    """The same `str.replace` ate any dot in the OUTPUT path too, so a
    perfectly ordinary `/data/study.v2/out` became a garbage tree."""
    scan = _volume(tmp_path / "in" / "A_scan.nii.gz")
    box = _roi(tmp_path / "box.mrk.json", center=(5.5, 5.5, 5.5), size=(4, 4, 4))
    out = tmp_path / "study.v2" / "out"

    sadt_autocrop3d.run(scans=scan, roi=box, output_dir=out)

    assert (out / "A_scan_cropped.nii.gz").is_file()
    assert not (tmp_path / "study").exists()


def test_a_folder_tree_is_mirrored_in_the_output(tmp_path):
    scans, out = tmp_path / "scans", tmp_path / "out"
    _volume(scans / "siteA" / "A_scan.nii.gz")
    _volume(scans / "siteB" / "deeper" / "B_scan.nii.gz")
    box = _roi(tmp_path / "box.mrk.json", center=(5.5, 5.5, 5.5), size=(4, 4, 4))

    sadt_autocrop3d.run(scans=scans, roi=box, output_dir=out)

    assert (out / "siteA" / "A_scan_cropped.nii.gz").is_file()
    assert (out / "siteB" / "deeper" / "B_scan_cropped.nii.gz").is_file()


def test_two_patients_with_the_same_base_name_do_not_overwrite_each_other(tmp_path):
    """The tree is mirrored, so `siteA/scan.nii.gz` and `siteB/scan.nii.gz`
    stay two files rather than one."""
    scans, out = tmp_path / "scans", tmp_path / "out"
    _volume(scans / "siteA" / "scan.nii.gz")
    _volume(scans / "siteB" / "scan.nii.gz")
    box = _roi(tmp_path / "box.mrk.json", center=(5.5, 5.5, 5.5), size=(4, 4, 4))

    sadt_autocrop3d.run(scans=scans, roi=box, output_dir=out)

    assert (out / "siteA" / "scan_cropped.nii.gz").is_file()
    assert (out / "siteB" / "scan_cropped.nii.gz").is_file()


def test_the_output_keeps_the_compound_extension(tmp_path):
    assert pipeline.output_name("A.nii.gz", "cropped") == "A_cropped.nii.gz"
    assert pipeline.output_name("A.gipl.gz", "cropped") == "A_cropped.gipl.gz"
    assert pipeline.output_name("study.v2.nrrd", "cropped") == "study.v2_cropped.nrrd"


def test_an_empty_suffix_leaves_no_trailing_underscore(tmp_path):
    scan = _volume(tmp_path / "in" / "A_scan.nii.gz")
    box = _roi(tmp_path / "box.mrk.json", center=(5.5, 5.5, 5.5), size=(4, 4, 4))
    out = tmp_path / "out"

    sadt_autocrop3d.run(scans=scan, roi=box, output_dir=out, suffix="")

    assert (out / "A_scan.nii.gz").is_file()


# ===========================================================================
# Defect 7 -- a bare `except:` around the write, then `index += 1` regardless
# ===========================================================================

def test_a_failed_scan_is_not_counted_as_a_success(tmp_path):
    scans, out = tmp_path / "scans", tmp_path / "out"
    _volume(scans / "A_scan.nii.gz")
    (scans / "B_scan.nii.gz").write_bytes(b"not a volume")
    box = _roi(tmp_path / "box.mrk.json", center=(5.5, 5.5, 5.5), size=(4, 4, 4))

    sadt_autocrop3d.run(scans=scans, roi=box, output_dir=out)

    report = _report(out)
    assert report["summary"]["scans_found"] == 2
    assert report["summary"]["cropped"] == 1
    assert list(report["failed"]) == ["B_scan.nii.gz"]


def test_a_run_where_every_scan_failed_raises(tmp_path):
    scans, out = tmp_path / "scans", tmp_path / "out"
    (scans / "A_scan.nii.gz").parent.mkdir(parents=True)
    (scans / "A_scan.nii.gz").write_bytes(b"not a volume")
    box = _roi(tmp_path / "box.mrk.json", center=(5.5, 5.5, 5.5), size=(4, 4, 4))

    with pytest.raises(ValueError, match="all 1 scan"):
        sadt_autocrop3d.run(scans=scans, roi=box, output_dir=out)


def test_one_bad_scan_does_not_stop_the_others(tmp_path):
    scans, out = tmp_path / "scans", tmp_path / "out"
    (scans / "aaa_bad.nii.gz").parent.mkdir(parents=True)
    (scans / "aaa_bad.nii.gz").write_bytes(b"not a volume")
    _volume(scans / "bbb_scan.nii.gz")
    _volume(scans / "ccc_scan.nii.gz")
    box = _roi(tmp_path / "box.mrk.json", center=(5.5, 5.5, 5.5), size=(4, 4, 4))

    sadt_autocrop3d.run(scans=scans, roi=box, output_dir=out)

    assert len(_outputs(out)) == 2


def test_the_failure_message_says_which_exception_and_why(tmp_path):
    scans, out = tmp_path / "scans", tmp_path / "out"
    _volume(scans / "A_scan.nii.gz")
    (scans / "B_scan.nii.gz").write_bytes(b"not a volume")
    box = _roi(tmp_path / "box.mrk.json", center=(5.5, 5.5, 5.5), size=(4, 4, 4))

    sadt_autocrop3d.run(scans=scans, roi=box, output_dir=out)

    assert "RuntimeError" in _report(out)["failed"]["B_scan.nii.gz"]


# ===========================================================================
# Defect 8 -- `"image_padded.nii.gz"`, a fixed relative path in the CWD
# ===========================================================================

def test_the_padded_volume_is_not_written_into_the_working_directory(tmp_path, monkeypatch):
    """Upstream wrote its temporary padded volume as a RELATIVE path, so it
    landed wherever the process happened to be -- the server's source tree."""
    workdir = tmp_path / "cwd"
    workdir.mkdir()
    monkeypatch.chdir(workdir)

    scans, out = tmp_path / "scans", tmp_path / "out"
    _labelmap(scans / "A_seg.nii.gz", {1: (np.s_[3:8], np.s_[3:8], np.s_[3:8])})
    box = _roi(tmp_path / "box.mrk.json", center=(9.5, 9.5, 9.5), size=(18, 18, 18))

    sadt_autocrop3d.run(scans=scans, roi=box, output_dir=out)

    assert list(workdir.iterdir()) == []


def test_the_scratch_volume_is_removed_when_the_surface_succeeds(tmp_path):
    scans, out = tmp_path / "scans", tmp_path / "out"
    _labelmap(scans / "A_seg.nii.gz", {1: (np.s_[3:8], np.s_[3:8], np.s_[3:8])})
    box = _roi(tmp_path / "box.mrk.json", center=(9.5, 9.5, 9.5), size=(18, 18, 18))

    sadt_autocrop3d.run(scans=scans, roi=box, output_dir=out)

    assert not any(name.startswith("padded") for name in os.listdir(out))


def test_the_scratch_volume_is_removed_even_when_the_surface_fails(tmp_path, monkeypatch):
    """Upstream's `os.remove` sat between the colouring loop and the write, so
    any exception in either left the temporary volume behind -- on a server,
    forever, holding a patient's anatomy."""
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    image = sitk.ReadImage(str(_labelmap(tmp_path / "s.nii.gz",
                                         {1: (np.s_[3:8], np.s_[3:8], np.s_[3:8])})))

    def boom(*_args, **_kwargs):
        raise RuntimeError("VTK gave up")

    monkeypatch.setattr(pipeline, "_color_by_label", boom)

    with pytest.raises(RuntimeError, match="VTK gave up"):
        pipeline.write_surface(image, str(tmp_path / "s.vtk"), str(scratch),
                               padding_mm=2.0, smoothing_iterations=0)

    assert list(scratch.iterdir()) == []


def test_two_concurrent_runs_do_not_share_a_temporary_file(tmp_path):
    """One fixed file name in the CWD means two requests overwrite each other's
    padded volume. Each run gets its own scratch directory now."""
    box = _roi(tmp_path / "box.mrk.json", center=(9.5, 9.5, 9.5), size=(18, 18, 18))
    jobs = []
    for index, extent in enumerate((np.s_[2:6], np.s_[10:16])):
        scans = tmp_path / f"scans{index}"
        _labelmap(scans / f"P{index}_seg.nii.gz", {index + 1: (extent, extent, extent)})
        jobs.append((scans, tmp_path / f"out{index}"))

    errors = []

    def go(scans, out):
        try:
            sadt_autocrop3d.run(scans=scans, roi=box, output_dir=out)
        except Exception as error:  # pragma: no cover - reported below
            errors.append(error)

    threads = [threading.Thread(target=go, args=job) for job in jobs]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert _report(jobs[0][1])["cases"]["P0_seg.nii.gz"]["surface_labels"] == [1]
    assert _report(jobs[1][1])["cases"]["P1_seg.nii.gz"]["surface_labels"] == [2]


# ===========================================================================
# Defect 9 -- LABEL_COLORS keyed 1..6 and indexed with `np.max(img_arr)`
# ===========================================================================

def test_a_segmentation_with_more_than_six_labels_produces_a_surface(tmp_path):
    """`LABEL_COLORS[np.max(img_arr)]` raised `KeyError` for any label above 6,
    and the caller's `except: pass` turned that into a missing file."""
    image = sitk.ReadImage(str(_labelmap(tmp_path / "s.nii.gz", {
        label: (np.s_[label:label + 2], np.s_[3:6], np.s_[3:6]) for label in range(1, 10)
    })))
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    labels = pipeline.write_surface(image, str(tmp_path / "s.vtk"), str(scratch),
                                    padding_mm=2.0, smoothing_iterations=0)

    assert labels == list(range(1, 10))
    assert (tmp_path / "s.vtk").is_file()


def test_an_empty_crop_produces_no_surface_instead_of_a_key_error(tmp_path):
    """`np.max` of an all-zero crop is 0, which is not in the table either."""
    image = sitk.ReadImage(str(_labelmap(tmp_path / "s.nii.gz", {})))
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    labels = pipeline.write_surface(image, str(tmp_path / "s.vtk"), str(scratch),
                                    padding_mm=2.0, smoothing_iterations=0)

    assert labels == []
    assert not (tmp_path / "s.vtk").exists()


def test_an_empty_crop_is_reported_rather_than_left_as_a_missing_file(tmp_path):
    scans, out = tmp_path / "scans", tmp_path / "out"
    _labelmap(scans / "A_seg.nii.gz", {1: (np.s_[15:18], np.s_[15:18], np.s_[15:18])})
    box = _roi(tmp_path / "box.mrk.json", center=(2.5, 2.5, 2.5), size=(4, 4, 4))

    sadt_autocrop3d.run(scans=scans, roi=box, output_dir=out)

    entry = _report(out)["cases"]["A_seg.nii.gz"]
    assert entry["surface"] is None and entry["surface_labels"] == []


def test_each_label_gets_its_own_colour(tmp_path):
    """Upstream applied the single maximum label's colour to EVERY cell, so the
    per-label colouring its own table implies never happened."""
    image = sitk.ReadImage(str(_labelmap(tmp_path / "s.nii.gz", {
        1: (np.s_[3:6], np.s_[3:6], np.s_[3:6]),
        4: (np.s_[12:16], np.s_[12:16], np.s_[12:16]),
    })))
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    destination = tmp_path / "s.vtk"

    pipeline.write_surface(image, str(destination), str(scratch),
                           padding_mm=2.0, smoothing_iterations=0)

    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(str(destination))
    reader.ReadAllScalarsOn()
    reader.Update()
    colors = reader.GetOutput().GetCellData().GetScalars()
    seen = {tuple(int(v) for v in colors.GetTuple3(i))
            for i in range(reader.GetOutput().GetNumberOfCells())}

    assert seen == {pipeline.LABEL_COLORS[1], pipeline.LABEL_COLORS[4]}


def test_the_first_six_labels_keep_upstreams_colours():
    """A five-structure segmentation still comes back the colours a clinician
    recognises; only the labels upstream could not colour at all are new."""
    assert pipeline.color_for_label(1) == (216, 101, 79)
    assert pipeline.color_for_label(2) == (128, 174, 128)
    assert pipeline.color_for_label(3) == (0, 0, 0)
    assert pipeline.color_for_label(4) == (230, 220, 70)
    assert pipeline.color_for_label(5) == (111, 184, 210)
    assert pipeline.color_for_label(6) == (172, 122, 101)


def test_a_colour_beyond_the_table_is_deterministic_and_distinct():
    """The same label is the same colour in every run and in every patient,
    which is what makes two timepoints comparable by eye."""
    generated = [pipeline.color_for_label(label) for label in range(7, 40)]
    assert generated == [pipeline.color_for_label(label) for label in range(7, 40)]
    assert len(set(generated)) == len(generated)
    assert all(0 <= channel <= 255 for color in generated for channel in color)


def test_present_labels_ignores_the_background(tmp_path):
    """Computed at upstream lines 29-32 and then never used -- `np.max` was
    reached for instead."""
    image = sitk.ReadImage(str(_labelmap(tmp_path / "s.nii.gz", {
        2: (np.s_[3:6], np.s_[3:6], np.s_[3:6]),
        7: (np.s_[10:13], np.s_[10:13], np.s_[10:13]),
    })))
    assert pipeline.present_labels(image) == [2, 7]


# ===========================================================================
# Defect 10 -- `if "seg" in ScanOutPath.lower()`, on the whole output path
# ===========================================================================

def test_an_output_folder_named_segmentations_does_not_surface_every_scan(tmp_path):
    """The substring test was on the OUTPUT PATH, so `/data/Segmentations/`
    sent every cropped CBCT through marching cubes."""
    scans, out = tmp_path / "scans", tmp_path / "Segmentations"
    _volume(scans / "A_scan.nii.gz")
    box = _roi(tmp_path / "box.mrk.json", center=(9.5, 9.5, 9.5), size=(18, 18, 18))

    sadt_autocrop3d.run(scans=scans, roi=box, output_dir=out)

    assert not list(out.glob("**/*.vtk"))


def test_a_file_whose_own_name_says_seg_gets_a_surface(tmp_path):
    scans, out = tmp_path / "scans", tmp_path / "out"
    _labelmap(scans / "A_seg.nii.gz", {1: (np.s_[3:8], np.s_[3:8], np.s_[3:8])})
    box = _roi(tmp_path / "box.mrk.json", center=(9.5, 9.5, 9.5), size=(18, 18, 18))

    sadt_autocrop3d.run(scans=scans, roi=box, output_dir=out)

    assert (out / "A_seg_cropped_vtk.vtk").is_file()


def test_a_segmentation_named_after_its_anatomy_can_still_be_surfaced(tmp_path):
    """`Mandible.nii.gz` says nothing about being a segmentation, so upstream
    gave it no surface and offered no way to ask for one. `surfaces="all"` is
    that way."""
    scans, out = tmp_path / "scans", tmp_path / "out"
    _labelmap(scans / "Mandible.nii.gz", {1: (np.s_[3:8], np.s_[3:8], np.s_[3:8])})
    box = _roi(tmp_path / "box.mrk.json", center=(9.5, 9.5, 9.5), size=(18, 18, 18))

    sadt_autocrop3d.run(scans=scans, roi=box, output_dir=out, surfaces="all")

    assert (out / "Mandible_cropped_vtk.vtk").is_file()


def test_surfaces_none_writes_no_surface_at_all(tmp_path):
    scans, out = tmp_path / "scans", tmp_path / "out"
    _labelmap(scans / "A_seg.nii.gz", {1: (np.s_[3:8], np.s_[3:8], np.s_[3:8])})
    box = _roi(tmp_path / "box.mrk.json", center=(9.5, 9.5, 9.5), size=(18, 18, 18))

    sadt_autocrop3d.run(scans=scans, roi=box, output_dir=out, surfaces="none")

    assert not list(out.glob("**/*.vtk"))


def test_an_unknown_surfaces_value_is_refused(tmp_path):
    scans, out = tmp_path / "scans", tmp_path / "out"
    _volume(scans / "A_scan.nii.gz")
    box = _roi(tmp_path / "box.mrk.json", center=(9.5, 9.5, 9.5), size=(18, 18, 18))

    with pytest.raises(ValueError, match="cropped nothing"):
        sadt_autocrop3d.run(scans=scans, roi=box, output_dir=out, surfaces="sometimes")
    assert "surfaces must be one of" in _report(out)["failed"]["A_scan.nii.gz"]


def test_the_segmentation_test_is_on_a_whole_token_of_the_stem():
    assert pipeline.is_segmentation_name("/data/x/A_seg.nii.gz")
    assert pipeline.is_segmentation_name("A_Segmentation.nii.gz")
    assert pipeline.is_segmentation_name("A_MASK.nii.gz")
    assert not pipeline.is_segmentation_name("/data/Segmentations/A_scan.nii.gz")
    assert not pipeline.is_segmentation_name("Segovia_scan.nii.gz")


# ===========================================================================
# Defect 11 -- `if originalSize == 'True'`, a string-typed boolean
# ===========================================================================

def test_keep_original_size_is_a_real_bool(tmp_path):
    """`"true"`, `"1"` and a real `True` all took the ELSE branch upstream,
    which is the single likeliest mis-wiring when a server calls the tool."""
    assert inspect.signature(sadt_autocrop3d.run).parameters["keep_original_size"].annotation is bool


def test_keep_original_size_true_returns_the_original_geometry(tmp_path):
    scan = _volume(tmp_path / "in" / "A_scan.nii.gz", spacing=(0.4, 0.5, 0.6),
                   origin=(5.0, -3.0, 2.0))
    box = _roi(tmp_path / "box.mrk.json", center=(6.0, -1.5, 3.8), size=(1.6, 2.0, 2.4))
    out = tmp_path / "out"

    sadt_autocrop3d.run(scans=scan, roi=box, output_dir=out, keep_original_size=True)

    original, cropped = _read(scan), _read(out / "A_scan_cropped.nii.gz")
    assert cropped.GetSize() == original.GetSize()
    assert cropped.GetOrigin() == original.GetOrigin()
    assert cropped.GetSpacing() == original.GetSpacing()


def test_keep_original_size_false_returns_only_the_box(tmp_path):
    scan = _volume(tmp_path / "in" / "A_scan.nii.gz")
    box = _roi(tmp_path / "box.mrk.json", center=(5.5, 5.5, 5.5), size=(4, 4, 4))
    out = tmp_path / "out"

    sadt_autocrop3d.run(scans=scan, roi=box, output_dir=out, keep_original_size=False)

    assert _read(out / "A_scan_cropped.nii.gz").GetSize() == (4, 4, 4)


# ===========================================================================
# Defect 12 -- the padding was in VOXELS, the smoothing constants hardcoded
# ===========================================================================

def test_the_surface_padding_is_millimetres_not_voxels(tmp_path):
    """A fixed `[50, 50, 10]` voxels is 25 mm on a 0.5 mm CBCT and 8 mm on a
    0.16 mm one -- the same number meaning two different margins."""
    image = sitk.ReadImage(str(_labelmap(tmp_path / "a.nii.gz", {1: (np.s_[2:4],) * 3},
                                         spacing=(0.5, 0.5, 0.5))))
    fine = sitk.ReadImage(str(_labelmap(tmp_path / "b.nii.gz", {1: (np.s_[2:4],) * 3},
                                        spacing=(0.25, 0.25, 0.25))))

    assert pipeline.pad_voxels(image, 5.0) == [10, 10, 10]
    assert pipeline.pad_voxels(fine, 5.0) == [20, 20, 20]


def test_the_padding_is_never_less_than_one_voxel(tmp_path):
    """The margin exists to give marching cubes a shell of background to close
    the surface against, so zero is not a useful answer."""
    image = sitk.ReadImage(str(_labelmap(tmp_path / "a.nii.gz", {1: (np.s_[2:4],) * 3},
                                         spacing=(2.0, 2.0, 2.0))))
    assert pipeline.pad_voxels(image, 0.0) == [1, 1, 1]


def test_a_negative_padding_is_refused(tmp_path):
    image = sitk.ReadImage(str(_labelmap(tmp_path / "a.nii.gz", {1: (np.s_[2:4],) * 3})))
    with pytest.raises(ValueError, match="must not be negative"):
        pipeline.pad_voxels(image, -1.0)


def test_the_padding_closes_a_structure_touching_the_edge_of_the_crop(tmp_path):
    """What the margin is FOR: without it the isosurface is open where the
    label runs into the boundary and the mesh has a hole."""
    image = sitk.ReadImage(str(_labelmap(tmp_path / "s.nii.gz",
                                         {1: (np.s_[0:20], np.s_[0:20], np.s_[0:20])},
                                         size=(6, 6, 6))))
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    destination = tmp_path / "s.vtk"

    pipeline.write_surface(image, str(destination), str(scratch),
                           padding_mm=2.0, smoothing_iterations=0)

    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(str(destination))
    reader.Update()
    edges = vtk.vtkFeatureEdges()
    edges.SetInputData(reader.GetOutput())
    edges.BoundaryEdgesOn()
    edges.FeatureEdgesOff()
    edges.NonManifoldEdgesOff()
    edges.ManifoldEdgesOff()
    edges.Update()

    assert edges.GetOutput().GetNumberOfCells() == 0


def test_the_smoothing_iterations_are_an_argument(tmp_path):
    """Hardcoded at 5 upstream. Zero keeps the raw marching-cubes surface,
    which is what someone checks against when a thin structure loses detail."""
    parameter = inspect.signature(sadt_autocrop3d.run).parameters["surface_smoothing_iterations"]
    assert parameter.annotation is int and parameter.default == 5


def test_smoothing_moves_the_surface_and_not_smoothing_does_not(tmp_path):
    image = sitk.ReadImage(str(_labelmap(tmp_path / "s.nii.gz",
                                         {1: (np.s_[4:10], np.s_[4:10], np.s_[4:10])})))
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    bounds = []
    for iterations in (0, 30):
        destination = tmp_path / f"s{iterations}.vtk"
        pipeline.write_surface(image, str(destination), str(scratch),
                               padding_mm=2.0, smoothing_iterations=iterations)
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(str(destination))
        reader.Update()
        bounds.append(reader.GetOutput().GetBounds())

    assert bounds[0] != bounds[1]


def test_a_negative_smoothing_count_is_refused(tmp_path):
    scan = _volume(tmp_path / "in" / "A_scan.nii.gz")
    box = _roi(tmp_path / "box.mrk.json", center=(5.5, 5.5, 5.5), size=(4, 4, 4))

    with pytest.raises(ValueError, match="must not be negative"):
        sadt_autocrop3d.run(scans=scan, roi=box, output_dir=tmp_path / "out",
                            surface_smoothing_iterations=-1)


def test_the_contour_values_come_from_the_labels_present(tmp_path):
    """`GenerateValues(100, 1, 100)` computed a hundred contours for a
    five-label map and contoured NOTHING above label 100."""
    image = sitk.ReadImage(str(_labelmap(tmp_path / "s.nii.gz",
                                         {250: (np.s_[3:8], np.s_[3:8], np.s_[3:8])})))
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    labels = pipeline.write_surface(image, str(tmp_path / "s.vtk"), str(scratch),
                                    padding_mm=2.0, smoothing_iterations=0)

    assert labels == [250]
    assert (tmp_path / "s.vtk").stat().st_size > 0


# ===========================================================================
# Defect 13 -- the volume read and the ROI read were outside the try block
# ===========================================================================

def test_a_malformed_roi_file_is_a_message_not_a_traceback(tmp_path):
    bad = tmp_path / "box.mrk.json"
    bad.write_text("{not json")

    with pytest.raises(ValueError, match="not valid JSON"):
        pipeline.read_roi(str(bad))


def test_a_markups_file_that_is_not_an_roi_is_refused_by_name(tmp_path):
    """A landmark file is also a `.mrk.json`, and it has no box."""
    path = tmp_path / "P1_lm.mrk.json"
    path.write_text(json.dumps({"markups": [{"type": "Fiducial", "controlPoints": []}]}))

    with pytest.raises(ValueError, match="no ROI box"):
        pipeline.read_roi(str(path))


def test_an_empty_markups_document_is_refused_by_name(tmp_path):
    path = tmp_path / "P1.mrk.json"
    path.write_text(json.dumps({"markups": []}))

    with pytest.raises(ValueError, match="holds no markup"):
        pipeline.read_roi(str(path))


def test_one_unreadable_roi_costs_one_patient_not_the_batch(tmp_path):
    scans, rois, out = tmp_path / "scans", tmp_path / "rois", tmp_path / "out"
    _volume(scans / "A_scan.nii.gz")
    _volume(scans / "B_scan.nii.gz")
    _roi(rois / "A_ROI.mrk.json", center=(5.5, 5.5, 5.5), size=(4, 4, 4))
    (rois / "B_ROI.mrk.json").write_text("{not json")

    sadt_autocrop3d.run(scans=scans, roi=rois, output_dir=out)

    report = _report(out)
    assert report["summary"]["cropped"] == 1
    assert "not valid JSON" in report["failed"]["B_scan.nii.gz"]


# ===========================================================================
# The geometry -- what a crop must preserve
# ===========================================================================

def test_the_crop_keeps_the_origin_spacing_and_direction(tmp_path):
    """A crop whose geometry is wrong opens beside its scan and floats."""
    scan = _volume(tmp_path / "in" / "A_scan.nii.gz", spacing=(0.4, 0.5, 0.6),
                   origin=(5.0, -3.0, 2.0), direction=(-1, 0, 0, 0, -1, 0, 0, 0, 1))
    image = _read(scan)
    lower, upper = (3, 4, 5), (9, 11, 13)
    cropped = pipeline.crop(image, lower, upper)

    assert cropped.GetSpacing() == image.GetSpacing()
    assert cropped.GetDirection() == image.GetDirection()
    assert cropped.GetOrigin() == image.TransformIndexToPhysicalPoint(lower)


def test_the_crop_holds_exactly_the_voxels_the_roi_covers(tmp_path):
    """The voxel values encode their own index, so this is checkable rather
    than plausible."""
    scan = _volume(tmp_path / "in" / "A_scan.nii.gz")
    image = _read(scan)
    cropped = pipeline.crop(image, (3, 4, 5), (7, 9, 11))

    array = sitk.GetArrayFromImage(cropped)
    assert array.shape == (6, 5, 4)
    assert array[0, 0, 0] == 100 * 5 + 10 * 4 + 3
    assert array[-1, -1, -1] == 100 * 10 + 10 * 8 + 6


def test_the_crop_bounds_come_from_the_roi_corners(tmp_path):
    scan = _volume(tmp_path / "in" / "A_scan.nii.gz", spacing=(0.5, 0.5, 0.5),
                   origin=(1.0, 1.0, 1.0))
    image = _read(scan)
    box = pipeline.read_roi(str(_roi(tmp_path / "box.mrk.json",
                                     center=(3.0, 3.0, 3.0), size=(2.0, 2.0, 2.0))))

    lower, upper, clamped = pipeline.crop_bounds(image, box)

    assert lower == [2, 2, 2] and upper == [6, 6, 6] and not clamped


def test_the_crop_bounds_are_truncated_the_way_upstream_truncates(tmp_path):
    """`.astype(int)` truncates toward zero; it does not floor and does not
    round. Kept, because changing it moves every boundary voxel of every result
    this tool has produced."""
    scan = _volume(tmp_path / "in" / "A_scan.nii.gz")
    image = _read(scan)
    box = pipeline.read_roi(str(_roi(tmp_path / "box.mrk.json",
                                     center=(5.0, 5.0, 5.0), size=(3.4, 3.4, 3.4))))

    lower, upper, _clamped = pipeline.crop_bounds(image, box)

    # continuous indices are 3.3 and 6.7, truncated to 3 and 6.
    assert lower == [3, 3, 3] and upper == [6, 6, 6]


def test_an_inverted_box_is_normalised_rather_than_producing_an_empty_crop(tmp_path):
    """A direction matrix with a negative axis maps the lower physical corner
    to the HIGHER index. Upstream swapped them and so does this."""
    scan = _volume(tmp_path / "in" / "A_scan.nii.gz",
                   direction=(-1, 0, 0, 0, -1, 0, 0, 0, 1), origin=(19.0, 19.0, 0.0))
    image = _read(scan)
    box = pipeline.read_roi(str(_roi(tmp_path / "box.mrk.json",
                                     center=(9.5, 9.5, 9.5), size=(4, 4, 4))))

    lower, upper, _clamped = pipeline.crop_bounds(image, box)

    assert all(lower[axis] < upper[axis] for axis in range(3))


def test_an_roi_larger_than_the_volume_is_clamped_and_says_so(tmp_path):
    scan = _volume(tmp_path / "in" / "A_scan.nii.gz")
    box = _roi(tmp_path / "box.mrk.json", center=(9.5, 9.5, 9.5), size=(400, 400, 400))
    out = tmp_path / "out"

    sadt_autocrop3d.run(scans=scan, roi=box, output_dir=out)

    entry = _report(out)["cases"]["A_scan.nii.gz"]
    assert entry["clamped_to_the_volume"] is True
    assert entry["size"] == [20, 20, 20]


def test_an_roi_that_misses_the_volume_is_refused_with_a_message(tmp_path):
    """Upstream sliced an empty range, wrote it inside a bare `except:` and
    counted the patient as a success."""
    scan = _volume(tmp_path / "in" / "A_scan.nii.gz")
    box = _roi(tmp_path / "box.mrk.json", center=(500.0, 500.0, 500.0), size=(4, 4, 4))
    out = tmp_path / "out"

    with pytest.raises(ValueError, match="cropped nothing"):
        sadt_autocrop3d.run(scans=scan, roi=box, output_dir=out)
    assert "does not overlap" in _report(out)["failed"]["A_scan.nii.gz"]


def test_the_repadded_crop_lands_the_box_back_at_the_right_voxel(tmp_path):
    """The assertion `keep_original_size` exists for: everything outside the box
    is zero, everything inside is exactly where it was."""
    scan = _volume(tmp_path / "in" / "A_scan.nii.gz")
    image = _read(scan)
    lower, upper = (3, 4, 5), (7, 9, 11)
    padded = pipeline.repad(image, pipeline.crop(image, lower, upper), lower)

    original = sitk.GetArrayFromImage(image)
    result = sitk.GetArrayFromImage(padded)
    inside = np.s_[lower[2]:upper[2], lower[1]:upper[1], lower[0]:upper[0]]

    assert np.array_equal(result[inside], original[inside])
    blanked = result.copy()
    blanked[inside] = 0
    assert not blanked.any(), "voxels outside the ROI survived the re-pad"


def test_the_repadded_crop_keeps_the_original_geometry(tmp_path):
    scan = _volume(tmp_path / "in" / "A_scan.nii.gz", spacing=(0.4, 0.5, 0.6),
                   origin=(5.0, -3.0, 2.0), direction=(-1, 0, 0, 0, -1, 0, 0, 0, 1))
    image = _read(scan)
    padded = pipeline.repad(image, pipeline.crop(image, (3, 4, 5), (7, 9, 11)), (3, 4, 5))

    assert padded.GetSize() == image.GetSize()
    assert padded.GetOrigin() == image.GetOrigin()
    assert padded.GetSpacing() == image.GetSpacing()
    assert padded.GetDirection() == image.GetDirection()


def test_the_repadded_crop_keeps_the_pixel_type(tmp_path):
    """A label map that came back as float is no longer a label map."""
    scan = _volume(tmp_path / "in" / "A_scan.nii.gz")
    image = _read(scan)
    padded = pipeline.repad(image, pipeline.crop(image, (3, 3, 3), (7, 7, 7)), (3, 3, 3))

    assert padded.GetPixelID() == image.GetPixelID()


def test_a_non_identity_direction_still_crops_the_box_that_was_drawn(tmp_path):
    """The whole point of going through physical space: the same ROI must select
    the same anatomy whichever way the volume is stored."""
    box = _roi(tmp_path / "box.mrk.json", center=(9.5, 9.5, 9.5), size=(4.0, 4.0, 4.0))
    roi = pipeline.read_roi(str(box))

    straight = _read(_volume(tmp_path / "a.nii.gz", origin=(0.0, 0.0, 0.0)))
    flipped = _read(_volume(tmp_path / "b.nii.gz", origin=(19.0, 19.0, 0.0),
                            direction=(-1, 0, 0, 0, -1, 0, 0, 0, 1)))

    for image in (straight, flipped):
        lower, upper, _clamped = pipeline.crop_bounds(image, roi)
        cropped = pipeline.crop(image, lower, upper)
        corner = cropped.TransformIndexToPhysicalPoint((0, 0, 0))
        far = cropped.TransformIndexToPhysicalPoint(
            tuple(value - 1 for value in cropped.GetSize()))
        assert cropped.GetSize() == (4, 4, 4)
        # Upstream truncates both index corners toward zero, so the window a
        # negative direction axis selects is offset from its twin's by one
        # voxel. What must hold either way is that the box the clinician drew
        # is the box that was cropped: the ROI centre is inside the result.
        for axis in range(3):
            low, high = sorted((corner[axis], far[axis]))
            assert low - 1.0 <= roi.center[axis] <= high + 1.0


# ---------------------------------------------------------------------------
# The ROI's own coordinate system, which upstream never read
# ---------------------------------------------------------------------------

def test_an_lps_roi_is_taken_as_written(tmp_path):
    box = pipeline.read_roi(str(_roi(tmp_path / "box.mrk.json", center=(1.0, 2.0, 3.0),
                                     size=(4, 4, 4), coordinate_system="LPS")))
    assert box.center == [1.0, 2.0, 3.0]
    assert box.coordinate_system == "LPS"


def test_a_ras_roi_is_converted_to_the_lps_the_volume_lives_in(tmp_path):
    """Slicer writes the frame into the file and upstream never read it, so an
    ROI saved in RAS cropped the mirror image of what the user drew."""
    box = pipeline.read_roi(str(_roi(tmp_path / "box.mrk.json", center=(1.0, 2.0, 3.0),
                                     size=(4, 4, 4), coordinate_system="RAS")))
    assert box.center == [-1.0, -2.0, 3.0]


def test_a_ras_roi_and_its_lps_twin_crop_the_same_voxels(tmp_path):
    scan = _volume(tmp_path / "in" / "A_scan.nii.gz")
    image = _read(scan)
    lps = pipeline.read_roi(str(_roi(tmp_path / "l.mrk.json", center=(9.5, 9.5, 9.5),
                                     size=(4, 4, 4), coordinate_system="LPS")))
    ras = pipeline.read_roi(str(_roi(tmp_path / "r.mrk.json", center=(-9.5, -9.5, 9.5),
                                     size=(4, 4, 4), coordinate_system="RAS")))

    assert pipeline.crop_bounds(image, lps) == pipeline.crop_bounds(image, ras)


def test_an_unknown_coordinate_system_is_refused(tmp_path):
    with pytest.raises(ValueError, match="unknown coordinate system"):
        pipeline.read_roi(str(_roi(tmp_path / "box.mrk.json", center=(1.0, 2.0, 3.0),
                                   size=(4, 4, 4), coordinate_system="MNI")))


def test_a_rotated_roi_is_cropped_axis_aligned_and_the_report_says_so(tmp_path):
    """Upstream reads `center` and `size` and nothing else, so a rotated box is
    treated as its axis-aligned self. Kept -- an oriented crop is a resampling,
    which is a different tool -- but no longer silent."""
    angle = 0.4
    cosine, sine = np.cos(angle), np.sin(angle)
    scan = _volume(tmp_path / "in" / "A_scan.nii.gz")
    box = _roi(tmp_path / "box.mrk.json", center=(9.5, 9.5, 9.5), size=(4, 4, 4),
               orientation=[cosine, -sine, 0.0, sine, cosine, 0.0, 0.0, 0.0, 1.0])
    out = tmp_path / "out"

    sadt_autocrop3d.run(scans=scan, roi=box, output_dir=out)

    assert _report(out)["cases"]["A_scan.nii.gz"]["roi_orientation_ignored"] is True


def test_an_unrotated_roi_is_not_flagged(tmp_path):
    scan = _volume(tmp_path / "in" / "A_scan.nii.gz")
    box = _roi(tmp_path / "box.mrk.json", center=(9.5, 9.5, 9.5), size=(4, 4, 4),
               orientation=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    out = tmp_path / "out"

    sadt_autocrop3d.run(scans=scan, roi=box, output_dir=out)

    assert _report(out)["cases"]["A_scan.nii.gz"]["roi_orientation_ignored"] is False


def test_a_signed_axis_permutation_is_not_flagged(tmp_path):
    """The ROI shipped with upstream's own test data carries a 180-degree flip
    about z. That maps an axis-aligned box onto itself, so ignoring it is
    exact and warning about it would be noise."""
    scan = _volume(tmp_path / "in" / "A_scan.nii.gz")
    box = _roi(tmp_path / "box.mrk.json", center=(9.5, 9.5, 9.5), size=(4, 4, 4),
               orientation=[-1.0, -0.0, -0.0, -0.0, -1.0, -0.0, 0.0, 0.0, 1.0])
    out = tmp_path / "out"

    sadt_autocrop3d.run(scans=scan, roi=box, output_dir=out)

    assert _report(out)["cases"]["A_scan.nii.gz"]["roi_orientation_ignored"] is False


def test_an_roi_with_no_coordinate_system_field_is_read_as_lps(tmp_path):
    """Upstream's own test ROI has no `coordinateSystem` key at all, so the
    fallback is the case that matters in practice."""
    path = tmp_path / "box.mrk.json"
    path.write_text(json.dumps({"markups": [
        {"type": "ROI", "center": [1.0, 2.0, 3.0], "size": [4.0, 4.0, 4.0]}
    ]}))

    box = pipeline.read_roi(str(path))

    assert box.center == [1.0, 2.0, 3.0] and box.coordinate_system == "LPS"


def test_a_negative_roi_size_is_read_as_an_extent(tmp_path):
    box = pipeline.read_roi(str(_roi(tmp_path / "box.mrk.json", center=(0.0, 0.0, 0.0),
                                     size=(-4.0, 4.0, -4.0))))
    assert box.size == [4.0, 4.0, 4.0]


# ===========================================================================
# The surface, in the space it belongs to
# ===========================================================================

def test_the_surface_lands_on_the_volume_it_came_from(tmp_path):
    """`vtkNIFTIImageReader` deliberately does not apply the file's qform: it
    reports origin (0, 0, 0) and leaves the matrix to the caller. Upstream never
    applied it, so every surface it wrote was offset by the volume's origin,
    un-rotated, and shifted again by the padding."""
    image = sitk.ReadImage(str(_labelmap(tmp_path / "s.nii.gz",
                                         {1: (np.s_[3:8], np.s_[3:8], np.s_[3:8])},
                                         spacing=(0.4, 0.4, 0.4), origin=(5.0, -3.0, 2.0))))
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    destination = tmp_path / "s.vtk"

    pipeline.write_surface(image, str(destination), str(scratch),
                           padding_mm=2.0, smoothing_iterations=0)

    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(str(destination))
    reader.Update()
    bounds = reader.GetOutput().GetBounds()

    # The isosurface sits half a voxel outside the labelled voxel centres.
    for axis, (low, high) in enumerate(((3, 7), (3, 7), (3, 7))):
        expected_low = image.GetOrigin()[axis] + (low - 0.5) * image.GetSpacing()[axis]
        expected_high = image.GetOrigin()[axis] + (high + 0.5) * image.GetSpacing()[axis]
        assert bounds[2 * axis] == pytest.approx(expected_low, abs=1e-4)
        assert bounds[2 * axis + 1] == pytest.approx(expected_high, abs=1e-4)


def test_the_surface_is_written_beside_its_crop(tmp_path):
    scans, out = tmp_path / "scans", tmp_path / "out"
    _labelmap(scans / "site" / "A_seg.nii.gz", {1: (np.s_[3:8], np.s_[3:8], np.s_[3:8])})
    box = _roi(tmp_path / "box.mrk.json", center=(9.5, 9.5, 9.5), size=(18, 18, 18))

    sadt_autocrop3d.run(scans=scans, roi=box, output_dir=out)

    assert (out / "site" / "A_seg_cropped.nii.gz").is_file()
    assert (out / "site" / "A_seg_cropped_vtk.vtk").is_file()


def test_the_surface_covers_only_the_cropped_part(tmp_path):
    """The surface is built from what was written, so a structure the ROI cut
    in half comes back cut in half rather than whole."""
    scans, out = tmp_path / "scans", tmp_path / "out"
    _labelmap(scans / "A_seg.nii.gz", {1: (np.s_[2:18], np.s_[2:18], np.s_[2:18])})
    box = _roi(tmp_path / "box.mrk.json", center=(5.0, 5.0, 5.0), size=(10, 10, 10))

    sadt_autocrop3d.run(scans=scans, roi=box, output_dir=out)

    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(str(out / "A_seg_cropped_vtk.vtk"))
    reader.Update()
    bounds = reader.GetOutput().GetBounds()

    assert bounds[1] <= 10.5 and bounds[3] <= 10.5 and bounds[5] <= 10.5


def test_the_surface_name_records_the_suffix(tmp_path):
    assert pipeline.surface_name("A_seg.nii.gz", "cropped") == "A_seg_cropped_vtk.vtk"
    assert pipeline.surface_name("A_seg.nii.gz", "") == "A_seg_vtk.vtk"


# ===========================================================================
# The run report, and the contract the server reads
# ===========================================================================

def test_the_report_names_every_scan_and_the_roi_it_used(tmp_path):
    scans, rois, out = tmp_path / "scans", tmp_path / "rois", tmp_path / "out"
    _volume(scans / "A_scan.nii.gz")
    _volume(scans / "B_scan.nii.gz")
    _roi(rois / "A_ROI.mrk.json", center=(5.5, 5.5, 5.5), size=(4, 4, 4))
    _roi(rois / "B_ROI.mrk.json", center=(5.5, 5.5, 5.5), size=(6, 6, 6))

    sadt_autocrop3d.run(scans=scans, roi=rois, output_dir=out)

    report = _report(out)
    assert report["cases"]["A_scan.nii.gz"]["roi"] == "A_ROI.mrk.json"
    assert report["cases"]["B_scan.nii.gz"]["roi"] == "B_ROI.mrk.json"
    assert report["summary"]["cropped"] == 2


def test_the_report_records_the_index_bounds_that_were_used(tmp_path):
    scan = _volume(tmp_path / "in" / "A_scan.nii.gz")
    box = _roi(tmp_path / "box.mrk.json", center=(5.5, 5.5, 5.5), size=(4, 4, 4))
    out = tmp_path / "out"

    sadt_autocrop3d.run(scans=scan, roi=box, output_dir=out)

    entry = _report(out)["cases"]["A_scan.nii.gz"]
    # The box spans physical 3.5..7.5, truncated toward zero to 3..7.
    assert entry["index_lower"] == [3, 3, 3] and entry["index_upper"] == [7, 7, 7]


def test_a_previous_runs_output_is_not_cropped_again(tmp_path):
    """`P1_cropped.nii.gz` sorts before `P1_scan.nii.gz`, so pointing a second
    run at an output folder would otherwise crop the first run's result."""
    scans, out = tmp_path / "scans", tmp_path / "out"
    _volume(scans / "A_scan.nii.gz")
    _volume(scans / "A_scan_cropped.nii.gz")
    box = _roi(tmp_path / "box.mrk.json", center=(5.5, 5.5, 5.5), size=(4, 4, 4))

    sadt_autocrop3d.run(scans=scans, roi=box, output_dir=out)

    assert list(_outputs(out)) == ["A_scan_cropped.nii.gz"]
    assert _report(out)["summary"]["scans_found"] == 1


def test_a_folder_holding_only_previous_output_is_still_cropped(tmp_path):
    """Excluding the last run's output must not turn a legitimate re-crop into
    'no scan found'."""
    scans, out = tmp_path / "scans", tmp_path / "out"
    _volume(scans / "A_scan_cropped.nii.gz")
    box = _roi(tmp_path / "box.mrk.json", center=(5.5, 5.5, 5.5), size=(4, 4, 4))

    sadt_autocrop3d.run(scans=scans, roi=box, output_dir=out)

    assert _report(out)["summary"]["cropped"] == 1


def test_run_returns_the_output_directory(tmp_path):
    scan = _volume(tmp_path / "in" / "A_scan.nii.gz")
    box = _roi(tmp_path / "box.mrk.json", center=(5.5, 5.5, 5.5), size=(4, 4, 4))
    out = tmp_path / "out"

    assert sadt_autocrop3d.run(scans=scan, roi=box, output_dir=out) == out


def test_run_writes_nothing_outside_the_output_directory(tmp_path):
    """The contract every tool in this repository holds to."""
    scans, out = tmp_path / "scans", tmp_path / "out"
    _labelmap(scans / "A_seg.nii.gz", {1: (np.s_[3:8], np.s_[3:8], np.s_[3:8])})
    box = _roi(tmp_path / "box.mrk.json", center=(9.5, 9.5, 9.5), size=(18, 18, 18))
    before = {path for path in scans.rglob("*")}

    sadt_autocrop3d.run(scans=scans, roi=box, output_dir=out)

    assert {path for path in scans.rglob("*")} == before


def test_run_annotations_are_stdlib_only():
    """`scripts/describe.py` turns this signature into the published schema and
    refuses anything it cannot render."""
    allowed = {"Path", "str", "int", "float", "bool"}
    for name, parameter in inspect.signature(sadt_autocrop3d.run).parameters.items():
        annotation = parameter.annotation
        assert annotation is not inspect.Parameter.empty, f"{name} has no annotation"
        if typing.get_origin(annotation) is typing.Literal:
            assert all(isinstance(option, str) for option in typing.get_args(annotation))
            continue
        assert getattr(annotation, "__name__", None) in allowed, f"{name}: {annotation}"


def test_every_optional_argument_has_a_usable_default():
    """No default means required, and `None` as a default makes a required
    argument look optional."""
    for name, parameter in inspect.signature(sadt_autocrop3d.run).parameters.items():
        if parameter.default is inspect.Parameter.empty:
            assert name in ("scans", "roi", "output_dir")
        else:
            assert parameter.default is not None, name


def test_the_surfaces_default_is_one_of_its_own_options():
    parameter = inspect.signature(sadt_autocrop3d.run).parameters["surfaces"]
    assert parameter.default in typing.get_args(parameter.annotation)


def test_importing_the_package_costs_no_heavy_dependency():
    """CI imports every tool on every PR to publish its schema. SimpleITK, vtk
    and numpy are imported inside the functions that need them."""
    source = (
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "src", "sadt_autocrop3d")
    )
    for name in ("__init__.py", "pipeline.py"):
        with open(os.path.join(source, name)) as handle:
            for line in handle:
                if line.startswith(("import ", "from ")):
                    assert not any(
                        heavy in line for heavy in ("SimpleITK", "import vtk", "numpy")
                    ), f"{name}: module-level heavy import: {line.strip()}"


def test_the_report_names_outputs_relative_to_the_output_folder(tmp_path):
    """The report travels to the client. A server-side job path in it is an
    information leak, and means nothing to the caller anyway -- found by
    reading a real API response, where every entry carried
    `/tmp/inference_server/job_<id>/output/...`."""
    scans = tmp_path / "in"
    scans.mkdir()
    _volume(scans / "a.nii.gz")
    roi = _roi(tmp_path / "a_ROI.mrk.json", center=(5.5, 5.5, 5.5), size=(4, 4, 4))

    output_dir = tmp_path / "out"
    sadt_autocrop3d.run(scans=scans, roi=roi, output_dir=output_dir)

    report = _report(output_dir)
    for entry in report["cases"].values():
        assert not os.path.isabs(entry["produced"][0]), entry["produced"][0]
        assert (output_dir / entry["produced"][0]).is_file()
        assert "_absolute" not in entry
