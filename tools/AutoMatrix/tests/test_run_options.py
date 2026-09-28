"""`run()` end to end: every argument, the report, and what the README claims.

Nothing is stubbed. Each of these builds real volumes, real markups files and
real transforms in `tmp_path` and reads back what SimpleITK wrote.
"""

import json

import numpy as np
import pytest
import SimpleITK as sitk

import sadt_automatrix
from sadt_automatrix import pipeline


def _report(output_dir):
    return json.loads((output_dir / "AutoMatrix_report.json").read_text())


def _outputs(output_dir):
    return sorted(
        path.name for path in output_dir.rglob("*")
        if path.is_file() and path.name != "AutoMatrix_report.json"
    )


def _labels(path):
    return set(np.unique(sitk.GetArrayFromImage(sitk.ReadImage(str(path)))))


def _offset_transform(path, amount=0.3):
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteTransform(
        sitk.TranslationTransform(3, (amount, amount, amount)), str(path))
    return path


# ---------------------------------------------------------------------------
# `content`
# ---------------------------------------------------------------------------

def test_a_segmentation_run_invents_no_label(tmp_path, label_volume):
    """The interpolation rule, through the whole tool rather than through
    `resample` alone: a client sends `content=Segmentation` and what comes back
    has to hold only the labels that went in."""
    label_volume(tmp_path / "in" / "P1_T1_Seg.nii.gz")
    _offset_transform(tmp_path / "tfm" / "P1_transform.tfm")

    out = sadt_automatrix.run(files=tmp_path / "in", transforms=tmp_path / "tfm",
                              output_dir=tmp_path / "out", content="Segmentation")

    assert _labels(out / "P1_T1_Seg_Reg.nii.gz") <= {0, 1, 3}


def test_a_scan_run_interpolates_linearly(tmp_path, label_volume):
    """The same input through the other value of the same argument. If this
    stopped blending, the test above would pass against a tool that had lost
    the rule entirely."""
    label_volume(tmp_path / "in" / "P1_T1.nii.gz")
    _offset_transform(tmp_path / "tfm" / "P1_transform.tfm")

    out = sadt_automatrix.run(files=tmp_path / "in", transforms=tmp_path / "tfm",
                              output_dir=tmp_path / "out", content="Scan")

    assert 2 in _labels(out / "P1_T1_Reg.nii.gz")


def test_automatic_is_the_default_and_detects_a_label_map(tmp_path, label_volume):
    """A caller who says nothing gets the interpolation each FILE needs, and
    finds out from the report which they got.

    The default used to be "Scan" -- one answer for the whole batch, which
    destroys a label map fed in with scans. It detects per file now, so this
    label volume keeps its values rather than being interpolated into new ones.
    """
    label_volume(tmp_path / "in" / "P1_T1.nii.gz")
    _offset_transform(tmp_path / "tfm" / "P1_transform.tfm")

    out = sadt_automatrix.run(files=tmp_path / "in", transforms=tmp_path / "tfm",
                              output_dir=tmp_path / "out")

    assert _report(out)["content"] == "Automatic"
    # Detected as a label map, so nearest-neighbour: no value appears that was
    # not in the input.
    written = _labels(out / "P1_T1_Reg.nii.gz")
    assert 2 not in written, "a label map must not gain values by interpolation"


@pytest.mark.parametrize("content", ["Scan", "Segmentation"])
def test_the_content_asked_for_is_recorded(tmp_path, volume, transform_file, content):
    volume(tmp_path / "in" / "P1_T1.nii.gz")
    transform_file(tmp_path / "tfm" / "P1_transform.tfm")

    out = sadt_automatrix.run(files=tmp_path / "in", transforms=tmp_path / "tfm",
                              output_dir=tmp_path / "out", content=content)

    assert _report(out)["content"] == content


def test_landmarks_are_moved_whatever_the_content_says(tmp_path, landmark_file,
                                                       transform_file):
    """`content` picks an interpolator, and a point is not interpolated. A
    landmark file sent with `content=Segmentation` still has to arrive moved."""
    landmark_file(tmp_path / "in" / "P1_lm.mrk.json", [(10.0, 0.0, 0.0)])
    transform_file(tmp_path / "tfm" / "P1_transform.tfm", (4.0, 0.0, 0.0))

    out = sadt_automatrix.run(files=tmp_path / "in", transforms=tmp_path / "tfm",
                              output_dir=tmp_path / "out", content="Segmentation")

    moved = json.loads((out / "P1_lm_Reg.mrk.json").read_text())
    assert moved["markups"][0]["controlPoints"][0]["position"] == [6.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# `name_output_after_transform` and `output_suffix`
# ---------------------------------------------------------------------------

def test_without_the_flag_an_output_is_named_after_its_input(tmp_path, volume,
                                                             transform_file):
    volume(tmp_path / "in" / "P1_T1.nii.gz")
    transform_file(tmp_path / "tfm" / "P1_CBReg_matrix.tfm")

    out = sadt_automatrix.run(files=tmp_path / "in", transforms=tmp_path / "tfm",
                              output_dir=tmp_path / "out",
                              name_output_after_transform=False)

    assert _outputs(out) == ["P1_T1_Reg.nii.gz"]


def test_with_the_flag_the_transform_is_named_too(tmp_path, volume, transform_file):
    volume(tmp_path / "in" / "P1_T1.nii.gz")
    transform_file(tmp_path / "tfm" / "P1_CBReg_matrix.tfm")

    out = sadt_automatrix.run(files=tmp_path / "in", transforms=tmp_path / "tfm",
                              output_dir=tmp_path / "out",
                              name_output_after_transform=True)

    assert _outputs(out) == ["P1_T1_Reg_P1_CBReg_matrix.nii.gz"]


def test_the_flag_names_landmark_outputs_too(tmp_path, landmark_file, transform_file):
    landmark_file(tmp_path / "in" / "P1_lm.mrk.json", [(1.0, 0.0, 0.0)])
    transform_file(tmp_path / "tfm" / "P1_CBReg_matrix.tfm")

    out = sadt_automatrix.run(files=tmp_path / "in", transforms=tmp_path / "tfm",
                              output_dir=tmp_path / "out",
                              name_output_after_transform=True)

    assert _outputs(out) == ["P1_lm_Reg_P1_CBReg_matrix.mrk.json"]


def test_a_suffix_is_appended_to_every_output(tmp_path, volume, landmark_file,
                                              transform_file):
    volume(tmp_path / "in" / "P1_T1.nii.gz")
    landmark_file(tmp_path / "in" / "P1_lm.mrk.json", [(1.0, 0.0, 0.0)])
    transform_file(tmp_path / "tfm" / "P1_transform.tfm")

    out = sadt_automatrix.run(files=tmp_path / "in", transforms=tmp_path / "tfm",
                              output_dir=tmp_path / "out", output_suffix="Moved")

    assert _outputs(out) == ["P1_T1_Moved.nii.gz", "P1_lm_Moved.mrk.json"]


def test_an_empty_suffix_leaves_the_name_alone(tmp_path, volume, transform_file):
    """No suffix means no suffix, not a trailing underscore: `P1_T1.nii.gz`."""
    volume(tmp_path / "in" / "P1_T1.nii.gz")
    transform_file(tmp_path / "tfm" / "P1_transform.tfm")

    out = sadt_automatrix.run(files=tmp_path / "in", transforms=tmp_path / "tfm",
                              output_dir=tmp_path / "out", output_suffix="")

    assert _outputs(out) == ["P1_T1.nii.gz"]


def test_an_empty_suffix_with_the_flag_gives_one_separator_not_two(
        tmp_path, volume, transform_file):
    volume(tmp_path / "in" / "P1_T1.nii.gz")
    transform_file(tmp_path / "tfm" / "P1_CBReg_matrix.tfm")

    out = sadt_automatrix.run(files=tmp_path / "in", transforms=tmp_path / "tfm",
                              output_dir=tmp_path / "out", output_suffix="",
                              name_output_after_transform=True)

    assert _outputs(out) == ["P1_T1_P1_CBReg_matrix.nii.gz"]


def test_the_suffix_asked_for_is_recorded(tmp_path, volume, transform_file):
    volume(tmp_path / "in" / "P1_T1.nii.gz")
    transform_file(tmp_path / "tfm" / "P1_transform.tfm")

    out = sadt_automatrix.run(files=tmp_path / "in", transforms=tmp_path / "tfm",
                              output_dir=tmp_path / "out", output_suffix="Moved")

    assert _report(out)["output_suffix"] == "Moved"


@pytest.mark.parametrize("name,expected", [
    ("P1_T1.nii.gz", "P1_T1_Reg.nii.gz"),
    ("P1_T1.nii", "P1_T1_Reg.nii"),
    ("P1_T1.nrrd", "P1_T1_Reg.nrrd"),
    # No `.nrrd.gz`: NRRD compresses inside the file and ITK ships no writer
    # for that spelling at all, so nothing can produce one to transform.
    ("P1_T1.gipl", "P1_T1_Reg.gipl"),
    ("P1_T1.gipl.gz", "P1_T1_Reg.gipl.gz"),
    ("P1_lm.mrk.json", "P1_lm_Reg.mrk.json"),
])
def test_a_compound_extension_stays_on_the_end_of_the_name(
        tmp_path, volume, landmark_file, transform_file, name, expected):
    """`os.path.splitext` alone would produce `P1_T1_Reg.gz`, which no reader
    opens."""
    if name.endswith(".mrk.json"):
        landmark_file(tmp_path / "in" / name, [(1.0, 0.0, 0.0)])
    else:
        volume(tmp_path / "in" / name)
    transform_file(tmp_path / "tfm" / "P1_transform.tfm")

    out = sadt_automatrix.run(files=tmp_path / "in", transforms=tmp_path / "tfm",
                              output_dir=tmp_path / "out")

    assert _outputs(out) == [expected]


def test_two_transforms_of_one_patient_never_write_the_same_name(
        tmp_path, volume, transform_file):
    """With the flag off both used to resolve to ONE output name, so the second
    silently overwrote the first while the report counted both as written. A
    patient with several transforms gets them named apart regardless."""
    volume(tmp_path / "in" / "P1_T1.nii.gz")
    transform_file(tmp_path / "tfm" / "P1_CBReg_matrix.tfm", (1.0, 0.0, 0.0))
    transform_file(tmp_path / "tfm" / "P1_MANDReg_matrix.tfm", (0.0, 2.0, 0.0))

    out = sadt_automatrix.run(files=tmp_path / "in", transforms=tmp_path / "tfm",
                              output_dir=tmp_path / "out",
                              name_output_after_transform=False)

    written = _outputs(out)
    assert len(written) == 2, f"one transform's result was lost: {written}"
    report = _report(out)
    assert report["summary"]["files_written"] == 2
    assert sorted(entry["file"] for entry in report["cases"]["P1"]["outputs"]) == written


def test_the_two_outputs_of_two_transforms_really_differ(tmp_path, volume,
                                                         transform_file):
    """Not just two names: two different resamplings."""
    volume(tmp_path / "in" / "P1_T1.nii.gz")
    transform_file(tmp_path / "tfm" / "P1_CBReg_matrix.tfm", (1.0, 0.0, 0.0))
    transform_file(tmp_path / "tfm" / "P1_MANDReg_matrix.tfm", (0.0, 2.0, 0.0))

    out = sadt_automatrix.run(files=tmp_path / "in", transforms=tmp_path / "tfm",
                              output_dir=tmp_path / "out")

    origins = {
        sitk.ReadImage(str(path)).GetOrigin()
        for path in out.glob("*.nii.gz")
    }
    assert origins == {(1.0, 0.0, 0.0), (0.0, 2.0, 0.0)}


# ---------------------------------------------------------------------------
# `reference`
# ---------------------------------------------------------------------------

def test_with_no_reference_the_image_keeps_its_own_grid(tmp_path, volume,
                                                        transform_file):
    volume(tmp_path / "in" / "P1_T1.nii.gz", spacing=(0.5, 0.5, 0.5))
    transform_file(tmp_path / "tfm" / "P1_transform.tfm", (1.0, 2.0, 3.0))

    out = sadt_automatrix.run(files=tmp_path / "in", transforms=tmp_path / "tfm",
                              output_dir=tmp_path / "out")

    written = sitk.ReadImage(str(out / "P1_T1_Reg.nii.gz"))
    assert written.GetSize() == (6, 6, 6)
    assert written.GetSpacing() == (0.5, 0.5, 0.5)
    assert written.GetOrigin() == (1.0, 2.0, 3.0)
    assert _report(out)["reference"] is None


def test_a_reference_defines_the_output_grid(tmp_path, volume, transform_file):
    volume(tmp_path / "in" / "P1_T1.nii.gz", spacing=(0.5, 0.5, 0.5))
    volume(tmp_path / "ref" / "atlas.nii.gz", spacing=(2.0, 2.0, 2.0),
           origin=(-4.0, -4.0, -4.0), size=10)
    transform_file(tmp_path / "tfm" / "P1_transform.tfm", (1.0, 2.0, 3.0))

    out = sadt_automatrix.run(files=tmp_path / "in", transforms=tmp_path / "tfm",
                              output_dir=tmp_path / "out",
                              reference=tmp_path / "ref" / "atlas.nii.gz")

    written = sitk.ReadImage(str(out / "P1_T1_Reg.nii.gz"))
    assert written.GetSize() == (10, 10, 10)
    assert written.GetSpacing() == (2.0, 2.0, 2.0)
    assert written.GetOrigin() == (-4.0, -4.0, -4.0)


def test_the_reference_is_named_in_the_report_by_its_file_name_only(
        tmp_path, volume, transform_file):
    """The report travels to a client, and a server-side directory is not the
    client's to see."""
    volume(tmp_path / "in" / "P1_T1.nii.gz")
    volume(tmp_path / "ref" / "atlas.nii.gz")
    transform_file(tmp_path / "tfm" / "P1_transform.tfm")

    out = sadt_automatrix.run(files=tmp_path / "in", transforms=tmp_path / "tfm",
                              output_dir=tmp_path / "out",
                              reference=tmp_path / "ref" / "atlas.nii.gz")

    assert _report(out)["reference"] == "atlas.nii.gz"


def test_a_transform_called_mirror_gets_no_special_treatment(tmp_path, volume,
                                                             transform_file):
    """Upstream forced the image as its own reference for any matrix whose name
    contained "mirror" -- a substring test on a file name deciding a resampling
    grid. Not ported: the reference the caller gave is the grid, whatever the
    transform is called."""
    volume(tmp_path / "in" / "P1_mirror_T1.nii.gz", spacing=(0.5, 0.5, 0.5))
    volume(tmp_path / "ref" / "atlas.nii.gz", spacing=(2.0, 2.0, 2.0), size=10)
    transform_file(tmp_path / "tfm" / "P1_mirror_transform.tfm")

    out = sadt_automatrix.run(files=tmp_path / "in", transforms=tmp_path / "tfm",
                              output_dir=tmp_path / "out",
                              reference=tmp_path / "ref" / "atlas.nii.gz")

    written = sitk.ReadImage(str(out / "P1_mirror_T1_Reg.nii.gz"))
    assert written.GetSpacing() == (2.0, 2.0, 2.0)
    assert written.GetSize() == (10, 10, 10)


# ---------------------------------------------------------------------------
# What is found, and where it goes
# ---------------------------------------------------------------------------

def test_a_single_file_may_be_given_instead_of_a_folder(tmp_path, volume,
                                                        transform_file):
    volume(tmp_path / "in" / "P1_T1.nii.gz")
    transform_file(tmp_path / "tfm" / "P1_transform.tfm")

    out = sadt_automatrix.run(files=tmp_path / "in" / "P1_T1.nii.gz",
                              transforms=tmp_path / "tfm" / "P1_transform.tfm",
                              output_dir=tmp_path / "out")

    assert _outputs(out) == ["P1_T1_Reg.nii.gz"]


def test_the_output_directory_is_created_when_it_is_not_there(tmp_path, volume,
                                                              transform_file):
    volume(tmp_path / "in" / "P1_T1.nii.gz")
    transform_file(tmp_path / "tfm" / "P1_transform.tfm")
    destination = tmp_path / "not" / "there" / "yet"

    out = sadt_automatrix.run(files=tmp_path / "in", transforms=tmp_path / "tfm",
                              output_dir=destination)

    assert out == destination and destination.is_dir()
    assert _outputs(out) == ["P1_T1_Reg.nii.gz"]


def test_the_input_tree_is_mirrored_to_any_depth(tmp_path, volume, transform_file):
    """Two patients' files in two folders keep their folders: upstream's flat
    output folder collapsed homonyms into one another."""
    volume(tmp_path / "in" / "siteA" / "2024" / "P1_T1.nii.gz")
    volume(tmp_path / "in" / "siteB" / "P2_T1.nii.gz")
    transform_file(tmp_path / "tfm" / "P1_transform.tfm")
    transform_file(tmp_path / "tfm" / "P2_transform.tfm")

    out = sadt_automatrix.run(files=tmp_path / "in", transforms=tmp_path / "tfm",
                              output_dir=tmp_path / "out")

    assert (out / "siteA" / "2024" / "P1_T1_Reg.nii.gz").is_file()
    assert (out / "siteB" / "P2_T1_Reg.nii.gz").is_file()


def test_transforms_are_found_in_the_tree_areg_writes_them_in(
        tmp_path, volume, transform_file):
    """The branch that consumed AREG's output could never run upstream: it read
    an argument argparse did not declare, so it raised `AttributeError` on its
    first landmark file. Here a transform is matched by patient, wherever in the
    tree it sits."""
    volume(tmp_path / "in" / "P1_T1.nii.gz")
    transform_file(tmp_path / "tfm" / "P1_OutReg" / "P1_CB_transform.tfm")

    out = sadt_automatrix.run(files=tmp_path / "in", transforms=tmp_path / "tfm",
                              output_dir=tmp_path / "out")

    assert _outputs(out) == ["P1_T1_Reg.nii.gz"]
    assert _report(out)["cases"]["P1"]["transforms"] == ["P1_CB_transform.tfm"]


def test_no_jaw_table_decides_which_transform_a_file_gets(tmp_path, volume,
                                                          transform_file):
    """Upstream mapped `_L` to the MAXILLARY registration and `_U` to the
    mandibular one, so a lower-arch file was transformed by the wrong matrix and
    written out as a success. There is no such table here: a patient's
    transforms are each applied and told apart by name."""
    volume(tmp_path / "in" / "P1_MAND_T1.nii.gz")
    transform_file(tmp_path / "tfm" / "P1_MANDReg_matrix.tfm", (1.0, 0.0, 0.0))
    transform_file(tmp_path / "tfm" / "P1_MAXReg_matrix.tfm", (0.0, 2.0, 0.0))

    out = sadt_automatrix.run(files=tmp_path / "in", transforms=tmp_path / "tfm",
                              output_dir=tmp_path / "out")

    assert _outputs(out) == [
        "P1_MAND_T1_Reg_P1_MANDReg_matrix.nii.gz",
        "P1_MAND_T1_Reg_P1_MAXReg_matrix.nii.gz",
    ]


def test_a_file_whose_name_only_contains_a_token_is_reported_not_paired(
        tmp_path, volume, transform_file):
    """`P1_L_T1.nii.gz` keys to `P1_L`, which no transform matches. It is named
    in `without_a_transform` -- upstream's substring test would have paired it
    with whichever transform the dict happened to yield first."""
    volume(tmp_path / "in" / "P1_L_T1.nii.gz")
    volume(tmp_path / "in" / "P1_MAND_T1.nii.gz")
    transform_file(tmp_path / "tfm" / "P1_MANDReg_matrix.tfm")

    out = sadt_automatrix.run(files=tmp_path / "in", transforms=tmp_path / "tfm",
                              output_dir=tmp_path / "out")

    assert _report(out)["without_a_transform"] == ["P1_L"]
    assert _outputs(out) == ["P1_MAND_T1_Reg.nii.gz"]


def test_hidden_files_are_not_transformed(tmp_path, volume, transform_file):
    """A `._P1_T1.nii.gz` left by a macOS copy is not a scan."""
    volume(tmp_path / "in" / "P1_T1.nii.gz")
    volume(tmp_path / "in" / ".P1_T1.nii.gz")
    transform_file(tmp_path / "tfm" / "P1_transform.tfm")

    out = sadt_automatrix.run(files=tmp_path / "in", transforms=tmp_path / "tfm",
                              output_dir=tmp_path / "out")

    assert _outputs(out) == ["P1_T1_Reg.nii.gz"]


def test_a_file_that_is_not_a_transform_is_ignored_in_the_transform_folder(
        tmp_path, volume, transform_file):
    volume(tmp_path / "in" / "P1_T1.nii.gz")
    transform_file(tmp_path / "tfm" / "P1_transform.tfm")
    (tmp_path / "tfm" / "notes.md").write_text("computed on 2026-09-01")

    out = sadt_automatrix.run(files=tmp_path / "in", transforms=tmp_path / "tfm",
                              output_dir=tmp_path / "out")

    assert _report(out)["cases"]["P1"]["transforms"] == ["P1_transform.tfm"]


def test_scans_segmentations_and_landmarks_of_one_patient_go_through_together(
        tmp_path, volume, landmark_file, transform_file):
    """The three kinds of file this tool exists to keep in step."""
    volume(tmp_path / "in" / "P1_T1_scan.nii.gz")
    volume(tmp_path / "in" / "P1_T1_MAND_Seg.nii.gz")
    landmark_file(tmp_path / "in" / "P1_T1_lm.mrk.json", [(1.0, 0.0, 0.0)])
    transform_file(tmp_path / "tfm" / "P1_CB_Reg_transform.tfm")

    out = sadt_automatrix.run(files=tmp_path / "in", transforms=tmp_path / "tfm",
                              output_dir=tmp_path / "out")

    assert _outputs(out) == [
        "P1_T1_MAND_Seg_Reg.nii.gz",
        "P1_T1_lm_Reg.mrk.json",
        "P1_T1_scan_Reg.nii.gz",
    ]
    assert list(_report(out)["cases"]) == ["P1"]


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

def test_the_report_carries_everything_a_caller_has_to_read(tmp_path, volume,
                                                            landmark_file,
                                                            transform_file):
    volume(tmp_path / "in" / "P1_T1.nii.gz")
    landmark_file(tmp_path / "in" / "P1_lm.mrk.json", [(1.0, 0.0, 0.0), (2.0, 0.0, 0.0)])
    volume(tmp_path / "in" / "P2_T1.nii.gz")
    transform_file(tmp_path / "tfm" / "P1_transform.tfm")
    transform_file(tmp_path / "tfm" / "P9_transform.tfm")

    report = _report(sadt_automatrix.run(
        files=tmp_path / "in", transforms=tmp_path / "tfm",
        output_dir=tmp_path / "out"))

    assert report["tool"] == "AutoMatrix"
    # The default, and it is "Automatic": the interpolator is decided per file.
    assert report["content"] == "Automatic"
    assert report["reference"] is None
    assert report["output_suffix"] == "Reg"
    assert report["without_a_transform"] == ["P2"]
    assert report["transforms_without_a_file"] == ["P9"]
    assert report["summary"] == {"cases": 1, "files_written": 2}
    assert isinstance(report["duration_seconds"], float)
    assert report["cases"]["P1"]["transforms"] == ["P1_transform.tfm"]
    assert report["cases"]["P1"]["outputs"] == [
        # `detected` is what "Automatic" decided for THIS file. Which
        # interpolator ran is not visible in the result and is the difference
        # between a label map that survived and one that grew labels nobody
        # segmented, so it is recorded per output rather than per run.
        {"file": "P1_T1_Reg.nii.gz", "detected": "segmentation"},
        # Landmarks are moved, not resampled: no interpolator, nothing to say.
        {"file": "P1_lm_Reg.mrk.json", "points_moved": 2},
    ]


def test_a_landmark_file_reports_how_many_points_moved(tmp_path, transform_file):
    """Which is how a caller learns that a file arrived full of undefined
    points rather than being told it was transformed."""
    source = tmp_path / "in" / "P1_lm.mrk.json"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps({"markups": [{"controlPoints": [
        {"label": "A", "position": [1.0, 0.0, 0.0], "positionStatus": "defined"},
        {"label": "B", "position": [2.0, 0.0, 0.0], "positionStatus": "undefined"},
    ]}]}))
    transform_file(tmp_path / "tfm" / "P1_transform.tfm")

    out = sadt_automatrix.run(files=tmp_path / "in", transforms=tmp_path / "tfm",
                              output_dir=tmp_path / "out")

    assert _report(out)["cases"]["P1"]["outputs"][0]["points_moved"] == 1


def test_a_patient_with_no_transform_produces_no_entry_and_no_file(
        tmp_path, volume, transform_file):
    volume(tmp_path / "in" / "P1_T1.nii.gz")
    volume(tmp_path / "in" / "P2_T1.nii.gz")
    transform_file(tmp_path / "tfm" / "P1_transform.tfm")

    out = sadt_automatrix.run(files=tmp_path / "in", transforms=tmp_path / "tfm",
                              output_dir=tmp_path / "out")

    assert _report(out)["without_a_transform"] == ["P2"]
    assert _outputs(out) == ["P1_T1_Reg.nii.gz"]


def test_the_two_leftover_lists_are_sorted(tmp_path, volume, transform_file):
    """They are read by a person, and an unsorted list of forty is unreadable."""
    for patient in ("P3", "P1", "P2"):
        volume(tmp_path / "in" / f"{patient}_T1.nii.gz")
    for patient in ("P9", "P7", "P8"):
        transform_file(tmp_path / "tfm" / f"{patient}_transform.tfm")
    transform_file(tmp_path / "tfm" / "P1_transform.tfm")

    report = _report(sadt_automatrix.run(
        files=tmp_path / "in", transforms=tmp_path / "tfm",
        output_dir=tmp_path / "out"))

    assert report["without_a_transform"] == ["P2", "P3"]
    assert report["transforms_without_a_file"] == ["P7", "P8", "P9"]


# ---------------------------------------------------------------------------
# Failure
# ---------------------------------------------------------------------------

def test_a_patient_that_fails_does_not_take_the_ones_that_work_with_it(
        tmp_path, volume, transform_file):
    """The successes are written and the failure is named, which is the whole
    difference between a batch tool and a script."""
    volume(tmp_path / "in" / "P1_T1.nii.gz")
    volume(tmp_path / "in" / "P2_T1.nii.gz")
    transform_file(tmp_path / "tfm" / "P1_transform.tfm")
    (tmp_path / "tfm" / "P2_transform.tfm").write_text("not a transform at all")

    out = sadt_automatrix.run(files=tmp_path / "in", transforms=tmp_path / "tfm",
                              output_dir=tmp_path / "out")

    assert _outputs(out) == ["P1_T1_Reg.nii.gz"]
    report = _report(out)
    assert report["summary"]["files_written"] == 1
    assert report["cases"]["P2"]["outputs"] == []
    failure = report["cases"]["P2"]["failed"][0]
    assert "P2_T1.nii.gz" in failure and "P2_transform.tfm" in failure


def test_one_bad_file_of_a_patient_does_not_lose_the_others(tmp_path, volume,
                                                            transform_file):
    volume(tmp_path / "in" / "P1_T1.nii.gz")
    (tmp_path / "in" / "P1_T2.nii.gz").write_bytes(b"not a volume")
    transform_file(tmp_path / "tfm" / "P1_transform.tfm")

    out = sadt_automatrix.run(files=tmp_path / "in", transforms=tmp_path / "tfm",
                              output_dir=tmp_path / "out")

    assert _outputs(out) == ["P1_T1_Reg.nii.gz"]
    assert "P1_T2.nii.gz" in _report(out)["cases"]["P1"]["failed"][0]


def test_when_everything_fails_the_refusal_carries_the_reason(tmp_path, volume):
    """Not the counts: "0 file(s) had no transform" is what hid a transform
    SimpleITK could not read."""
    volume(tmp_path / "in" / "P1_T1.nii.gz")
    (tmp_path / "tfm").mkdir()
    (tmp_path / "tfm" / "P1_transform.tfm").write_text("greetings\n")

    with pytest.raises(ValueError) as failure:
        sadt_automatrix.run(files=tmp_path / "in", transforms=tmp_path / "tfm",
                            output_dir=tmp_path / "out")

    message = str(failure.value)
    assert "AutoMatrix transformed nothing" in message
    assert "P1_T1.nii.gz" in message
    assert "P1_transform.tfm" in message
    assert "neither an ITK transform nor a 4x4 matrix" in message


def test_the_refusal_names_several_failures_not_only_the_first(tmp_path, volume):
    (tmp_path / "tfm").mkdir()
    for patient in ("P1", "P2", "P3"):
        volume(tmp_path / "in" / f"{patient}_T1.nii.gz")
        (tmp_path / "tfm" / f"{patient}_transform.tfm").write_text("greetings\n")

    with pytest.raises(ValueError) as failure:
        sadt_automatrix.run(files=tmp_path / "in", transforms=tmp_path / "tfm",
                            output_dir=tmp_path / "out")

    message = str(failure.value)
    assert all(f"{patient}_T1.nii.gz" in message for patient in ("P1", "P2", "P3"))


def test_nothing_paired_says_so_in_counts_rather_than_in_failures(tmp_path, volume,
                                                                  transform_file):
    """The other half of the same refusal: nothing failed, nothing matched."""
    # TWO patients. One file and one transform are paired outright now -- there
    # is nothing else either could belong to, so pairing them costs no guess --
    # and that rule stops at a cohort, where the guess is how one patient's
    # matrix silently lands on everybody.
    volume(tmp_path / "in" / "P1_T1.nii.gz")
    volume(tmp_path / "in" / "P2_T1.nii.gz")
    transform_file(tmp_path / "tfm" / "P9_transform.tfm")

    with pytest.raises(ValueError, match="2 file\\(s\\) had no transform, "
                                         "1 transform\\(s\\) had no file"):
        sadt_automatrix.run(files=tmp_path / "in", transforms=tmp_path / "tfm",
                            output_dir=tmp_path / "out")


def test_no_report_is_written_when_the_run_produced_nothing(tmp_path, volume,
                                                            transform_file):
    """A report beside no output reads as a run that succeeded and found
    nothing to do."""
    # Two patients: see the test above for why one would now be paired.
    volume(tmp_path / "in" / "P1_T1.nii.gz")
    volume(tmp_path / "in" / "P2_T1.nii.gz")
    transform_file(tmp_path / "tfm" / "P9_transform.tfm")

    with pytest.raises(ValueError):
        sadt_automatrix.run(files=tmp_path / "in", transforms=tmp_path / "tfm",
                            output_dir=tmp_path / "out")

    assert not (tmp_path / "out" / "AutoMatrix_report.json").exists()


def test_an_empty_input_folder_is_refused_by_name(tmp_path, transform_file):
    (tmp_path / "empty").mkdir()
    transform_file(tmp_path / "tfm" / "P1_transform.tfm")

    with pytest.raises(ValueError, match="No scan, segmentation or landmark file "
                                         "found in 'empty'"):
        sadt_automatrix.run(files=tmp_path / "empty", transforms=tmp_path / "tfm",
                            output_dir=tmp_path / "out")


def test_an_empty_transform_folder_is_refused_with_the_extensions_it_wanted(
        tmp_path, volume):
    volume(tmp_path / "in" / "P1_T1.nii.gz")
    (tmp_path / "none").mkdir()

    with pytest.raises(ValueError, match="No transform found in 'none'"):
        sadt_automatrix.run(files=tmp_path / "in", transforms=tmp_path / "none",
                            output_dir=tmp_path / "out")


def test_a_folder_holding_only_untransformable_files_is_refused(tmp_path,
                                                                transform_file):
    (tmp_path / "in").mkdir()
    (tmp_path / "in" / "P1_T1.vtk").write_text("a surface, not a volume")
    (tmp_path / "in" / "P1.json").write_text("{}")
    transform_file(tmp_path / "tfm" / "P1_transform.tfm")

    with pytest.raises(ValueError, match="No scan, segmentation or landmark"):
        sadt_automatrix.run(files=tmp_path / "in", transforms=tmp_path / "tfm",
                            output_dir=tmp_path / "out")


def test_the_module_exports_only_run():
    """`run()` is the entry point the schema is generated from; everything else
    is `pipeline`'s."""
    assert sadt_automatrix.__all__ == ["run"]
    assert callable(sadt_automatrix.run)
    assert callable(pipeline.read_transform)


# ---------------------------------------------------------------------------
# What the package is allowed to be made of
# ---------------------------------------------------------------------------

def _imported_modules():
    """Every top-level module the package imports, lazy imports included."""
    import ast
    import pathlib

    found = set()
    root = pathlib.Path(sadt_automatrix.__file__).parent
    for source in sorted(root.rglob("*.py")):
        for node in ast.walk(ast.parse(source.read_text())):
            if isinstance(node, ast.Import):
                found.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                found.add(node.module.split(".")[0])
    return found


def test_the_whole_tool_is_simpleitk_and_the_shared_pairing_rule():
    """The manifest declares 1.4 GB of models for AutoMatrix and this port needs
    none of them: it is a resampler. Pinned as a set rather than as a list of
    forbidden names, so a dependency added without a thought fails here whatever
    it is called."""
    third_party = _imported_modules() - {
        "json", "logging", "os", "time", "pathlib", "typing",
    }

    # numpy is here for `looks_like_a_label_map`, which is what lets the
    # interpolator be chosen per file rather than per batch.
    assert third_party == {"SimpleITK", "numpy", "sadt_areg_common"}


def test_nothing_here_reaches_the_network():
    """Said separately from the assertion above because it is the reason for it:
    a server holding patient data does not make outbound calls mid-request."""
    assert not _imported_modules() & {
        "urllib", "requests", "http", "socket", "ftplib", "httpx", "aiohttp",
    }
