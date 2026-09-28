"""AutoMatrix on synthetic volumes: SimpleITK does the work, for real."""

import json
import logging
import os
import sys

import numpy as np
import pytest
import SimpleITK as sitk

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import sadt_automatrix
from sadt_automatrix import pipeline


def _volume(path, value=7):
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.zeros((6, 6, 6), np.int16)
    array[2:4, 2:4, 2:4] = value
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((1.0, 1.0, 1.0))
    sitk.WriteImage(image, str(path))
    return path


def _transform(path, translation=(1.0, 0.0, 0.0)):
    path.parent.mkdir(parents=True, exist_ok=True)
    tfm = sitk.TranslationTransform(3, translation)
    sitk.WriteTransform(tfm, str(path))
    return path


def _landmarks(path, points):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"markups": [{"controlPoints": [
        {"label": f"L{i}", "position": list(p), "positionStatus": "defined"}
        for i, p in enumerate(points)]}]}))
    return path


# ---------------------------------------------------------------------------
# The rule that matters most
# ---------------------------------------------------------------------------

def test_a_segmentation_is_resampled_without_inventing_labels(tmp_path):
    """Nearest neighbour for a label map, linear otherwise. Interpolating
    labels linearly produces values that were never in the input -- a voxel
    half-way between label 1 and label 3 becomes label 2, which is a different
    anatomical structure."""
    array = np.zeros((8, 8, 8), np.int16)
    array[2:5, 2:6, 2:6] = 1
    array[5:8, 2:6, 2:6] = 3
    image = sitk.GetImageFromArray(array)
    # Not half a voxel: a 0.5 shift lands exactly on the boundary, where the
    # blend of 1 and 3 truncates back to an existing label and hides the
    # effect. An off-grid shift is what a real registration produces.
    shift = sitk.TranslationTransform(3, (0.3, 0.3, 0.3))

    as_labels = sitk.GetArrayFromImage(pipeline.resample(image, shift, is_segmentation=True))
    as_scan = sitk.GetArrayFromImage(pipeline.resample(image, shift, is_segmentation=False))

    assert set(np.unique(as_labels)) <= {0, 1, 3}, "nearest neighbour invented a label"
    assert 2 in set(np.unique(as_scan)), (
        "linear interpolation did not blend, so this test no longer demonstrates "
        "what the nearest-neighbour rule protects against"
    )


def test_landmarks_move_by_the_inverse(tmp_path):
    """A transform produced by a registration maps IMAGE space; a point follows
    the opposite way. Upstream applies the inverse and so does this."""
    source = _landmarks(tmp_path / "P1_lm.mrk.json", [(10.0, 0.0, 0.0)])
    tfm = sitk.TranslationTransform(3, (4.0, 0.0, 0.0))
    out = tmp_path / "out.mrk.json"

    moved = pipeline.apply_to_landmarks(str(source), tfm, str(out))

    assert moved == 1
    position = json.loads(out.read_text())["markups"][0]["controlPoints"][0]["position"]
    assert position == pytest.approx([6.0, 0.0, 0.0]), "point moved forwards, not by the inverse"


def test_a_point_that_is_not_defined_is_left_alone(tmp_path):
    source = tmp_path / "P1_lm.mrk.json"
    source.write_text(json.dumps({"markups": [{"controlPoints": [
        {"label": "A", "position": [1.0, 2.0, 3.0], "positionStatus": "undefined"},
        {"label": "B", "position": [10.0, 0.0, 0.0], "positionStatus": "defined"},
    ]}]}))
    out = tmp_path / "out.mrk.json"

    assert pipeline.apply_to_landmarks(str(source), sitk.TranslationTransform(3, (4.0, 0, 0)), str(out)) == 1
    points = json.loads(out.read_text())["markups"][0]["controlPoints"]
    assert points[0]["position"] == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# Discovery and pairing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,image,landmark", [
    ("P1_T1.nii.gz", True, False),
    ("P1_T1.nrrd", True, False),
    ("P1_lm_Pred.mrk.json", False, True),
    # A landmark file ends in .json but must never be read as an image.
    ("P1_something.json", False, False),
    ("notes.txt", False, False),
])
def test_what_counts_as_transformable(name, image, landmark):
    assert pipeline.is_image_file(name) is image
    assert pipeline.is_landmark_file(name) is landmark


def test_a_patient_with_no_transform_is_named_not_dropped(tmp_path):
    """Upstream skipped such a file in silence, so a run could transform 3 of
    40 and look complete."""
    _volume(tmp_path / "in" / "P1_T1.nii.gz")
    _volume(tmp_path / "in" / "P2_T1.nii.gz")
    _transform(tmp_path / "tfm" / "P1_transform.tfm")

    out = sadt_automatrix.run(files=tmp_path / "in", transforms=tmp_path / "tfm",
                              output_dir=tmp_path / "out")

    report = json.loads((out / "AutoMatrix_report.json").read_text())
    assert report["without_a_transform"] == ["P2"]
    assert list(report["cases"]) == ["P1"]


def test_a_transform_with_no_file_is_named_too(tmp_path):
    _volume(tmp_path / "in" / "P1_T1.nii.gz")
    _transform(tmp_path / "tfm" / "P1_transform.tfm")
    _transform(tmp_path / "tfm" / "P9_transform.tfm")

    out = sadt_automatrix.run(files=tmp_path / "in", transforms=tmp_path / "tfm",
                              output_dir=tmp_path / "out")

    report = json.loads((out / "AutoMatrix_report.json").read_text())
    assert report["transforms_without_a_file"] == ["P9"]


def test_several_transforms_of_one_patient_are_all_applied(tmp_path):
    """And they have to be told apart, which is what the name flag is for."""
    _volume(tmp_path / "in" / "P1_T1.nii.gz")
    _transform(tmp_path / "tfm" / "P1_CBReg_matrix.tfm", (1.0, 0, 0))
    _transform(tmp_path / "tfm" / "P1_MANDReg_matrix.tfm", (0, 2.0, 0))

    out = sadt_automatrix.run(files=tmp_path / "in", transforms=tmp_path / "tfm",
                              output_dir=tmp_path / "out",
                              name_output_after_transform=True)

    written = sorted(p.name for p in out.rglob("*.nii.gz"))
    assert len(written) == 2, f"expected one output per transform, got {written}"
    assert len(set(written)) == 2, "two transforms wrote the same file name"


def test_nothing_transformable_is_refused(tmp_path):
    (tmp_path / "in").mkdir()
    (tmp_path / "in" / "notes.txt").write_text("nothing")
    _transform(tmp_path / "tfm" / "P1_transform.tfm")

    with pytest.raises(ValueError, match="No scan, segmentation or landmark"):
        sadt_automatrix.run(files=tmp_path / "in", transforms=tmp_path / "tfm",
                            output_dir=tmp_path / "out")


def test_no_transform_at_all_is_refused(tmp_path):
    _volume(tmp_path / "in" / "P1_T1.nii.gz")
    (tmp_path / "tfm").mkdir()

    with pytest.raises(ValueError, match="No transform found"):
        sadt_automatrix.run(files=tmp_path / "in", transforms=tmp_path / "tfm",
                            output_dir=tmp_path / "out")


def test_the_input_tree_is_mirrored(tmp_path):
    _volume(tmp_path / "in" / "siteA" / "P1_T1.nii.gz")
    _transform(tmp_path / "tfm" / "P1_transform.tfm")

    out = sadt_automatrix.run(files=tmp_path / "in", transforms=tmp_path / "tfm",
                              output_dir=tmp_path / "out")

    assert list((out / "siteA").glob("*.nii.gz")), "the subfolder was flattened"


# ---------------------------------------------------------------------------
# The two shapes a transform arrives in
# ---------------------------------------------------------------------------

def test_a_bare_4x4_matrix_is_read(tmp_path):
    """Greedy writes four lines of numbers and calls it `.mat`; so does
    upstream's own `writeIdentityInit`. `sitk.ReadTransform` refuses it with a
    MatlabTransformIO error, which is why AutoMatrix could not consume what
    GreedyReg produced."""
    path = tmp_path / "P1_transform.mat"
    path.write_text("1 0 0 3.5\n0 1 0 0\n0 0 1 0\n0 0 0 1\n")

    transform = pipeline.read_transform(str(path))

    assert transform.TransformPoint((0.0, 0.0, 0.0)) == pytest.approx([3.5, 0.0, 0.0])


def test_an_itk_transform_is_still_read(tmp_path):
    path = _transform(tmp_path / "P1_transform.tfm", (2.0, 0.0, 0.0))
    assert pipeline.read_transform(str(path)).TransformPoint((0.0, 0.0, 0.0)) == pytest.approx([2.0, 0, 0])


def test_a_file_that_is_neither_says_so(tmp_path):
    path = tmp_path / "P1_transform.mat"
    path.write_text("this is not a transform\n")

    with pytest.raises(RuntimeError, match="neither an ITK transform nor a 4x4 matrix"):
        pipeline.read_transform(str(path))


def test_the_refusal_carries_why_nothing_was_written(tmp_path):
    """Reporting only the counts hid a transform SimpleITK could not read
    behind "0 file(s) had no transform"."""
    _volume(tmp_path / "in" / "P1_T1.nii.gz")
    (tmp_path / "tfm").mkdir()
    (tmp_path / "tfm" / "P1_transform.tfm").write_text("not a transform at all")

    with pytest.raises(ValueError, match="neither an ITK transform"):
        sadt_automatrix.run(files=tmp_path / "in", transforms=tmp_path / "tfm",
                            output_dir=tmp_path / "out")


def test_a_batch_says_which_patient_it_is_on(tmp_path, monkeypatch):
    """One event per patient -- including one that has no transform.

    The loop reports before it decides whether there is anything to do, which
    is what keeps the count in the message equal to the count a caller sent.
    """
    events_file = tmp_path / "events.jsonl"
    monkeypatch.setenv("SADT_PROGRESS_FILE", str(events_file))
    _volume(tmp_path / "in" / "P1_T1.nii.gz")
    _volume(tmp_path / "in" / "P2_T1.nii.gz")
    _transform(tmp_path / "tfm" / "P1_transform.tfm")

    sadt_automatrix.run(files=tmp_path / "in", transforms=tmp_path / "tfm",
                        output_dir=tmp_path / "out")

    events = [json.loads(line) for line in events_file.read_text().splitlines() if line]
    assert [e["message"] for e in events] == ["patient 1 of 2", "patient 2 of 2"]
    assert [e["fraction"] for e in events] == [0.0, 0.5]


def test_a_failure_names_the_position_and_never_the_file(tmp_path, caplog):
    """Both counters, and never the name the caller gave the file.

    A tool's stderr is captured to a file in the job directory, and on a FAILED
    run the server copies its tail into its own persistent log -- so a name
    written on this path outlives the run and its job directory. The report
    still names the file under `failed`; that goes back to whoever sent it.
    """
    (tmp_path / "in").mkdir()
    (tmp_path / "in" / "MAMP_0001_T1.nii.gz").write_bytes(b"not a volume at all")
    _transform(tmp_path / "tfm" / "MAMP_0001_transform.tfm")
    _volume(tmp_path / "in" / "P2_T1.nii.gz")
    _transform(tmp_path / "tfm" / "P2_transform.tfm")

    with caplog.at_level(logging.INFO, logger="AutoMatrix"):
        sadt_automatrix.run(files=tmp_path / "in", transforms=tmp_path / "tfm",
                            output_dir=tmp_path / "out")

    messages = [record.getMessage() for record in caplog.records]
    assert "AutoMatrix failed on patient 1 of 2, file 1 of 1" in messages, messages
    assert not any("MAMP_0001" in m for m in messages), messages

    report = json.loads((tmp_path / "out" / "AutoMatrix_report.json").read_text())
    assert report["cases"]["MAMP_0001"]["failed"], "the report still names it"


# ---------------------------------------------------------------------------
# Compatibility with SlicerAutomatedDentalTools.
#
# All four of the datasets published with the legacy module paired under its
# rules and under none of this port's, so every one of them transformed nothing.
# These pin the fallback that fixes that, and -- the case that matters more --
# that it stays a fallback.
# ---------------------------------------------------------------------------


def test_legacy_names_pair_as_a_fallback(tmp_path):
    """Upstream cuts a name at `_Left`; this port keeps whole tokens.

    `P1_T1_Left_MA.tfm` keys to `P1_Left_MA` here and to `P1` upstream, so
    without the fallback this pairs with nothing.
    """
    _volume(tmp_path / "in" / "P1_T1_.nii.gz")
    _transform(tmp_path / "tfm" / "P1_T1_Left_MA.tfm")

    out = sadt_automatrix.run(files=tmp_path / "in", transforms=tmp_path / "tfm",
                              output_dir=tmp_path / "out")

    report = json.loads((out / "AutoMatrix_report.json").read_text())
    assert report["summary"]["files_written"] == 1
    assert report["paired_by"] == {"P1": "legacy file names"}


def test_one_transform_reaches_a_cohort_only_when_asked(tmp_path):
    """The mirror matrix: it belongs to no patient, so no name can pair it.

    VFACE drives exactly this eight times a run, one `Mirror.tfm` against a
    folder of patients. Upstream infers it from the argument being a single
    FILE; here the caller says so, because the same shape is what a mislabelled
    per-patient transform looks like.
    """
    _volume(tmp_path / "in" / "P1_T1.nii.gz")
    _volume(tmp_path / "in" / "P2_T1.nii.gz")
    _transform(tmp_path / "Mirror.tfm")

    out = sadt_automatrix.run(files=tmp_path / "in", transforms=tmp_path / "Mirror.tfm",
                              output_dir=tmp_path / "out",
                              same_transform_for_every_patient=True)

    report = json.loads((out / "AutoMatrix_report.json").read_text())
    assert report["summary"]["files_written"] == 2
    assert set(report["paired_by"]) == {"P1", "P2"}
    assert report["without_a_transform"] == []


def test_a_cohort_is_never_given_one_transform_by_guess(tmp_path):
    """The failure that must stay loud, for the tools that will call this.

    One transform, three patients, a name neither rule recognises. Guessing
    puts one patient's matrix on everybody and reports a clean success -- the
    legacy module's own callers were bitten by precisely this. Refusing is the
    behaviour this port had before the compatibility fallback existed, and it
    is kept.
    """
    for name in ("alpha", "beta", "gamma"):
        _volume(tmp_path / "in" / f"{name}_T1.nii.gz")
    _transform(tmp_path / "alpha-subject.tfm")

    with pytest.raises(ValueError, match="transformed nothing"):
        sadt_automatrix.run(files=tmp_path / "in",
                            transforms=tmp_path / "alpha-subject.tfm",
                            output_dir=tmp_path / "out")


def test_one_transform_and_one_patient_needs_no_flag(tmp_path):
    """Nothing else it could belong to, so there is no guess to refuse.

    This is what makes the legacy module's single-file datasets run unchanged.
    """
    _volume(tmp_path / "in" / "P1_T1.nii.gz")
    _transform(tmp_path / "anything.tfm")

    out = sadt_automatrix.run(files=tmp_path / "in", transforms=tmp_path / "anything.tfm",
                              output_dir=tmp_path / "out")

    assert json.loads((out / "AutoMatrix_report.json").read_text())[
        "summary"]["files_written"] == 1


def test_the_fallback_never_overrides_a_pair_this_port_found(tmp_path):
    """The guarantee the whole design rests on: additive, never a replacement.

    One transform names P1 and two patients are present. Upstream would give it
    to both -- a single file is broadcast there whatever the names say. Here the
    normal rule pairs P1, so the fallback is never reached and P2 keeps having
    no transform, exactly as it did before the fallback existed.
    """
    _volume(tmp_path / "in" / "P1_T1.nii.gz")
    _volume(tmp_path / "in" / "P2_T1.nii.gz")
    _transform(tmp_path / "P1_transform.tfm")

    out = sadt_automatrix.run(files=tmp_path / "in", transforms=tmp_path / "P1_transform.tfm",
                              output_dir=tmp_path / "out")

    report = json.loads((out / "AutoMatrix_report.json").read_text())
    assert report["summary"]["files_written"] == 1
    assert report["without_a_transform"] == ["P2"]
    assert "paired_by" not in report, "the fallback ran when it was not needed"


# ---------------------------------------------------------------------------
# Scan or segmentation, read off the file.
#
# The argument was per RUN, so a folder holding both could never be right for
# both, and a clinician had to answer a question about interpolation to use the
# tool at all. What settles it is in the data: a label map holds tens of whole,
# non-negative values where a CBCT holds thousands and reaches below zero.
# ---------------------------------------------------------------------------


def _labelled(path):
    """A volume that is a label map by content: two structures, no background
    gradient, nothing negative."""
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.zeros((8, 8, 8), np.int16)
    array[2:5, 2:6, 2:6] = 1
    array[5:8, 2:6, 2:6] = 3
    sitk.WriteImage(sitk.GetImageFromArray(array), str(path))
    return path


def _scanlike(path):
    """A volume that is a scan by content: Hounsfield numbers, negatives and all."""
    path.parent.mkdir(parents=True, exist_ok=True)
    array = (np.arange(8 * 8 * 8).reshape(8, 8, 8) * 7 - 1000).astype(np.int16)
    sitk.WriteImage(sitk.GetImageFromArray(array), str(path))
    return path


def _outputs(out):
    report = json.loads((out / "AutoMatrix_report.json").read_text())
    return {o["file"]: o for entry in report["cases"].values()
            for o in entry["outputs"]}


def test_a_label_map_is_recognised_and_never_blended(tmp_path):
    _labelled(tmp_path / "in" / "P1_T1.nii.gz")
    _transform(tmp_path / "tfm" / "P1_transform.tfm", translation=(0.3, 0.3, 0.3))

    out = sadt_automatrix.run(files=tmp_path / "in", transforms=tmp_path / "tfm",
                              output_dir=tmp_path / "out")

    assert _outputs(out)["P1_T1_Reg.nii.gz"]["detected"] == "segmentation"
    written = sitk.GetArrayFromImage(sitk.ReadImage(str(out / "P1_T1_Reg.nii.gz")))
    assert set(np.unique(written)) <= {0, 1, 3}, "a label nobody segmented was invented"


def test_a_scan_is_recognised_by_what_is_in_it(tmp_path):
    _scanlike(tmp_path / "in" / "P1_T1.nii.gz")
    _transform(tmp_path / "tfm" / "P1_transform.tfm", translation=(0.3, 0.3, 0.3))

    out = sadt_automatrix.run(files=tmp_path / "in", transforms=tmp_path / "tfm",
                              output_dir=tmp_path / "out")

    assert _outputs(out)["P1_T1_Reg.nii.gz"]["detected"] == "scan"


def test_one_folder_of_both_is_resampled_each_its_own_way(tmp_path):
    """What the per-run argument could never do, and the reason for the change.

    VFACE runs AutoMatrix once per structure precisely because one answer had to
    cover every file in the folder.
    """
    _labelled(tmp_path / "in" / "P1_T1_Seg.nii.gz")
    _scanlike(tmp_path / "in" / "P1_T1.nii.gz")
    _transform(tmp_path / "tfm" / "P1_transform.tfm", translation=(0.3, 0.3, 0.3))

    out = sadt_automatrix.run(files=tmp_path / "in", transforms=tmp_path / "tfm",
                              output_dir=tmp_path / "out")

    detected = {name: o["detected"] for name, o in _outputs(out).items()}
    assert detected == {"P1_T1_Seg_Reg.nii.gz": "segmentation",
                        "P1_T1_Reg.nii.gz": "scan"}


def test_a_caller_who_names_the_content_is_believed(tmp_path):
    """The escape hatch, and it must beat the data: a volume that LOOKS like a
    label map but is meant as a scan is resampled linearly when asked."""
    _labelled(tmp_path / "in" / "P1_T1.nii.gz")
    _transform(tmp_path / "tfm" / "P1_transform.tfm", translation=(0.3, 0.3, 0.3))

    out = sadt_automatrix.run(files=tmp_path / "in", transforms=tmp_path / "tfm",
                              output_dir=tmp_path / "out", content="Scan")

    assert "detected" not in _outputs(out)["P1_T1_Reg.nii.gz"]
    written = sitk.GetArrayFromImage(sitk.ReadImage(str(out / "P1_T1_Reg.nii.gz")))
    assert 2 in set(np.unique(written)), "the named content was overruled by the data"
