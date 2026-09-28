"""The batch: what is written, what is reported, and what one failure costs.

Every test here stubs greedy. What is exercised is the half of the tool that
upstream got wrong -- the naming, the report, and the fact that patient 3
failing used to lose patients 4 to 40.
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

import sadt_greedyreg
from conftest import Recorder, cohort, make_scan


def report_of(output_dir):
    return json.loads((Path(output_dir) / "GreedyReg_report.json").read_text())


def run_batch(tmp_path, patients=("A1",), **kwargs):
    t1, t2 = cohort(tmp_path, patients)
    kwargs.setdefault("output_dir", tmp_path / "out")
    return sadt_greedyreg.run(t1=t1, t2=t2, **kwargs)


# ---------------------------------------------------------------------------
# What lands in the output folder
# ---------------------------------------------------------------------------

def test_each_patient_gets_a_registered_volume_and_a_transform(tmp_path, greedy):
    out = run_batch(tmp_path, ["A1", "B2"])

    for patient in ("A1", "B2"):
        assert (out / f"{patient}_registered.nii.gz").exists()
        assert (out / f"{patient}_transform.mat").exists()


def test_the_transform_is_written_beside_the_resampled_volume(tmp_path, greedy):
    """README's sentence, and the reason a caller can find one from the other."""
    out = run_batch(tmp_path)

    assert (out / "A1_registered.nii.gz").parent == (out / "A1_transform.mat").parent


def test_the_suffix_names_the_registered_volume(tmp_path, greedy):
    out = run_batch(tmp_path, output_suffix="ontoT1")

    assert (out / "A1_ontoT1.nii.gz").exists()
    assert not (out / "A1_registered.nii.gz").exists()


def test_the_transform_is_not_suffixed(tmp_path, greedy):
    """`<patient>_transform.mat` whatever the volume is called: the suffix
    describes the resampling, and a transform is a transform."""
    out = run_batch(tmp_path, output_suffix="ontoT1")

    assert (out / "A1_transform.mat").exists()


def test_the_output_directory_is_created_when_it_is_absent(tmp_path, greedy):
    out = run_batch(tmp_path, output_dir=tmp_path / "deep" / "and" / "absent")

    assert out.is_dir()
    assert (out / "GreedyReg_report.json").exists()


def test_run_returns_the_output_directory(tmp_path, greedy):
    out = run_batch(tmp_path, output_dir=tmp_path / "out")

    assert out == Path(tmp_path / "out")


def test_the_folders_may_be_given_as_strings(tmp_path, greedy):
    """The server passes Paths; a `sup` call or a test passes what it has."""
    t1, t2 = cohort(tmp_path, ["A1"])

    out = sadt_greedyreg.run(t1=str(t1), t2=str(t2), output_dir=str(tmp_path / "out"))

    assert (Path(out) / "A1_registered.nii.gz").exists()


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

def test_the_report_is_json_at_the_root_of_the_output(tmp_path, greedy):
    out = run_batch(tmp_path)

    assert json.loads((out / "GreedyReg_report.json").read_text())["tool"] == "GreedyReg"


def test_the_report_records_the_choices_the_run_was_made_with(tmp_path, greedy):
    out = run_batch(tmp_path, metric="SSD", transform_type="Affine", output_suffix="reg")

    report = report_of(out)
    assert report["metric"] == "SSD"
    assert report["transform_type"] == "Affine"
    assert report["output_suffix"] == "reg"


def test_the_report_counts_what_happened(tmp_path, greedy):
    out = run_batch(tmp_path, ["A1", "B2", "C3"])

    assert report_of(out)["summary"] == {"cases": 3, "registered": 3, "failed": 0}


def test_the_report_times_the_run(tmp_path, greedy):
    out = run_batch(tmp_path)

    duration = report_of(out)["duration_seconds"]
    assert isinstance(duration, float) and duration >= 0


def test_a_patient_entry_names_the_two_scans_it_registered(tmp_path, greedy):
    out = run_batch(tmp_path)

    entry = report_of(out)["cases"]["A1"]
    assert entry["t1"] == "A1_T1.nii.gz"
    assert entry["t2"] == "A1_T2.nii.gz"
    assert entry["status"] == "ok"


def test_a_patient_entry_lists_the_files_it_produced(tmp_path, greedy):
    out = run_batch(tmp_path)

    assert report_of(out)["cases"]["A1"]["produced"] == [
        "A1_registered.nii.gz", "A1_transform.mat"]


def test_the_report_says_which_way_the_transform_maps(tmp_path, greedy):
    """A transform with no stated direction is a transform someone will apply
    backwards; greedy's own `-r` consumes it in this direction."""
    out = run_batch(tmp_path)

    assert "T2" in report_of(out)["cases"]["A1"]["transform_maps"]
    assert "-r" in report_of(out)["cases"]["A1"]["transform_maps"]


def test_the_report_keys_patients_by_the_shared_patient_key(tmp_path, greedy):
    out = run_batch(tmp_path, ["A1", "B2"])

    assert sorted(report_of(out)["cases"]) == ["A1", "B2"]


# ---------------------------------------------------------------------------
# The commands the batch actually issues
# ---------------------------------------------------------------------------

def test_the_registration_runs_before_the_resample(tmp_path, greedy):
    run_batch(tmp_path)

    assert "-a" in greedy.commands[0]
    assert "-rm" in greedy.commands[1]


def test_the_resample_consumes_the_transform_the_registration_wrote(tmp_path, greedy):
    run_batch(tmp_path)

    written = greedy.commands[0][greedy.commands[0].index("-o") + 1]
    consumed = greedy.commands[1][greedy.commands[1].index("-r") + 1]
    assert written == consumed


def test_the_moving_image_is_the_second_timepoint(tmp_path, greedy):
    """T2 onto T1: the fixed image is the first timepoint's, and getting this
    the wrong way round is a run that succeeds and answers backwards."""
    run_batch(tmp_path)

    command = greedy.registration_for("A1")
    assert command[command.index("-i") + 1].endswith("A1_T1.nii.gz")
    assert command[command.index("-i") + 2].endswith("A1_T2.nii.gz")


def test_the_chosen_metric_and_transform_reach_greedy(tmp_path, greedy):
    run_batch(tmp_path, metric="NMI", transform_type="Affine")

    command = greedy.registration_for("A1")
    assert command[command.index("-m") + 1] == "NMI"
    assert command[command.index("-dof") + 1] == "12"


def test_an_unknown_metric_is_refused_before_anything_runs(tmp_path, greedy):
    """One error, not forty identical per-patient failures -- and no output
    directory full of half a cohort."""
    t1, t2 = cohort(tmp_path, ["A1", "B2"])

    with pytest.raises(ValueError, match="Unknown metric"):
        sadt_greedyreg.run(t1=t1, t2=t2, output_dir=tmp_path / "out", metric="ncc")

    assert not greedy.commands
    assert not (tmp_path / "out").exists()


def test_an_unknown_transform_type_is_refused_before_anything_runs(tmp_path, greedy):
    t1, t2 = cohort(tmp_path, ["A1"])

    with pytest.raises(ValueError, match="Unknown transform_type"):
        sadt_greedyreg.run(t1=t1, t2=t2, output_dir=tmp_path / "out", transform_type="rigid")

    assert not greedy.commands


# ---------------------------------------------------------------------------
# The initial transform
# ---------------------------------------------------------------------------

def test_without_an_initial_transform_the_search_starts_from_the_nudged_identity(
        tmp_path, monkeypatch):
    """The identity is written to scratch and deleted with it, so what it held
    has to be read while the run is in flight."""
    seen = {}

    class Reading(Recorder):
        def __call__(self, command):
            if "-ia" in command:
                seen["init"] = Path(command[list(command).index("-ia") + 1]).read_text()
            return super().__call__(command)

    recorder = Reading()
    monkeypatch.setattr(sadt_greedyreg, "run_greedy", recorder)
    run_batch(tmp_path)

    matrix = np.array([[float(v) for v in line.split()] for line in seen["init"].splitlines()])
    assert matrix.shape == (4, 4)
    assert matrix[0, 3] == 0.001


def test_without_an_initial_transform_the_report_names_none(tmp_path, greedy):
    out = run_batch(tmp_path)

    assert "initial_transform" not in report_of(out)["cases"]["A1"]


@pytest.mark.parametrize("transform_name", [
    "A1.mat",                # the patient, and nothing else
    "A1_T1.mat",             # named after the timepoint it starts from
    "A1_transform.mat",      # what THIS tool writes
    "A1_T1_transform.mat",   # and what it writes for a timepoint-named scan
])
def test_an_initial_transform_is_passed_to_greedy_and_reported(
        tmp_path, greedy, transform_name):
    """`<patient>_transform.mat` is what this tool writes, so it is what a
    caller hands back for a second pass. Keyed on a bare patient stem it read
    as a patient called `A1_transform`, matched nobody, and every patient
    silently restarted from identity."""
    t1, t2 = cohort(tmp_path, ["A1"])
    (tmp_path / "init").mkdir()
    (tmp_path / "init" / transform_name).write_text("1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n")

    out = sadt_greedyreg.run(t1=t1, t2=t2, output_dir=tmp_path / "out",
                             initial_transforms=tmp_path / "init")

    command = greedy.registration_for("A1")
    assert command[command.index("-ia") + 1].endswith(transform_name)
    assert report_of(out)["cases"]["A1"]["initial_transform"] == transform_name


def test_a_transform_carrying_an_unknown_token_is_a_different_patient(tmp_path, greedy):
    """The same rule the scans obey: `A1_start` is no more `A1` than
    `A1_other.nii.gz` is. It is reported rather than silently ignored, which is
    the only way a caller sees that their naming missed."""
    t1, t2 = cohort(tmp_path, ["A1"])
    (tmp_path / "init").mkdir()
    (tmp_path / "init" / "A1_start.mat").write_text("1 0 0 0\n")

    out = sadt_greedyreg.run(t1=t1, t2=t2, output_dir=tmp_path / "out",
                             initial_transforms=tmp_path / "init")

    assert report_of(out)["unused_initial_transforms"] == ["A1_start"]
    assert "initial_transform" not in report_of(out)["cases"]["A1"]


def test_an_initial_transform_that_matches_nobody_is_reported(tmp_path, greedy):
    t1, t2 = cohort(tmp_path, ["A1"])
    (tmp_path / "init").mkdir()
    (tmp_path / "init" / "Z9_transform.mat").write_text("1 0 0 0\n")

    out = sadt_greedyreg.run(t1=t1, t2=t2, output_dir=tmp_path / "out",
                             initial_transforms=tmp_path / "init")

    assert report_of(out)["unused_initial_transforms"] == ["Z9"]
    assert "initial_transform" not in report_of(out)["cases"]["A1"]


def test_files_that_are_not_transforms_are_not_read_as_transforms(tmp_path, greedy):
    t1, t2 = cohort(tmp_path, ["A1"])
    (tmp_path / "init").mkdir()
    (tmp_path / "init" / "A1_notes.txt").write_text("not a transform\n")

    out = sadt_greedyreg.run(t1=t1, t2=t2, output_dir=tmp_path / "out",
                             initial_transforms=tmp_path / "init")

    assert "initial_transform" not in report_of(out)["cases"]["A1"]
    assert report_of(out)["unused_initial_transforms"] == []


def test_a_nested_initial_transform_follows_the_patient_into_its_subfolder(tmp_path, greedy):
    make_scan(tmp_path / "t1" / "site_a" / "A1_T1.nii.gz")
    make_scan(tmp_path / "t2" / "site_a" / "A1_T2.nii.gz")
    (tmp_path / "init" / "site_a").mkdir(parents=True)
    (tmp_path / "init" / "site_a" / "A1_transform.mat").write_text("1 0 0 0\n")

    out = sadt_greedyreg.run(t1=tmp_path / "t1", t2=tmp_path / "t2",
                             output_dir=tmp_path / "out",
                             initial_transforms=tmp_path / "init")

    assert report_of(out)["cases"]["site_a/A1"]["initial_transform"] == "A1_transform.mat"


# ---------------------------------------------------------------------------
# The mask
# ---------------------------------------------------------------------------

def test_without_masks_no_metric_mask_is_passed(tmp_path, greedy):
    out = run_batch(tmp_path)

    assert "-gm" not in greedy.registration_for("A1")
    assert "mask" not in report_of(out)["cases"]["A1"]


@pytest.mark.parametrize("mask_name", [
    "A1_T1.nii.gz",             # named exactly like the scan
    "A1_mask.nii.gz",           # what a clinician types
    "A1_seg.nii.gz",            # what a segmentation step leaves
    "A1_T1_MAND_seg.nii.gz",    # what AMASSS writes
    "A1_seg_CBMASK.nii.gz",     # and its other naming
])
def test_a_mask_is_found_however_it_names_itself(tmp_path, greedy, mask_name):
    """Keyed on a bare patient stem, every one of these but the first read as
    its own patient -- so the argument was inert and the run went ahead
    unmasked without saying so."""
    t1, t2 = cohort(tmp_path, ["A1"])
    make_scan(tmp_path / "masks" / mask_name, data=np.ones((4, 4, 4), np.float32))

    out = sadt_greedyreg.run(t1=t1, t2=t2, output_dir=tmp_path / "out",
                             masks=tmp_path / "masks")

    assert "-gm" in greedy.registration_for("A1")
    assert report_of(out)["cases"]["A1"]["mask"] == mask_name
    assert report_of(out)["unused_masks"] == []


def test_greedy_is_given_the_binarised_copy_not_the_callers_file(tmp_path, greedy):
    """The mask belongs to the caller and greedy wants a float 0/1 volume, so
    what `-gm` names is the scratch copy."""
    t1, t2 = cohort(tmp_path, ["A1"])
    original = make_scan(tmp_path / "masks" / "A1_mask.nii.gz",
                         data=np.array([[[0, 2, 5]]], dtype=np.int16))

    sadt_greedyreg.run(t1=t1, t2=t2, output_dir=tmp_path / "out", masks=tmp_path / "masks")

    command = greedy.registration_for("A1")
    given = command[command.index("-gm") + 1]
    assert given != str(original)
    assert os.path.basename(given) == "mask.nii.gz"


def test_the_mask_greedy_receives_holds_only_zero_and_one(tmp_path, monkeypatch):
    """Read during the run: the binarised copy lives in the scratch directory,
    which is removed as soon as the patient is done."""
    seen = {}

    class Reading(Recorder):
        def __call__(self, command):
            if "-gm" in command:
                image = nib.load(command[list(command).index("-gm") + 1])
                seen["values"] = sorted(np.unique(image.get_fdata()))
                seen["dtype"] = image.get_data_dtype()
            return super().__call__(command)

    monkeypatch.setattr(sadt_greedyreg, "run_greedy", Reading())
    t1, t2 = cohort(tmp_path, ["A1"])
    make_scan(tmp_path / "masks" / "A1_mask.nii.gz",
              data=np.array([[[0, 2, 5]]], dtype=np.int16))

    sadt_greedyreg.run(t1=t1, t2=t2, output_dir=tmp_path / "out", masks=tmp_path / "masks")

    assert seen["values"] == [0.0, 1.0]
    assert seen["dtype"] == np.float32


def test_a_mask_that_matches_nobody_is_reported_rather_than_dropped(tmp_path, greedy):
    """A run that quietly registered without the mask it was handed looks,
    from the outside, exactly like one that used it."""
    t1, t2 = cohort(tmp_path, ["A1"])
    make_scan(tmp_path / "masks" / "Z9_mask.nii.gz", data=np.ones((4, 4, 4), np.float32))

    out = sadt_greedyreg.run(t1=t1, t2=t2, output_dir=tmp_path / "out",
                             masks=tmp_path / "masks")

    assert report_of(out)["unused_masks"] == ["Z9"]
    assert "-gm" not in greedy.registration_for("A1")


def test_one_patients_mask_does_not_become_anothers(tmp_path, greedy):
    t1, t2 = cohort(tmp_path, ["A1", "B2"])
    make_scan(tmp_path / "masks" / "A1_mask.nii.gz", data=np.ones((4, 4, 4), np.float32))

    out = sadt_greedyreg.run(t1=t1, t2=t2, output_dir=tmp_path / "out",
                             masks=tmp_path / "masks")

    assert "-gm" in greedy.registration_for("A1")
    assert "-gm" not in greedy.registration_for("B2")
    assert "mask" not in report_of(out)["cases"]["B2"]


def test_the_unused_lists_are_present_even_when_nothing_was_given(tmp_path, greedy):
    out = run_batch(tmp_path)

    assert report_of(out)["unused_masks"] == []
    assert report_of(out)["unused_initial_transforms"] == []


# ---------------------------------------------------------------------------
# Failure, which must cost one patient
# ---------------------------------------------------------------------------

def test_a_failing_patient_leaves_the_others_registered(tmp_path, failing_greedy):
    failing_greedy("B2")
    out = run_batch(tmp_path, ["A1", "B2", "C3"])

    report = report_of(out)
    assert report["summary"] == {"cases": 3, "registered": 2, "failed": 1}
    assert report["cases"]["A1"]["status"] == "ok"
    assert report["cases"]["C3"]["status"] == "ok"


def test_a_failure_is_named_by_its_exception_and_its_message(tmp_path, failing_greedy):
    """Not "the tool failed on the server": the reason is greedy's own, and it
    is what tells a clinician whether to re-crop or to give up on the pair."""
    failing_greedy("B2")
    out = run_batch(tmp_path, ["A1", "B2"])

    entry = report_of(out)["cases"]["B2"]
    assert entry["status"] == "failed"
    assert entry["reason"] == "RuntimeError: greedy: convergence failed"


def test_a_failing_patient_still_names_the_scans_it_was_given(tmp_path, failing_greedy):
    failing_greedy("A1")
    out = run_batch(tmp_path, ["A1", "B2"])

    entry = report_of(out)["cases"]["A1"]
    assert entry["t1"] == "A1_T1.nii.gz" and entry["t2"] == "A1_T2.nii.gz"
    assert "outputs" not in entry


def test_a_case_that_times_out_is_one_failed_patient(tmp_path, failing_greedy):
    """The bound exists precisely so a pathological pair costs its own patient
    and nothing else -- if it took the batch down, the timeout would be no
    better than the hang."""
    failing_greedy("B2", raises=lambda patient: subprocess.TimeoutExpired(
        cmd=["greedy", patient], timeout=600))
    out = run_batch(tmp_path, ["A1", "B2", "C3"])

    report = report_of(out)
    assert report["summary"] == {"cases": 3, "registered": 2, "failed": 1}
    assert report["cases"]["B2"]["reason"].startswith("TimeoutExpired:")
    assert "600" in report["cases"]["B2"]["reason"]


def test_every_patient_failing_raises_rather_than_reporting_success(tmp_path, failing_greedy):
    failing_greedy("A1", "B2")
    t1, t2 = cohort(tmp_path, ["A1", "B2"])

    with pytest.raises(ValueError) as failure:
        sadt_greedyreg.run(t1=t1, t2=t2, output_dir=tmp_path / "out")

    assert "registered none" in str(failure.value)


def test_every_patient_failing_names_each_patients_reason(tmp_path, failing_greedy):
    failing_greedy("A1", "B2")
    t1, t2 = cohort(tmp_path, ["A1", "B2"])

    with pytest.raises(ValueError) as failure:
        sadt_greedyreg.run(t1=t1, t2=t2, output_dir=tmp_path / "out")

    message = str(failure.value)
    assert "A1: RuntimeError: greedy: convergence failed" in message
    assert "B2: RuntimeError" in message


def test_a_batch_that_registered_nothing_writes_no_report(tmp_path, failing_greedy):
    """An empty success is the thing being refused, and a report saying
    "0 registered" beside a 200 is what that would look like."""
    failing_greedy("A1")
    t1, t2 = cohort(tmp_path, ["A1"])

    with pytest.raises(ValueError):
        sadt_greedyreg.run(t1=t1, t2=t2, output_dir=tmp_path / "out")

    assert not (tmp_path / "out" / "GreedyReg_report.json").exists()


# ---------------------------------------------------------------------------
# Nested cohorts, and the scratch directories
# ---------------------------------------------------------------------------

def test_a_nested_patient_is_written_into_the_mirrored_subfolder(tmp_path, greedy):
    """The patient key carries the directory it was found in, and greedy does
    not create directories: this raised `FileNotFoundError` before the output
    folder was made, for every nested cohort."""
    make_scan(tmp_path / "t1" / "site_a" / "A1_T1.nii.gz")
    make_scan(tmp_path / "t2" / "site_a" / "A1_T2.nii.gz")

    out = sadt_greedyreg.run(t1=tmp_path / "t1", t2=tmp_path / "t2",
                             output_dir=tmp_path / "out")

    assert (out / "site_a" / "A1_registered.nii.gz").exists()
    assert (out / "site_a" / "A1_transform.mat").exists()


def test_a_nested_patient_lists_its_outputs_relative_to_the_output_folder(tmp_path, greedy):
    make_scan(tmp_path / "t1" / "site_a" / "A1_T1.nii.gz")
    make_scan(tmp_path / "t2" / "site_a" / "A1_T2.nii.gz")

    out = sadt_greedyreg.run(t1=tmp_path / "t1", t2=tmp_path / "t2",
                             output_dir=tmp_path / "out")

    assert report_of(out)["cases"]["site_a/A1"]["produced"] == [
        "site_a/A1_registered.nii.gz", "site_a/A1_transform.mat"]


def test_a_nested_patient_does_not_take_the_batch_down(tmp_path, greedy):
    """Its scratch directory used to be created from the patient key, outside
    the per-patient guard -- so `mkdtemp(prefix="greedyreg_site_a/A1_")` raised
    and lost every other patient in the cohort."""
    make_scan(tmp_path / "t1" / "site_a" / "A1_T1.nii.gz")
    make_scan(tmp_path / "t2" / "site_a" / "A1_T2.nii.gz")
    make_scan(tmp_path / "t1" / "B2_T1.nii.gz")
    make_scan(tmp_path / "t2" / "B2_T2.nii.gz")

    out = sadt_greedyreg.run(t1=tmp_path / "t1", t2=tmp_path / "t2",
                             output_dir=tmp_path / "out")

    assert report_of(out)["summary"] == {"cases": 2, "registered": 2, "failed": 0}


def test_no_scratch_directory_survives_a_successful_run(tmp_path, greedy):
    """The scratch holds the caller's binarised mask: confidential, and gone
    the moment the patient is done."""
    before = set(os.listdir(tempfile.gettempdir()))
    run_batch(tmp_path, ["A1", "B2"])

    left = {name for name in set(os.listdir(tempfile.gettempdir())) - before
            if name.startswith("greedyreg_")}
    assert left == set()


def test_no_scratch_directory_survives_a_failed_patient(tmp_path, failing_greedy):
    failing_greedy("A1")
    before = set(os.listdir(tempfile.gettempdir()))
    run_batch(tmp_path, ["A1", "B2"])

    left = {name for name in set(os.listdir(tempfile.gettempdir())) - before
            if name.startswith("greedyreg_")}
    assert left == set()
