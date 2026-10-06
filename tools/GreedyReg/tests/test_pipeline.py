"""The three things this tool computes itself, and the boundary it runs greedy behind.

Nothing here loads `picsl_greedy`: the child process is real, but the script it
runs is replaced, so a full run of this file costs a few subprocess launches and
no registration.
"""

import json
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from sadt_greedyreg import pipeline

# A child that answers with its own arguments, so what crossed the process
# boundary can be read back exactly as it arrived.
ECHO_ARGV = "import sys, json; print(json.dumps(sys.argv[1:]), end='')"


def read_matrix(path):
    return np.array([[float(value) for value in line.split()]
                     for line in Path(path).read_text().splitlines()])


def write_mask(path, data, affine=None):
    path = Path(path)
    image = nib.Nifti1Image(np.asarray(data), np.eye(4) if affine is None else affine)
    nib.save(image, str(path))
    return path


# ---------------------------------------------------------------------------
# write_identity_init
# ---------------------------------------------------------------------------

def test_the_identity_init_parses_back_as_a_four_by_four(tmp_path):
    path = tmp_path / "init.mat"
    pipeline.write_identity_init(str(path))

    assert len(path.read_text().splitlines()) == 4
    assert read_matrix(path).shape == (4, 4)


def test_the_nudge_is_one_micron_and_it_is_the_x_translation(tmp_path):
    """The right cell, not merely a non-identity somewhere: a nudge in a
    rotation cell would be a rotation, and greedy would start from a frame
    nobody asked for."""
    path = tmp_path / "init.mat"
    pipeline.write_identity_init(str(path))

    matrix = read_matrix(path)
    assert matrix[0, 3] == 0.001


def test_the_init_is_an_identity_everywhere_else(tmp_path):
    path = tmp_path / "init.mat"
    pipeline.write_identity_init(str(path))

    matrix = read_matrix(path)
    matrix[0, 3] = 0.0
    assert np.array_equal(matrix, np.eye(4))


def test_the_init_is_not_an_exact_identity(tmp_path):
    path = tmp_path / "init.mat"
    pipeline.write_identity_init(str(path))

    assert not np.array_equal(read_matrix(path), np.eye(4))


def test_the_reason_for_the_nudge_is_written_down():
    """Upstream's comment, kept: greedy reads an exact identity as "no
    initialisation given" and substitutes its own guess, so the nudge is what
    makes "start from where the images already are" expressible. Without the
    comment the next reader deletes the 0.001 as a typo."""
    documentation = pipeline.write_identity_init.__doc__
    assert "no initialisation given" in documentation
    assert "identity" in documentation


def test_the_init_is_rewritten_rather_than_appended_to(tmp_path):
    """Two patients in one batch reuse nothing, but a scratch path could be
    reused by a caller; four lines in, four lines out."""
    path = tmp_path / "init.mat"
    pipeline.write_identity_init(str(path))
    pipeline.write_identity_init(str(path))

    assert len(path.read_text().splitlines()) == 4


# ---------------------------------------------------------------------------
# binarise_mask
# ---------------------------------------------------------------------------

def test_an_all_zero_mask_stays_all_zero(tmp_path):
    write_mask(tmp_path / "in.nii.gz", np.zeros((3, 3, 3), np.float32))

    pipeline.binarise_mask(str(tmp_path / "in.nii.gz"), str(tmp_path / "out.nii.gz"))

    assert np.array_equal(nib.load(str(tmp_path / "out.nii.gz")).get_fdata(), np.zeros((3, 3, 3)))


def test_an_all_one_mask_stays_all_one(tmp_path):
    write_mask(tmp_path / "in.nii.gz", np.ones((3, 3, 3), np.float32))

    pipeline.binarise_mask(str(tmp_path / "in.nii.gz"), str(tmp_path / "out.nii.gz"))

    assert np.array_equal(nib.load(str(tmp_path / "out.nii.gz")).get_fdata(), np.ones((3, 3, 3)))


def test_a_multi_label_mask_becomes_one_region(tmp_path):
    """AMASSS writes several structures into one volume as 1, 2, 3... Greedy's
    `-gm` wants "inside or outside", so every label is inside."""
    data = np.array([[[0, 1, 2, 3, 7]]], dtype=np.int16)
    write_mask(tmp_path / "in.nii.gz", data)

    pipeline.binarise_mask(str(tmp_path / "in.nii.gz"), str(tmp_path / "out.nii.gz"))

    written = nib.load(str(tmp_path / "out.nii.gz")).get_fdata()
    assert sorted(np.unique(written)) == [0.0, 1.0]
    assert np.array_equal(written.ravel(), [0, 1, 1, 1, 1])


def test_a_float_mask_is_thresholded_strictly_above_zero(tmp_path):
    data = np.array([[[-1.0, 0.0, 0.4, 0.5, 1.0]]], dtype=np.float32)
    write_mask(tmp_path / "in.nii.gz", data)

    pipeline.binarise_mask(str(tmp_path / "in.nii.gz"), str(tmp_path / "out.nii.gz"))

    written = nib.load(str(tmp_path / "out.nii.gz")).get_fdata()
    assert np.array_equal(written.ravel(), [0, 0, 1, 1, 1])


def test_a_not_a_number_voxel_falls_outside_the_mask(tmp_path):
    """`nan > 0` is False, so an undefined voxel is outside rather than a
    value greedy would have to interpret."""
    data = np.array([[[np.nan, 1.0]]], dtype=np.float32)
    write_mask(tmp_path / "in.nii.gz", data)

    pipeline.binarise_mask(str(tmp_path / "in.nii.gz"), str(tmp_path / "out.nii.gz"))

    assert np.array_equal(
        nib.load(str(tmp_path / "out.nii.gz")).get_fdata().ravel(), [0, 1])


def test_the_binarised_mask_is_written_as_float(tmp_path):
    """Greedy's `-gm` wants a float mask; an integer one is read as labels."""
    write_mask(tmp_path / "in.nii.gz", np.ones((2, 2, 2), np.int16))

    pipeline.binarise_mask(str(tmp_path / "in.nii.gz"), str(tmp_path / "out.nii.gz"))

    assert nib.load(str(tmp_path / "out.nii.gz")).get_data_dtype() == np.float32


def test_the_mask_keeps_its_own_geometry(tmp_path):
    """A mask whose grid differs from the scan is NOT resampled here: greedy
    reads the affine, and silently regridding it would move the region the
    metric is computed in."""
    affine = np.diag([0.5, 0.4, 0.3, 1.0])
    write_mask(tmp_path / "in.nii.gz", np.ones((3, 5, 7), np.float32), affine)

    pipeline.binarise_mask(str(tmp_path / "in.nii.gz"), str(tmp_path / "out.nii.gz"))

    written = nib.load(str(tmp_path / "out.nii.gz"))
    assert written.shape == (3, 5, 7)
    assert np.allclose(written.affine, affine)


def test_binarising_leaves_the_source_untouched(tmp_path):
    """The mask belongs to the caller; the binarised copy goes to scratch."""
    source = write_mask(tmp_path / "in.nii.gz", np.array([[[0, 2, 5]]], dtype=np.int16))
    before = source.read_bytes()

    pipeline.binarise_mask(str(source), str(tmp_path / "out.nii.gz"))

    assert source.read_bytes() == before


# ---------------------------------------------------------------------------
# run_greedy: the process boundary
# ---------------------------------------------------------------------------

def test_a_child_that_succeeds_returns_what_it_printed(monkeypatch):
    monkeypatch.setattr(pipeline, "_CHILD", "print('registration done', end='')")

    assert pipeline.run_greedy(["-d", "3"]) == "registration done"


def test_the_arguments_reach_the_child_as_strings(monkeypatch):
    """A caller passes Paths and numbers; `Greedy3D.execute` joins argv into
    one string, so anything not already a str has to be made one here."""
    monkeypatch.setattr(pipeline, "_CHILD", ECHO_ARGV)

    answered = json.loads(pipeline.run_greedy(["-d", 3, Path("/scans/A1_T1.nii.gz")]))

    assert answered == ["-d", "3", "/scans/A1_T1.nii.gz"]


def test_the_whole_command_reaches_the_child_in_order(monkeypatch):
    monkeypatch.setattr(pipeline, "_CHILD", ECHO_ARGV)
    command = pipeline.registration_command("f", "m", "o.mat", "i.mat", "NCC", "Affine")

    assert json.loads(pipeline.run_greedy(command)) == command


def test_a_child_that_exits_non_zero_carries_its_own_message(monkeypatch):
    """Nothing this tool knows explains why a registration did not converge,
    so greedy's own message is the one that travels."""
    monkeypatch.setattr(
        pipeline, "_CHILD",
        "import sys; sys.stderr.write('greedy: images do not overlap\\n'); sys.exit(1)")

    with pytest.raises(RuntimeError, match="images do not overlap"):
        pipeline.run_greedy(["-d", "3"])


def test_a_silent_failure_falls_back_to_what_was_printed(monkeypatch):
    """greedy prints its progress to stdout and dies without a word on
    stderr often enough that an empty error message would be the common case."""
    monkeypatch.setattr(
        pipeline, "_CHILD",
        "import sys; print('Level 0: iter 3'); sys.exit(2)")

    with pytest.raises(RuntimeError, match="Level 0"):
        pipeline.run_greedy(["-d", "3"])


def test_a_failure_with_nothing_printed_at_all_still_raises(monkeypatch):
    monkeypatch.setattr(pipeline, "_CHILD", "import sys; sys.exit(3)")

    with pytest.raises(RuntimeError, match="exit code 3, nothing printed"):
        pipeline.run_greedy(["-d", "3"])


def test_a_case_that_runs_too_long_is_abandoned(monkeypatch):
    """Upstream bounded each call with `subprocess.run(..., timeout=600)`.
    Moving to an in-process `Greedy3D.execute()` would have dropped that in
    silence, and the server's TOOL_TIMEOUT_SECONDS defaults to 0 -- none,
    because a cohort legitimately takes hours -- so one pathological pair would
    have held a concurrency slot for ever."""
    monkeypatch.setattr(pipeline, "_CHILD", "import time; time.sleep(30)")

    with pytest.raises(TimeoutError):
        pipeline.run_greedy(["-d", "3"], timeout=0.5)


def test_the_timeout_is_reported_as_a_timeout(monkeypatch):
    """A timeout and a failure call for different things -- a bigger bound, or
    different images -- so they must not arrive as the same exception."""
    monkeypatch.setattr(pipeline, "_CHILD", "import time; time.sleep(30)")

    with pytest.raises(TimeoutError) as expired:
        pipeline.run_greedy(["-d", "3"], timeout=0.5)

    assert not isinstance(expired.value, RuntimeError)
    assert str(expired.value) == "greedy call did not finish within 0.5s"


def test_the_default_bound_is_upstreams_ten_minutes(monkeypatch):
    seen = {}

    def fake_run(command, **kwargs):
        seen.update(command=command, timeout=kwargs.get("timeout"))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    pipeline.run_greedy(["-d", "3"])

    assert pipeline.CASE_TIMEOUT_SECONDS == 600
    assert seen["timeout"] == 600


def test_a_missing_interpreter_is_not_swallowed(monkeypatch):
    """The tool's own venv is what runs the child. If it is gone, that has to
    surface as an error naming the path, not as a run that quietly did nothing."""
    monkeypatch.setattr(sys, "executable", "/nonexistent/python")

    with pytest.raises(FileNotFoundError) as missing:
        pipeline.run_greedy(["-d", "3"])

    assert "/nonexistent/python" in str(missing.value)


def test_greedy_is_run_as_a_child_of_this_interpreter(monkeypatch):
    monkeypatch.setattr(pipeline, "_CHILD", "import sys; print(sys.executable, end='')")

    assert pipeline.run_greedy([]) == sys.executable


def test_the_child_script_is_valid_python_and_calls_greedy():
    """It is a string, so nothing else in this suite would catch a syntax
    error in it -- and only a real registration would."""
    compile(pipeline._CHILD, "<child>", "exec")
    assert "from picsl_greedy import Greedy3D" in pipeline._CHILD
    assert "Greedy3D().execute(" in pipeline._CHILD


def test_the_child_hands_greedy_one_joined_argument_string():
    """`Greedy3D.execute` takes the command line as a single string, which is
    what lets upstream's argument list survive the port unchanged."""
    assert '" ".join(sys.argv[1:])' in pipeline._CHILD


def test_the_output_of_greedy_is_captured_rather_than_inherited(monkeypatch):
    """greedy prints for minutes; the server must not have that on its own
    stdout, and the caller needs it to explain a failure."""
    seen = {}

    def fake_run(command, **kwargs):
        seen.update(kwargs)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    pipeline.run_greedy(["-d", "3"])

    assert seen["capture_output"] is True
    assert seen["text"] is True


# ---------------------------------------------------------------------------
# What a failing greedy says
# ---------------------------------------------------------------------------

def test_a_failure_carries_only_the_last_line_of_a_traceback(monkeypatch):
    """The child is a Python interpreter, so greedy throwing leaves a whole
    traceback on stderr. Only its last line says why, and it is put after the
    step so a cut message still keeps the cause."""
    traceback = (
        "Traceback (most recent call last):\n"
        '  File "<string>", line 4, in <module>\n'
        "RuntimeError: Image dimensions do not match\n\n"
    )
    monkeypatch.setattr(
        subprocess, "run",
        lambda command, **kw: subprocess.CompletedProcess(command, 1, stdout="", stderr=traceback),
    )
    command = pipeline.registration_command("f", "m", "o.mat", "i.mat", "NCC", "Rigid")

    with pytest.raises(RuntimeError) as failure:
        pipeline.run_greedy(command)

    assert str(failure.value) == (
        "greedy registration (NCC, 6 dof) failed: RuntimeError: Image dimensions do not match")


def test_a_failing_resample_says_it_was_the_resample(monkeypatch):
    monkeypatch.setattr(
        subprocess, "run",
        lambda command, **kw: subprocess.CompletedProcess(command, 1, stdout="", stderr="boom\n"),
    )

    with pytest.raises(RuntimeError, match="^greedy resample failed: boom$"):
        pipeline.run_greedy(pipeline.resample_command("f", "m", "o", "t"))


def test_a_timeout_names_the_step_and_the_bound_and_no_path(monkeypatch):
    """TimeoutExpired's own message is the whole command line -- the child
    script and every scan path -- so it is replaced, not chained."""
    def expire(command, **kwargs):
        raise subprocess.TimeoutExpired(cmd=command, timeout=kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", expire)
    command = pipeline.registration_command(
        "/scans/A1_T1.nii.gz", "/scans/A1_T2.nii.gz", "o.mat", "i.mat", "NCC", "Rigid")

    with pytest.raises(TimeoutError) as expired:
        pipeline.run_greedy(command)

    assert str(expired.value) == "greedy registration did not finish within 600s"
    assert expired.value.__cause__ is None and expired.value.__suppress_context__


@pytest.mark.parametrize("command,expected", [
    (pipeline.registration_command("f", "m", "o", "i", "NMI", "Affine"),
     "registration (NMI, 12 dof)"),
    (pipeline.registration_command("f", "m", "o", "i", "NCC", "Rigid", "mask.nii.gz"),
     "registration (NCC, 6 dof, masked)"),
    (pipeline.resample_command("f", "m", "o", "t"), "resample"),
])
def test_a_command_is_described_in_words(command, expected):
    assert pipeline.describe(command) == expected


def test_the_final_metric_is_the_last_levels():
    printed = (
        "Level   0  LastIter   Metrics  -8762.396413  Energy = -8762.396413\n"
        "Level   0  Final RAS Transform:\n"
        "Level   1  LastIter   Metrics  -7929.584809  Energy = -7929.584809\n"
    )
    assert pipeline.final_metric(printed) == pytest.approx(-7929.584809)
    assert pipeline.final_metric("nothing of the kind") is None
    assert pipeline.final_metric("") is None
