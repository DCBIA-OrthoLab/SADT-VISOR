"""GreedyReg, with greedy itself stubbed: no registration, everything around it."""

import json
import logging
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import sadt_greedyreg
from sadt_greedyreg import pipeline
from sadt_areg_common import pairing


def _scan(path):
    import nibabel as nib

    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), np.float32), np.eye(4)), str(path))
    return path


@pytest.fixture
def stubbed(monkeypatch):
    """greedy replaced by a recorder that writes the files it would write."""
    calls = []

    def fake(command):
        calls.append(command)
        # The resample invocation names its output third after -rm.
        if "-rm" in command:
            index = command.index("-rm")
            _scan(__import__("pathlib").Path(command[index + 2]))
        elif "-o" in command:
            open(command[command.index("-o") + 1], "w").write("stub transform\n")
        return ""

    monkeypatch.setattr(sadt_greedyreg, "run_greedy", fake)
    return calls


# ---------------------------------------------------------------------------
# The patient rule, which is why this port exists
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename", [
    # Every one of these is a name this repository's own data uses, and every
    # one was dropped in silence by upstream's `^([A-Za-z]+\d+)`: it wants
    # letters then digits with nothing between, and a file that matched
    # nothing never entered the dict.
    "C_0001_T1_Or.nii.gz",
    "MAMP_0002_T1.nii.gz",
    "IC_0005.nii.gz",
    "0001_T1.nii.gz",
])
def test_the_names_this_repository_ships_are_paired(filename):
    assert pairing.patient_stem(filename), f"{filename} yields no patient"


def test_two_scans_of_one_patient_do_not_overwrite_each_other(tmp_path):
    """Upstream keyed a dict on the ID prefix, so `A1_scan` and `A1_other`
    collided and one was lost with no message."""
    _scan(tmp_path / "t1" / "A1_T1.nii.gz")
    _scan(tmp_path / "t2" / "A1_T2.nii.gz")

    matched = pairing.pair(str(tmp_path / "t1"), str(tmp_path / "t2"), "registered")

    assert len(matched) == 1
    assert matched.unmatched_report() == {"t1_without_t2": [], "t2_without_t1": []}


# ---------------------------------------------------------------------------
# The commands, which must stay upstream's verbatim
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("metric,expected", [
    ("NCC", ["-m", "NCC", "4x4x4"]),
    ("NMI", ["-m", "NMI"]),
    ("SSD", ["-m", "SSD"]),
])
def test_the_metric_flag_is_upstreams(metric, expected):
    assert pipeline.metric_arguments(metric) == expected


@pytest.mark.parametrize("transform,dof", [("Rigid", "6"), ("Affine", "12")])
def test_the_degrees_of_freedom_are_upstreams(transform, dof):
    command = pipeline.registration_command(
        "f.nii.gz", "m.nii.gz", "o.mat", "i.mat", "NCC", transform)
    assert command[command.index("-dof") + 1] == dof
    assert "-n" in command and command[command.index("-n") + 1] == "100x100x50x25"
    assert command[command.index("-search") + 1:command.index("-search") + 4] == ["100", "10", "20"]


def test_a_mask_adds_gm_and_nothing_else():
    without = pipeline.registration_command("f", "m", "o", "i", "NCC", "Rigid")
    with_mask = pipeline.registration_command("f", "m", "o", "i", "NCC", "Rigid", "mask.nii.gz")
    assert "-gm" not in without
    assert with_mask[:len(without)] == without
    assert with_mask[len(without):] == ["-gm", "mask.nii.gz"]


def test_the_identity_init_is_nudged(tmp_path):
    """Greedy reads an exact identity as "nothing given" and substitutes its
    own guess, so the 1-micron nudge is what makes "start where you are"
    expressible. Upstream's comment, upstream's value."""
    path = tmp_path / "init.mat"
    pipeline.write_identity_init(str(path))

    matrix = np.array([[float(v) for v in line.split()]
                       for line in path.read_text().splitlines()])
    assert matrix.shape == (4, 4)
    assert matrix[0, 3] == 0.001
    assert not np.array_equal(matrix, np.eye(4))


# ---------------------------------------------------------------------------
# The batch
# ---------------------------------------------------------------------------

def test_a_batch_registers_every_pair(tmp_path, stubbed):
    for patient in ("A1", "B2"):
        _scan(tmp_path / "t1" / f"{patient}_T1.nii.gz")
        _scan(tmp_path / "t2" / f"{patient}_T2.nii.gz")

    out = sadt_greedyreg.run(t1=tmp_path / "t1", t2=tmp_path / "t2",
                             output_dir=tmp_path / "out")

    report = json.loads((out / "GreedyReg_report.json").read_text())
    assert report["summary"] == {"cases": 2, "registered": 2, "failed": 0}
    assert (out / "A1_registered.nii.gz").exists()
    assert (out / "A1_transform.mat").exists()


def test_one_patient_failing_does_not_cost_the_others(tmp_path, stubbed, monkeypatch):
    """Upstream called sys.exit(1) here, so patient 3 lost patients 4 to 40 --
    and the batch reported nothing about any of them."""
    for patient in ("A1", "B2"):
        _scan(tmp_path / "t1" / f"{patient}_T1.nii.gz")
        _scan(tmp_path / "t2" / f"{patient}_T2.nii.gz")

    def sometimes(command):
        if any("B2" in str(part) for part in command):
            raise RuntimeError("greedy: convergence failed")
        return stubbed and _stub_write(command)

    def _stub_write(command):
        if "-rm" in command:
            _scan(__import__("pathlib").Path(command[command.index("-rm") + 2]))
        elif "-o" in command:
            open(command[command.index("-o") + 1], "w").write("stub\n")
        return ""

    monkeypatch.setattr(sadt_greedyreg, "run_greedy", sometimes)

    out = sadt_greedyreg.run(t1=tmp_path / "t1", t2=tmp_path / "t2",
                             output_dir=tmp_path / "out")

    report = json.loads((out / "GreedyReg_report.json").read_text())
    assert report["summary"] == {"cases": 2, "registered": 1, "failed": 1}
    assert report["cases"]["B2"]["status"] == "failed"
    assert "convergence" in report["cases"]["B2"]["reason"]
    assert report["cases"]["A1"]["status"] == "ok"


def test_no_pair_at_all_is_refused(tmp_path, stubbed):
    _scan(tmp_path / "t1" / "A1_T1.nii.gz")
    _scan(tmp_path / "t2" / "Z9_T2.nii.gz")

    with pytest.raises(ValueError, match="No patient appears in both"):
        sadt_greedyreg.run(t1=tmp_path / "t1", t2=tmp_path / "t2",
                           output_dir=tmp_path / "out")


# ---------------------------------------------------------------------------
# The bound the port had to give back
# ---------------------------------------------------------------------------

def test_greedy_runs_in_a_child_process_with_a_timeout(monkeypatch):
    """Upstream bounded each call with `subprocess.run(..., timeout=600)`.
    Moving to an in-process `Greedy3D.execute()` would have dropped that
    silently, and the server's TOOL_TIMEOUT_SECONDS defaults to 0 -- "none",
    because a cohort legitimately takes hours. One pathological pair would
    then hold a concurrency slot for ever. The package replaces the binary,
    not the boundary."""
    import subprocess

    seen = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        seen["timeout"] = kwargs.get("timeout")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert pipeline.run_greedy(["-d", "3", "-a"]) == "ok"
    assert seen["timeout"] == pipeline.CASE_TIMEOUT_SECONDS == 600
    assert seen["command"][0].endswith("python") or "python" in seen["command"][0]
    assert seen["command"][1] == "-c"
    assert seen["command"][-3:] == ["-d", "3", "-a"]


def test_a_failing_greedy_carries_its_own_message(monkeypatch):
    """Nothing this tool knows explains why a registration did not converge,
    so greedy's message is what travels."""
    import subprocess

    monkeypatch.setattr(
        subprocess, "run",
        lambda command, **kw: subprocess.CompletedProcess(
            command, 1, stdout="", stderr="greedy: images do not overlap"),
    )

    with pytest.raises(RuntimeError, match="images do not overlap"):
        pipeline.run_greedy(["-d", "3"])


def test_a_batch_says_which_pair_it_is_on(tmp_path, stubbed, monkeypatch):
    """One event per patient, rising, and the position rather than the key.

    The patient key is derived from the caller's file names, so it is patient
    metadata; a progress message is stored on the server and shown.
    """
    events_file = tmp_path / "events.jsonl"
    monkeypatch.setenv("SADT_PROGRESS_FILE", str(events_file))
    for patient in ("A1", "B2"):
        _scan(tmp_path / "t1" / f"{patient}_T1.nii.gz")
        _scan(tmp_path / "t2" / f"{patient}_T2.nii.gz")

    sadt_greedyreg.run(t1=tmp_path / "t1", t2=tmp_path / "t2",
                       output_dir=tmp_path / "out")

    events = [json.loads(line) for line in events_file.read_text().splitlines() if line]
    assert [e["message"] for e in events] == ["patient 1 of 2", "patient 2 of 2"]
    assert [e["fraction"] for e in events] == [0.0, 0.5]
    assert not any("A1" in e["message"] for e in events)


def test_a_failure_names_the_position_and_never_the_patient(
    tmp_path, stubbed, monkeypatch, caplog
):
    """The counter in the log too, and here it matters more than in the bar.

    A tool's stderr is captured to a file in the job directory, and on a FAILED
    run the server copies its tail into its own persistent log -- so a patient
    key written on this path outlives the run and its job directory. The run
    report keeps naming the pair; that goes back to whoever sent the data.

    Asserted on the composed message: greedy's own stderr travels in the
    exception, and what a third-party program prints is a separate exposure.
    """
    for patient in ("MAMP_0001", "MAMP_0002"):
        _scan(tmp_path / "t1" / f"{patient}_T1.nii.gz")
        _scan(tmp_path / "t2" / f"{patient}_T2.nii.gz")

    stubbed_greedy = sadt_greedyreg.run_greedy

    def fail_for_the_first_patient(command):
        if any("MAMP_0001" in str(part) for part in command):
            raise RuntimeError("greedy: images do not overlap")
        return stubbed_greedy(command)

    monkeypatch.setattr(sadt_greedyreg, "run_greedy", fail_for_the_first_patient)

    with caplog.at_level(logging.INFO, logger="GreedyReg"):
        sadt_greedyreg.run(t1=tmp_path / "t1", t2=tmp_path / "t2",
                           output_dir=tmp_path / "out")

    messages = [record.getMessage() for record in caplog.records]
    assert any(m.startswith("GreedyReg failed on patient ") and m.endswith(" of 2")
               for m in messages), messages
    assert not any("MAMP_0001" in m for m in messages), messages

    report = json.loads((tmp_path / "out" / "GreedyReg_report.json").read_text())
    assert report["cases"]["MAMP_0001"]["status"] == "failed"
