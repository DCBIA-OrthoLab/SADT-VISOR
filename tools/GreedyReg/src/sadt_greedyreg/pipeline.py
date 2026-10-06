"""Greedy affine registration of two CBCT timepoints.

Ported from `GreedyReg_CLI/GreedyReg_CLI.py`. For each patient present at both
timepoints, T2 is registered onto T1 and resampled into its frame; the
transform is written beside the resampled volume.

Greedy is called through `picsl_greedy`, which is greedy's own Python
distribution: `Greedy3D.execute(command)` takes the same argument string the
executable does, so the two invocations below are the upstream ones verbatim.
Upstream shelled out to a binary whose path was an ARGUMENT, which a client
cannot supply on a server.
"""

import logging

logger = logging.getLogger("GreedyReg")

# The two choices `run()` publishes as `Literal`s, restated here because a
# `Literal` is PUBLISHED, not enforced: the runner calls `run(**params)` and a
# direct API call can pass anything. `check_choices` is what refuses it, and it
# needs the sets written down.
METRICS = ("NCC", "NMI", "SSD")
TRANSFORMS = ("Rigid", "Affine")

# Degrees of freedom per transform type, as upstream chose them.
DEGREES_OF_FREEDOM = {"Rigid": "6", "Affine": "12"}


def _check_metric(metric: str) -> None:
    if metric not in METRICS:
        raise ValueError(
            f"Unknown metric {metric!r}. GreedyReg optimises one of: "
            f"{', '.join(METRICS)}."
        )


def _check_transform(transform_type: str) -> None:
    if transform_type not in TRANSFORMS:
        # Named, rather than the bare `KeyError: 'rigid'` the degrees-of-freedom
        # lookup used to raise: this message is what a 422 carries back.
        raise ValueError(
            f"Unknown transform_type {transform_type!r}. GreedyReg registers "
            f"with one of: {', '.join(TRANSFORMS)}."
        )


def check_choices(metric: str, transform_type: str) -> None:
    """Refuse an unrecognised metric or transform type, naming the allowed ones.

    Called once before a batch as well as inside the command builders: a typo
    is one error before anything runs, not the same error reported forty times
    as forty failed patients.
    """
    _check_metric(metric)
    _check_transform(transform_type)


def metric_arguments(metric: str) -> list:
    """Greedy's `-m` flag. NCC carries a radius, the other two do not.

    An unrecognised metric is REFUSED rather than defaulted. The first version
    fell through to SSD, so `"ncc"` -- the same word in the wrong case, which is
    what a `sup` call or a direct API call can send -- silently changed what was
    optimised and the report said NCC anyway.
    """
    _check_metric(metric)
    if metric == "NCC":
        return ["-m", "NCC", "4x4x4"]
    if metric == "NMI":
        return ["-m", "NMI"]
    return ["-m", "SSD"]


def registration_command(fixed: str, moving: str, transform_out: str, init: str,
                         metric: str, transform_type: str, mask: str = "") -> list:
    """The affine search, verbatim from upstream's `buildRegistrationCommand`.

    `-n 100x100x50x25` is the multi-resolution schedule, `-search 100 10 20`
    the random search that precedes it. They are not exposed: they describe how
    this registration was tuned, not a per-request choice.
    """
    _check_transform(transform_type)
    command = ["-d", "3", "-a"]
    command += metric_arguments(metric)
    command += ["-i", fixed, moving]
    command += ["-o", transform_out]
    command += ["-n", "100x100x50x25"]
    command += ["-e", "0.5"]
    command += ["-search", "100", "10", "20"]
    command += ["-dof", DEGREES_OF_FREEDOM[transform_type]]
    command += ["-ia", init]
    if mask:
        command += ["-gm", mask]
    return command


def resample_command(fixed: str, moving: str, out: str, transform: str) -> list:
    return ["-d", "3", "-rf", fixed, "-rm", moving, out, "-r", transform]


def write_identity_init(path: str) -> None:
    """A 4x4 identity with the x translation nudged by 1 micron.

    Verbatim from upstream, comment included: Greedy treats an exact identity
    as "no initialisation given" and falls back to its own guess, so the nudge
    is what makes "start from where the images already are" expressible.
    """
    import numpy as np

    matrix = np.eye(4)
    matrix[0, 3] = 0.001
    with open(path, "w") as handle:
        for row in matrix:
            handle.write(" ".join(str(value) for value in row) + "\n")


def binarise_mask(source: str, destination: str) -> None:
    """Anything above zero becomes 1.0. Greedy's `-gm` wants a float mask."""
    import nibabel as nib
    import numpy as np

    mask = nib.load(source)
    data = (mask.get_fdata() > 0).astype(np.float32)
    written = nib.Nifti1Image(data, mask.affine)
    written.header.set_data_dtype(np.float32)
    nib.save(written, destination)


# How long one registration may run before it is abandoned. Upstream bounded
# each greedy call with `subprocess.run(..., timeout=600)`; dropping to an
# in-process `Greedy3D.execute()` would have removed that bound silently, and
# the server's own TOOL_TIMEOUT_SECONDS defaults to 0 -- "none", because a
# cohort legitimately takes hours. So one pathological pair could hold a
# concurrency slot for ever. Greedy is therefore still run as a SUBPROCESS,
# with its own interpreter: the package replaces the binary, not the boundary.
CASE_TIMEOUT_SECONDS = 600

_CHILD = """
import sys
from picsl_greedy import Greedy3D
Greedy3D().execute(" ".join(sys.argv[1:]), out=sys.stdout)
"""


def describe(command: list) -> str:
    """What one greedy invocation is, in words: the step, and for the affine
    search the metric and the degrees of freedom.

    Read back from the command rather than passed alongside it, so the
    description cannot disagree with what was actually run.
    """
    command = [str(part) for part in command]
    if "-a" in command:
        details = []
        if "-m" in command:
            details.append(command[command.index("-m") + 1])
        if "-dof" in command:
            details.append(f"{command[command.index('-dof') + 1]} dof")
        if "-gm" in command:
            details.append("masked")
        return f"registration ({', '.join(details)})" if details else "registration"
    if "-rf" in command:
        return "resample"
    return "call"


def last_line(text: str) -> str:
    """The last non-empty line of a child's output.

    The child is a Python interpreter, so when greedy throws, its stderr is a
    whole traceback whose LAST line is the one that says what went wrong; the
    frames above it are boilerplate, and a message cut at ~300 characters
    would keep the boilerplate and lose the cause.
    """
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    return lines[-1] if lines else ""


def final_metric(stdout: str):
    """The metric value of greedy's last resolution level, or None.

    greedy's affine search prints `Level N  LastIter   Metrics  <value>  Energy
    = <value>` once per resolution level; the last one is the metric the
    transform was accepted at. Read by splitting words rather than with a
    regular expression, which this tool keeps for nothing but patient names.
    """
    value = None
    for line in (stdout or "").splitlines():
        words = line.split()
        if "LastIter" in words and "Metrics" in words[:-1]:
            try:
                value = float(words[words.index("Metrics") + 1])
            except ValueError:
                continue
    return value


def run_greedy(command: list, timeout: float = CASE_TIMEOUT_SECONDS) -> str:
    """One greedy invocation, in a child process. Returns whatever it printed.

    A non-zero exit carries greedy's own message, which is the one a caller
    needs: nothing this tool knows explains why a registration did not
    converge. Only its last line travels, prefixed with the step. A timeout is
    raised as a `TimeoutError` rather than as a generic failure, because the
    two call for different things -- a bigger bound, or different images.
    """
    import subprocess
    import sys

    step = describe(command)
    try:
        finished = subprocess.run(
            [sys.executable, "-c", _CHILD, *[str(part) for part in command]],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        # `from None`: TimeoutExpired's own message is the whole command line,
        # child script and scan paths included, and a failed run's stderr is
        # copied into the server's persistent log.
        raise TimeoutError(
            f"greedy {step.split(' (')[0]} did not finish within {timeout:g}s"
        ) from None
    if finished.returncode != 0:
        cause = (last_line(finished.stderr) or last_line(finished.stdout)
                 or f"exit code {finished.returncode}, nothing printed")
        raise RuntimeError(f"greedy {step} failed: {cause}")
    return finished.stdout
