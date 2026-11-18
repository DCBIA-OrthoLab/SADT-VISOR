"""Reference tool package: copy this directory to start a new tool.

The whole public surface of a tool is the single callable `run` below. Nothing
else here is part of the contract -- no base class to subclass, no registry to
edit, no import from the server. The server's runner imports this module and
calls `run(**params)`; `scripts/describe.py` reads the same signature to publish
the tool's JSON schema. Signature and schema therefore cannot drift.

What this placeholder computes (intensity statistics over `.npy` arrays) is not
the point and is meant to be deleted. The rules it demonstrates are:

1. Stdlib annotations only -- `Path`, `str`, `int`, `float`, `bool`,
   `Literal[...]` and `list[...]` of those. describe.py refuses anything else
   rather than guess, so an `Optional[int]` or a custom marker type fails the
   build instead of shipping a wrong schema to the client.
2. No default means required. That is the *only* place `required` comes from.
3. Batch input. `scans` takes a directory as readily as one file: the server
   pays a process start-up cost per call, so a folder of 40 scans must be one
   call, not 40.
4. Heavy imports live INSIDE run(). describe.py imports this module in every CI
   job; a module-level `import torch` would make schema generation cost a CUDA
   stack and fail on a machine that has none.
5. Everything written goes under `output_dir`, which the caller owns.
6. An argument with a fixed set of options says so with `Literal[...]`, and
   describe.py publishes them as `choices` for the client to render as a
   picker. `list[Literal[...]]` is several-of, a bare `Literal[...]` is
   exactly-one -- the old schema's "multichoice" and "choice". Saying it in the
   annotation is the point: a separate table of options drifts from the code,
   which is what the `ArgSpec.choices` it replaces used to do.
7. Nothing here unpacks a `.zip`. The server unpacks archives before `run()`
   is called, so a tool always receives a real file or directory.
8. A batch says where it is. `progress.py` appends one line per item to the
   file the server names in `SADT_PROGRESS_FILE`, so a clinician watching a
   forty-scan run sees "scan 14 of 40" instead of a spinner -- see
   `pipeline.summarise`. Stdlib, best effort, and the message carries the
   POSITION in the batch and never a file name: it is stored on the server and
   shown, and a file name is patient metadata.
9. A failure explains itself to someone who sees only a few redacted lines.
   The job directory is deleted when a run fails, so what is left is the
   exception and this package's log records -- see the top of `pipeline.py`
   for what reaches the operator and `pipeline.summarise` for the per-item
   warning, the guard against a batch that produced nothing, and the closing
   summary line.
"""

from pathlib import Path
from typing import Literal

from .pipeline import summarise


def run(
    scans: Path,
    output_dir: Path,
    metrics: list[Literal["mean", "max", "min", "std"]] = ["mean", "max"],
    reduction: Literal["per_scan", "pooled"] = "per_scan",
    threshold: float = 0.0,
    per_scan_report: bool = False,
) -> Path:
    """Summarise the intensity distribution of a scan or a folder of scans.

    This first line is what the client shows a clinician in the tool list, so it
    describes the clinical action, not the implementation. Everything below it
    is for whoever maintains the tool and never reaches the UI.

    Args:
        scans: One `.npy` array, or a folder of them for a batch.
        output_dir: Where results are written. Nothing is written outside it.
        metrics: Statistics to compute. Several may be chosen.
        reduction: Whether to report one row per scan or one row for the batch.
        threshold: Ignore voxels at or below this intensity.
        per_scan_report: Also write one `<scan>.txt` next to the summary.

    Returns:
        The path of the written summary file.
    """
    # The work lives in pipeline.py, which imports numpy inside summarise() --
    # rule 4 covers the whole import chain, not just this file, so `from
    # .pipeline import summarise` at the top must stay free of heavy modules.
    return summarise(
        scans=Path(scans),
        output_dir=Path(output_dir),
        metrics=metrics,
        reduction=reduction,
        threshold=threshold,
        per_scan_report=per_scan_report,
    )
