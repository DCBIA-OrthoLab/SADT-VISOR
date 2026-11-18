"""The actual work, kept out of __init__.py so that run() reads as a contract.

Real tools put the ported upstream pipeline here (one module per stage if it is
large). Three things in this file are worth copying verbatim into a new tool:
`iter_scans`, which is how a batch-capable argument is expanded, the lazy
import inside `summarise`, and the way the batch loop reports a failed item.

**Write every failure for someone who sees only a few redacted lines.** When a
run fails, the job directory -- and any report in it -- is deleted. What the
server operator keeps is the exception's class, the innermost frame of this
package, the last progress message, `str(exception)` and this package's log
lines, all with paths, file names and patient-like tokens redacted and each cut
at about 200 characters. So: put the essential cause FIRST and keep it short,
say where in the batch it happened ("scan 3 of 40") instead of naming a file,
describe an expected layout in words (a path is redacted into `<path>`), and
name the argument a bad input came from.
"""

import logging
from collections import Counter
from pathlib import Path

from . import progress

# The operator sees the INFO-and-above records of loggers under this package --
# which `getLogger(__name__)` is -- and nothing else: not print(), not stderr,
# not DEBUG, not a third-party library's logger (torch, nnunetv2, ...), and
# not anything logged inside a worker PROCESS, which does not forward its
# records. Whatever the operator needs to understand a failure has to be logged
# here, from this process.
logger = logging.getLogger(__name__)

SUFFIX = ".npy"

METRICS = {
    "mean": lambda np, values: float(np.mean(values)),
    "max": lambda np, values: float(np.max(values)),
    "min": lambda np, values: float(np.min(values)),
    "std": lambda np, values: float(np.std(values)),
}

REDUCTIONS = ("per_scan", "pooled")


class ToolInputError(ValueError):
    """An input the tool cannot work with, phrased for whoever sent it.

    The runner turns this into a 422 for the client. Anything else surfaces as
    an internal error, so raise this for bad inputs and let genuine bugs crash.
    """


def iter_scans(scans: Path) -> list[Path]:
    """Expand a batch argument into the files to process, in a stable order.

    Copied into every tool rather than shared: see CONTRIBUTING.md on why there
    is no sadt-core package. Sorting matters -- readdir order varies between
    filesystems, and an unordered batch makes two runs on the same folder
    produce differently ordered reports.

    The messages name the ARGUMENT and leave the path out: the path is redacted
    before anyone reads it, so "<path> does not exist" cannot tell a caller
    which of two path arguments was wrong, and "'scans' path does not exist"
    can.
    """
    if scans.is_dir():
        found = sorted(p for p in scans.rglob(f"*{SUFFIX}") if p.is_file())
        if not found:
            raise ToolInputError(f"'scans' folder holds no {SUFFIX} file.")
        return found
    if scans.is_file():
        return [scans]
    raise ToolInputError("'scans' path does not exist.")


def summarise(
    scans: Path,
    output_dir: Path,
    metrics: list,
    reduction: str,
    threshold: float,
    per_scan_report: bool,
) -> Path:
    import numpy as np

    # `Literal` is published as `choices` so a client can render a picker, but
    # the runner still calls run(**params) from a JSON object -- a direct caller
    # or a stale client can send anything. Checked here, not assumed.
    unknown = [name for name in metrics if name not in METRICS]
    if unknown:
        raise ToolInputError(
            f"Unknown metric(s): {', '.join(unknown)}. Available: {', '.join(METRICS)}."
        )
    if not metrics:
        raise ToolInputError("Select at least one metric.")
    if reduction not in REDUCTIONS:
        raise ToolInputError(
            f"Unknown reduction '{reduction}'. Available: {', '.join(REDUCTIONS)}."
        )

    # Created here, not by the caller: a tool that assumes its output directory
    # already exists fails the first time it is run outside the server.
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    pooled = []
    failures = []
    # Copied into every tool, like `iter_scans` above and for the same reason.
    # `report` is one line appended to the file the server named, and it is the
    # difference between a clinician seeing "scan 14 of 40" and seeing nothing
    # for an hour. The counter travels, the file name never does.
    found = iter_scans(scans)
    total = len(found)
    for index, scan in enumerate(found, start=1):
        progress.report(index, total, "scan")
        # One bad item must not cost the other 39, and must not vanish either:
        # the warning carries the position, the step, the exception's class and
        # its message -- never the file name, which is redacted anyway and would
        # leave the operator with "<file> failed" and nothing else.
        try:
            values = _load(np, scan).ravel()
        except Exception as exc:  # noqa: BLE001 -- counted and re-raised below
            logger.warning(
                "scan %d of %d: reading failed (%s: %s)",
                index, total, type(exc).__name__, exc,
            )
            failures.append(exc)
            continue
        kept = values[values > threshold]
        row = {"scan": scan.name, "voxels": int(kept.size)}
        # An empty selection would make np.mean warn and return nan; report the
        # count and skip the statistics instead of writing nan into a result.
        row.update(
            {name: METRICS[name](np, kept) for name in metrics} if kept.size else {}
        )
        rows.append(row)

        if reduction == "pooled":
            pooled.append(kept)

        if per_scan_report:
            (output_dir / f"{scan.stem}.txt").write_text(_format(row), encoding="utf-8")

    # A guard named after the OUTPUT: a run that summarised nothing must fail,
    # not return an empty summary that reads as success.
    if not rows:
        raise _nothing_summarised(failures, total) from failures[0]
    _log_summary(len(rows), total)

    if reduction == "pooled":
        values = np.concatenate(pooled) if pooled else np.empty(0)
        row = {"scan": "all", "voxels": int(values.size)}
        row.update({name: METRICS[name](np, values) for name in metrics} if values.size else {})
        rows = [row]

    summary = output_dir / "summary.txt"
    summary.write_text("\n".join(_format(row) for row in rows) + "\n", encoding="utf-8")
    return summary


def _load(np, scan: Path):
    """Read one array, turning an unreadable file into the caller's fault.

    The message is the same for every scan -- the position is in the log line
    -- so that identical failures group into one "most common failure".
    """
    try:
        return np.load(scan)
    except (OSError, ValueError) as exc:
        raise ToolInputError(f"not a readable {SUFFIX} array ({exc})") from exc


def _nothing_summarised(failures: list, total: int) -> Exception:
    """The error for a batch in which no scan could be summarised.

    The essential cause comes first, then the commonest failure with its share,
    because only the first ~300 characters survive. It stays a ToolInputError
    -- a 422 the caller can act on -- only when EVERY failure was the input's
    fault; one internal failure among them makes the batch a server-side
    problem, which a 422 would wrongly blame on the caller.
    """
    counts = Counter(f"{type(exc).__name__}: {exc}" for exc in failures)
    common, count = counts.most_common(1)[0]
    message = (
        f"0 of {total} scans summarised; most common failure: "
        f"{common} ({count} of {total})"
    )
    if all(isinstance(exc, ToolInputError) for exc in failures):
        return ToolInputError(message)
    return RuntimeError(message)


def _log_summary(done: int, total: int) -> None:
    """One closing line: INFO when every scan made it, WARNING when some did not.

    The clinician hears about a partial batch too, through `progress.log(...,
    user=True)` -- in words about positions, never a file name.
    """
    failed = total - done
    if not failed:
        logger.info("%d of %d scans summarised", done, total)
        return
    logger.warning("%d of %d scans summarised, %d failed", done, total, failed)
    progress.log(
        f"{failed} of {total} scans could not be read and are missing from the summary",
        level="warning",
        user=True,
    )


def _format(row: dict) -> str:
    return " ".join(f"{key}={value}" for key, value in row.items())
