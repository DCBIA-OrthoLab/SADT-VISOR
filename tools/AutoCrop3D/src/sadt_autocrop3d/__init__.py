"""Crop CBCT volumes and their segmentations to a Slicer ROI box."""

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Literal

from . import progress
from .pipeline import (
    SCAN_EXTENSIONS,
    UNREADABLE_EXTENSIONS,
    crop,
    crop_bounds,
    is_roi_file,
    is_scan_file,
    is_segmentation_name,
    output_name,
    patient_key,
    read_roi,
    repad,
    surface_name,
    write_surface,
)

logger = logging.getLogger("AutoCrop3D")

__all__ = ["run"]


def run(
    scans: Path,
    roi: Path,
    output_dir: Path,
    # Declaration order IS the panel's reading order: a client lays its sections
    # out in the order the schema first names them. The options come before
    # `suffix` so the boxes read Inputs, Options, Outputs -- how you crop, then
    # where it goes. Keyword-only from here, so moving one can never change what
    # a positional call means.
    *,
    keep_original_size: bool = False,
    surfaces: Literal["segmentations", "all", "none"] = "segmentations",
    surface_padding_mm: float = 5.0,
    surface_smoothing_iterations: int = 5,
    suffix: str = "cropped",
) -> Path:
    """Crop scans or segmentations to a Region Of Interest drawn in Slicer.

    Args:
        scans: A volume or a folder of them, searched recursively.
            `.nii`, `.nii.gz`, `.nrrd`, `.gipl` and `.gipl.gz` are read.
        roi: A Slicer ROI saved as `.mrk.json`, or a folder of them. One file,
            or a folder holding one, crops every scan; a folder holding several
            is matched to the scans by patient name.
        output_dir: Where the cropped volumes are written. The input folder
            tree is mirrored, and `AutoCrop3D_report.json` records what
            happened to each scan.
        suffix: Appended to each output name.
        keep_original_size: Put the crop back into a volume of the ORIGINAL
            size and geometry, everything outside the box set to zero, so the
            result still overlays the scan it came from. Off means the output
            is only the box.
        surfaces: Also write a smoothed `.vtk` surface of the labels.
            "segmentations" does it for files whose own name says they are one,
            "all" for every scan, "none" for nothing.
        surface_padding_mm: Margin of background added around a label map
            before the surface is built, so a structure touching the edge of
            the crop is still closed.
        surface_smoothing_iterations: Laplacian smoothing passes. 0 keeps the
            raw marching-cubes surface.

    Returns:
        The output directory.
    """
    started = time.monotonic()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if surface_smoothing_iterations < 0:
        raise ValueError(
            f"surface_smoothing_iterations must not be negative, "
            f"got {surface_smoothing_iterations}."
        )

    suffix = str(suffix).strip()
    scan_paths = _discover_scans(str(scans), suffix)
    if not scan_paths:
        raise ValueError(
            f"No scan found in '{os.path.basename(str(scans).rstrip(os.sep))}'. "
            f"Expected one of {', '.join(SCAN_EXTENSIONS)}."
        )

    single_roi, roi_by_patient = _discover_rois(str(roi))

    report = {
        "tool": "AutoCrop3D",
        "suffix": suffix,
        "keep_original_size": keep_original_size,
        "surfaces": surfaces,
        "roi": (
            os.path.basename(single_roi)
            if single_roi
            else {key: os.path.basename(path) for key, path in sorted(roi_by_patient.items())}
        ),
        "cases": {},
        "failed": {},
        "without_a_roi": [],
    }

    written = []
    errors = []
    # One scratch directory for the whole run, removed whatever happens. The
    # surface path needs a file on disk to hand to VTK's NIfTI reader, and
    # upstream put it in the process working directory under a FIXED name --
    # two concurrent requests overwrote each other's.
    with tempfile.TemporaryDirectory(prefix="autocrop3d_") as scratch:
        total = len(scan_paths)
        for index, scan_path in enumerate(scan_paths, start=1):
            # One scan is read, cropped and written before the next is opened,
            # so the position in the batch is exactly what has been done.
            progress.report(index, total, "scan")
            relative = _relative_to(scan_path, str(scans))
            try:
                entry = _crop_one(
                    where=f"scan {index} of {total}",
                    scan_path=scan_path,
                    relative=relative,
                    single_roi=single_roi,
                    roi_by_patient=roi_by_patient,
                    output_dir=output_dir,
                    suffix=suffix,
                    keep_original_size=keep_original_size,
                    surfaces=surfaces,
                    surface_padding_mm=surface_padding_mm,
                    surface_smoothing_iterations=surface_smoothing_iterations,
                    scratch=scratch,
                )
            except _NoRoi as absent:
                # Named, not skipped in silence. Upstream logged a warning and
                # continued, so a whole cohort whose names did not match its
                # ROIs left an empty output folder behind exit code 0.
                report["without_a_roi"].append(
                    {"scan": relative, "patient": str(absent)}
                )
                progress.log(
                    f"scan {index} of {total} matched no ROI and was not cropped",
                    "warning", user=True,
                )
                continue
            except _StepFailed as failed:
                # Per item, so one unreadable file costs one file. Upstream
                # read the volume and the ROI OUTSIDE its try block, so either
                # ended the batch, and wrapped only the write in a bare
                # `except:` that logged and then counted the patient anyway.
                #
                # Logged by position, step, class and message. The scan's file
                # name is patient metadata and many of these messages carry
                # it, so the scan's and the ROI's own names are scrubbed out
                # of the logged text -- the per-scan report, which returns to
                # whoever sent the scans, keeps the full text.
                error = failed.error
                errors.append(error)
                logger.warning(
                    "scan %d of %d: %s failed (%s: %s)",
                    index, total, failed.step, type(error).__name__,
                    _scrubbed(error, failed.names),
                )
                progress.log(
                    f"scan {index} of {total} could not be cropped; the report "
                    f"says why", "warning", user=True,
                )
                report["failed"][relative] = f"{type(error).__name__}: {error}"
                continue

            report["cases"][relative] = entry
            written.append(entry.pop("_absolute"))

    report["summary"] = {
        "scans_found": len(scan_paths),
        "cropped": len(written),
        "failed": len(report["failed"]),
        "without_a_roi": len(report["without_a_roi"]),
        "surfaces": sum(1 for entry in report["cases"].values() if entry.get("surface")),
    }
    report["duration_seconds"] = round(time.monotonic() - started, 2)
    (output_dir / "AutoCrop3D_report.json").write_text(json.dumps(report, indent=2))

    if not written:
        raise _nothing_written(report, scan_paths, roi_by_patient, errors)

    summary = report["summary"]
    partial = summary["failed"] or summary["without_a_roi"]
    logger.log(
        logging.WARNING if partial else logging.INFO,
        "%d of %d scans cropped, %d failed, %d matched no ROI",
        summary["cropped"], summary["scans_found"], summary["failed"],
        summary["without_a_roi"],
    )
    return output_dir


class _NoRoi(Exception):
    """No ROI matched this scan. Carries the patient key that found nothing."""


class _StepFailed(Exception):
    """One scan failed at a named step. Carries the original error.

    `names` are the scan's and the ROI's file names, so the logged message can
    be scrubbed of them; the original error goes to the report untouched.
    """

    def __init__(self, step: str, error: Exception, names: tuple):
        super().__init__(f"{step} failed")
        self.step = step
        self.error = error
        self.names = names


def _scrubbed(error: Exception, names: tuple) -> str:
    """`str(error)` with the given file names and paths replaced."""
    text = str(error)
    # Longest first, so a full path is replaced before the base name inside it.
    for name in sorted({n for n in names if n}, key=len, reverse=True):
        text = text.replace(name, "<file>")
    return text


def _is_input_fault(error: Exception) -> bool:
    """Whether the server would answer this error as the caller's fault.

    By class, the way the server maps it: `ValueError` (which includes the
    unreadable-scan and malformed-ROI errors raised here) and
    `FileNotFoundError` are the caller's to fix; anything else is not.
    """
    return isinstance(error, (ValueError, FileNotFoundError))


def _nothing_written(report: dict, scan_paths: list, roi_by_patient: dict,
                     errors: list) -> Exception:
    """Why the run produced nothing, as the exception to raise.

    The single most valuable message this tool emits, because the failure it
    describes used to be silent: upstream exited 0 with an empty output folder
    whenever the scan keys and the ROI keys disagreed, which they did for every
    cohort whose identifiers contain an underscore.

    A `ValueError` when every failure was the caller's to fix, a
    `RuntimeError` otherwise: a crop that died inside ITK or VTK is not a bad
    request, and must not be answered as one. Patient keys are described by
    their SHAPE, never listed: they are identifiers, and the server would
    redact them into a row of placeholders anyway.
    """
    total = len(scan_paths)
    unmatched = len(report["without_a_roi"])
    if errors:
        counts = {}
        for error in errors:
            key = f"{type(error).__name__}: {_first_line(error)}"
            counts[key] = counts.get(key, 0) + 1
        common, count = max(counts.items(), key=lambda item: item[1])
        # The count before the message, not after it: the server cuts a long
        # reason from the end, and ITK's messages are long.
        message = (
            f"0 of {total} scans cropped; most common failure "
            f"({count} of {total}): {common}"
        )
        if unmatched:
            message += f"; {unmatched} more matched no ROI"
        if all(_is_input_fault(error) for error in errors):
            return ValueError(message)
        return RuntimeError(message)

    scan_keys = sorted({patient_key(os.path.basename(path)) for path in scan_paths})
    roi_keys = sorted(roi_by_patient)
    return ValueError(
        f"0 of {total} scans cropped: none matched an ROI by patient name. "
        f"{_key_mismatch(scan_keys, roi_keys)} Rename the files so the two "
        f"agree, or pass a single ROI to use for every scan."
    )


def _first_line(error: Exception) -> str:
    lines = [line.strip() for line in str(error).splitlines() if line.strip()]
    return lines[0] if lines else ""


def _key_shape(key: str) -> str:
    """A patient key described without its content: `2 parts (letters, digits)`."""
    parts = [part for part in key.split("_")] or [key]

    def kind(part: str) -> str:
        if part.isdigit():
            return "digits"
        if part.isalpha():
            return "letters"
        return "mixed"

    noun = "part" if len(parts) == 1 else "parts"
    return f"{len(parts)} {noun} ({', '.join(kind(part) for part in parts)})"


def _most_common_shape(keys: list) -> str:
    shapes = {}
    for key in keys:
        shape = _key_shape(key)
        shapes[shape] = shapes.get(shape, 0) + 1
    return max(shapes.items(), key=lambda item: item[1])[0] if shapes else "none"


def _key_mismatch(scan_keys: list, roi_keys: list) -> str:
    """How the two sides' patient keys differ, in words an operator can act on."""
    lowered = {key.lower() for key in roi_keys}
    if any(key.lower() in lowered for key in scan_keys):
        return "Some scan and ROI names differ only in letter case."
    scan_shape = _most_common_shape(scan_keys)
    roi_shape = _most_common_shape(roi_keys)
    if scan_shape != roi_shape:
        return (
            f"Scan names reduce to keys of {scan_shape}, ROI names to keys of "
            f"{roi_shape}."
        )
    return (
        f"Both sides reduce to keys of {scan_shape}, but none of the "
        f"{len(scan_keys)} scan keys equals any of the {len(roi_keys)} ROI keys."
    )


def _crop_one(scan_path, roi_by_patient, single_roi, where="", **options) -> dict:
    """One scan through the crop. Returns its report entry, or raises.

    Raises `_NoRoi` when no ROI matches, and `_StepFailed` naming the step for
    anything else, so the per-scan log can say what was being done.
    """
    name = os.path.basename(scan_path)
    patient = patient_key(name)

    roi_path = single_roi or roi_by_patient.get(patient)
    if roi_path is None:
        raise _NoRoi(patient)

    step = ["checking the scan"]
    try:
        return _crop_steps(scan_path, name, patient, roi_path, where, step, **options)
    except Exception as error:  # noqa: BLE001 - reported per scan, never swallowed
        raise _StepFailed(
            step[0], error,
            (scan_path, name, roi_path, os.path.basename(roi_path)),
        ) from error


def _crop_steps(scan_path, name, patient, roi_path, where, step, relative,
                output_dir, suffix, keep_original_size, surfaces,
                surface_padding_mm, surface_smoothing_iterations, scratch) -> dict:
    """The crop itself; `step[0]` always names the stage in progress."""
    import SimpleITK as sitk

    if name.lower().endswith(UNREADABLE_EXTENSIONS):
        raise ValueError(
            f"'{name}' cannot be read: NRRD compresses inside the file, so ITK has no "
            f"reader for a gzipped .nrrd. Decompress it to '.nrrd' first."
        )

    step[0] = "reading the ROI"
    box = read_roi(roi_path, where)
    step[0] = "reading the scan"
    try:
        image = sitk.ReadImage(scan_path)
    except RuntimeError as error:
        # ITK raises RuntimeError for a file it cannot parse, which the server
        # would answer as its own fault. An unreadable upload is the
        # caller's, so it is said as one; ITK's own text follows, last line
        # first, since its multi-line preamble is source paths.
        raise ValueError(f"the scan could not be read: {_last_line(error)}") from error

    step[0] = "cropping"
    lower, upper, clamped = crop_bounds(image, box)
    cropped = crop(image, lower, upper)
    result = repad(image, cropped, lower) if keep_original_size else cropped

    step[0] = "writing the crop"
    destination = output_dir / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination = destination.parent / output_name(name, suffix)
    sitk.WriteImage(result, str(destination))

    entry = {
        "patient": patient,
        "roi": os.path.basename(roi_path),
        # Relative to `output_dir`, never absolute: this report travels to the
        # client, and the server's job directory is no business of its.
        # The union, as every tool of this catalogue now reports it. The
        # cropped volume is always in it; the surface joins it below when one
        # was written.
        "produced": [str(destination.relative_to(output_dir))],
        "_absolute": str(destination),
        "index_lower": list(lower),
        "index_upper": list(upper),
        "size": list(result.GetSize()),
        "clamped_to_the_volume": clamped,
        "roi_orientation_ignored": box.orientation_ignored,
        "roi_coordinate_system": box.coordinate_system,
    }

    step[0] = "building the surface"
    if _wants_surface(surfaces, name):
        surface_path = destination.parent / surface_name(name, suffix)
        labels = write_surface(
            result, str(surface_path), scratch,
            padding_mm=surface_padding_mm,
            smoothing_iterations=surface_smoothing_iterations,
        )
        if labels:
            entry["surface"] = str(surface_path)
            entry["produced"].append(str(surface_path))
            entry["surface_labels"] = labels
        else:
            # An empty crop has no surface. Said, rather than left as a missing
            # file or -- as upstream did -- a KeyError swallowed by `except: pass`.
            entry["surface"] = None
            entry["surface_labels"] = []
    return entry


def _last_line(error: Exception) -> str:
    lines = [line.strip() for line in str(error).splitlines() if line.strip()]
    return lines[-1] if lines else type(error).__name__


def _wants_surface(surfaces: str, name: str) -> bool:
    if surfaces == "none":
        return False
    if surfaces == "all":
        return True
    if surfaces == "segmentations":
        return is_segmentation_name(name)
    raise ValueError(
        f"surfaces must be one of 'segmentations', 'all' or 'none', got '{surfaces}'."
    )


def _relative_to(path: str, root: str) -> str:
    """Where a scan sits under the input root, for mirroring into the output.

    A single file is its own base name. Upstream computed `os.path.relpath`
    against the file itself, got `"."`, and then used `str.replace(".", name)`
    on the whole output path.
    """
    if os.path.isdir(root):
        return os.path.relpath(path, root)
    return os.path.basename(path)


def _discover_scans(root: str, suffix: str) -> list:
    """Every volume under `root`, sorted, minus this run's own earlier output.

    Re-running into the same folder must not re-crop what the last run wrote:
    `P1_cropped.nii.gz` sorts before `P1_scan.nii.gz`. Matched on a whole
    trailing token so a patient called `Cropped_01` is not excluded by the
    default suffix.
    """
    from sadt_areg_common import pairing

    if os.path.isfile(root):
        return [root] if is_scan_file(os.path.basename(root)) else []
    if not os.path.isdir(root):
        raise FileNotFoundError(f"'scans': no such file or folder: '{root}'.")

    fresh, previous = [], []
    for directory, _subdirectories, names in os.walk(root):
        for name in sorted(names):
            if name.startswith(".") or not is_scan_file(name):
                continue
            path = os.path.join(directory, name)
            (previous if pairing.is_previous_output(name, suffix) else fresh).append(path)
    if not fresh and previous:
        # Not an error -- a folder of crops can be cropped again -- but it is
        # the one case where the exclusion above is silently undone, so it is
        # said.
        logger.info(
            "'scans' holds only earlier AutoCrop3D outputs (%d ending in the "
            "suffix); cropping those", len(previous),
        )
    return sorted(fresh) or sorted(previous)


def _discover_rois(root: str):
    """`(single roi path or None, {patient: roi path})`.

    Three shapes, and upstream only handled one of them. It built its lookup
    table `if len(ROIList) > 1`, so a folder holding EXACTLY ONE `.mrk.json`
    left `ROI_Path` pointing at the folder and `open()` raised
    `IsADirectoryError` on the first patient.
    """
    if os.path.isfile(root):
        if not is_roi_file(os.path.basename(root)):
            raise ValueError(
                f"'{os.path.basename(root)}' is not a Slicer ROI. Expected a '.mrk.json' file."
            )
        return root, {}
    if not os.path.isdir(root):
        raise FileNotFoundError(f"'roi': no such file or folder: '{root}'.")

    found = []
    for directory, _subdirectories, names in os.walk(root):
        for name in sorted(names):
            if not name.startswith(".") and is_roi_file(name):
                found.append(os.path.join(directory, name))
    found.sort()

    if not found:
        raise ValueError(
            f"No '.mrk.json' ROI found in '{os.path.basename(root.rstrip(os.sep))}'."
        )
    if len(found) == 1:
        return found[0], {}

    by_patient: dict = {}
    collisions: dict = {}
    for path in found:
        key = patient_key(os.path.basename(path))
        if key in by_patient:
            collisions.setdefault(key, [os.path.basename(by_patient[key])]).append(
                os.path.basename(path)
            )
            continue
        by_patient[key] = path

    if collisions:
        # Upstream's `result[patient] = file` overwrote in silence, so
        # `P01_T1_ROI.mrk.json` and `P01_T2_ROI.mrk.json` both keyed to `P01`,
        # the last one won, and the T1 scans were cropped with the T2 box --
        # a plausible-looking result that is quietly the wrong anatomy.
        detail = "; ".join(
            f"{key}: {', '.join(names)}" for key, names in sorted(collisions.items())
        )
        raise ValueError(
            f"Several ROI files name the same patient, so which one to use is undecidable: "
            f"{detail}. Rename them so each patient has one ROI."
        )
    return None, by_patient
