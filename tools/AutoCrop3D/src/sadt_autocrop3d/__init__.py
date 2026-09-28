"""Crop CBCT volumes and their segmentations to a Slicer ROI box."""

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Literal

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
    # One scratch directory for the whole run, removed whatever happens. The
    # surface path needs a file on disk to hand to VTK's NIfTI reader, and
    # upstream put it in the process working directory under a FIXED name --
    # two concurrent requests overwrote each other's.
    with tempfile.TemporaryDirectory(prefix="autocrop3d_") as scratch:
        for scan_path in scan_paths:
            name = os.path.basename(scan_path)
            relative = _relative_to(scan_path, str(scans))
            try:
                entry = _crop_one(
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
                continue
            except Exception as error:  # noqa: BLE001 - reported per scan, never swallowed
                # Per item, so one unreadable file costs one file. Upstream
                # read the volume and the ROI OUTSIDE its try block, so either
                # ended the batch, and wrapped only the write in a bare
                # `except:` that logged and then counted the patient anyway.
                logger.warning("AutoCrop3D failed on %s: %s", name, error)
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
        raise ValueError(_nothing_written_message(report, scan_paths, roi_by_patient))

    return output_dir


class _NoRoi(Exception):
    """No ROI matched this scan. Carries the patient key that found nothing."""


def _nothing_written_message(report: dict, scan_paths: list, roi_by_patient: dict) -> str:
    """Why the run produced nothing, in terms the caller can act on.

    The single most valuable message this tool emits, because the failure it
    describes used to be silent: upstream exited 0 with an empty output folder
    whenever the scan keys and the ROI keys disagreed, which they did for every
    cohort whose identifiers contain an underscore.
    """
    failures = list(report["failed"].values())
    if failures:
        return (
            f"AutoCrop3D cropped nothing: all {len(scan_paths)} scan(s) failed. "
            + "; ".join(failures[:5])
        )
    scan_keys = sorted({patient_key(os.path.basename(path)) for path in scan_paths})
    return (
        f"AutoCrop3D cropped nothing: none of the {len(scan_paths)} scan(s) matched an ROI. "
        f"Scan patients: {', '.join(scan_keys[:8])}. "
        f"ROI patients: {', '.join(sorted(roi_by_patient)[:8])}. "
        f"Rename the files so the two agree, or pass a single ROI to use for every scan."
    )


def _crop_one(scan_path, relative, single_roi, roi_by_patient, output_dir, suffix,
              keep_original_size, surfaces, surface_padding_mm,
              surface_smoothing_iterations, scratch) -> dict:
    """One scan through the crop. Returns its report entry, or raises."""
    import SimpleITK as sitk

    name = os.path.basename(scan_path)
    patient = patient_key(name)

    roi_path = single_roi or roi_by_patient.get(patient)
    if roi_path is None:
        raise _NoRoi(patient)

    if name.lower().endswith(UNREADABLE_EXTENSIONS):
        raise ValueError(
            f"'{name}' cannot be read: NRRD compresses inside the file, so ITK has no "
            f"reader for a gzipped .nrrd. Decompress it to '.nrrd' first."
        )

    box = read_roi(roi_path)
    image = sitk.ReadImage(scan_path)

    lower, upper, clamped = crop_bounds(image, box)
    cropped = crop(image, lower, upper)
    result = repad(image, cropped, lower) if keep_original_size else cropped

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
        raise FileNotFoundError(f"No such file or folder: '{root}'.")

    fresh, previous = [], []
    for directory, _subdirectories, names in os.walk(root):
        for name in sorted(names):
            if name.startswith(".") or not is_scan_file(name):
                continue
            path = os.path.join(directory, name)
            (previous if pairing.is_previous_output(name, suffix) else fresh).append(path)
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
        raise FileNotFoundError(f"No such file or folder: '{root}'.")

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
