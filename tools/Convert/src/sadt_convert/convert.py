"""Read every scan or DICOM series under a path and write it in one format.

Other tools call this one (`sup.run("Convert", ...)`) when the file a previous
step wrote is not the family they read: ASO writes its oriented scan with the
input's extension, so a `.nrrd` in gives a `.nrrd` out. Keeping the readers here
means a calling tool needs no image library of its own.

Written for the people who see only a few redacted lines: messages carry a
position ("scan 3 of 40") and an exception class, never a file or folder name.
"""

import json
import logging
import os
import shutil
from pathlib import Path

from . import progress
from .errors import ToolInputError

logger = logging.getLogger(__name__)

TARGETS = ("nii.gz", "nrrd", "gipl.gz")
REPORT_NAME = "run_report.json"

def _reason(exc: Exception) -> str:
    """The class and a fixed phrase. A library's own text quotes the file, and a file name
    (with spaces in it, no pattern can tell where it ends) is patient metadata."""
    if isinstance(exc, SeriesError):
        return str(exc)
    return f"{type(exc).__name__} while reading or writing the scan"


class SeriesError(ValueError):
    """A DICOM series this tool refuses to convert; the message is ours, so it is safe to show."""


def _series_in(directory: str) -> list:
    """[(file names, ...)] one tuple per DICOM series in this folder, in a stable order."""
    import SimpleITK as sitk

    # "No Series can be found" is printed for every folder without one.
    sitk.ProcessObject_SetGlobalWarningDisplay(False)
    try:
        reader = sitk.ImageSeriesReader()
        ids = sorted(reader.GetGDCMSeriesIDs(directory))
        return [tuple(reader.GetGDCMSeriesFileNames(directory, uid)) for uid in ids]
    except RuntimeError:
        # GDCM raises on a folder it cannot scan: "no series here", not a failure.
        return []
    finally:
        sitk.ProcessObject_SetGlobalWarningDisplay(True)


def find_items(root: Path) -> list:
    """`(case, kind, source, parent)` for every volume file and DICOM series (`source` is the file, or the series' files), in a stable order.

    `case` is the path relative to `root` without its extension (a series: its
    folder), so two patients with the same file name in different folders stay apart.
    """
    from sadt_naming import VOLUME, split_extension

    if root.is_file():
        stem, extension = split_extension(root.name)
        if extension.lower() not in VOLUME:
            raise ToolInputError("'scans' is not a 3D scan file this tool reads (nii.gz, nrrd, gipl).")
        return [(stem, "file", root, "")]
    if not root.is_dir():
        raise ToolInputError("'scans' path does not exist.")

    found = []
    for directory, subdirs, names in os.walk(root):
        subdirs.sort()
        relative = os.path.relpath(directory, root)
        relative = "" if relative == "." else relative
        for name in sorted(names):
            stem, extension = split_extension(name)
            if extension.lower() in VOLUME and extension.lower() != ".nrrd.gz":  # ITK cannot read it
                found.append((os.path.join(relative, stem), "file", Path(directory) / name, relative))
        series = _series_in(directory)
        for k, files in enumerate(series, start=1):
            # One case per series: a folder holding a scout and the scan is two scans.
            case = relative or "scan"
            parent = os.path.dirname(relative)
            if len(series) > 1:
                case, parent = os.path.join(case, f"series_{k}"), relative
            found.append((case, "dicom", files, parent))
    if not found:
        raise ToolInputError("'scans' holds no 3D scan (nii.gz, nrrd, gipl) and no DICOM series.")
    return found


def _check_slices(files: tuple) -> None:
    """Refuse a series with a missing or unevenly spaced slice: ITK would average the spacing."""
    import SimpleITK as sitk

    reader = sitk.ImageFileReader()
    positions = []
    for name in files:
        reader.SetFileName(name)
        reader.ReadImageInformation()
        positions.append(reader.GetOrigin())
    if len(positions) < 3:
        return
    import math

    # Gantry tilt: equal steps, but the slices drift sideways, and ITK stacks them straight and
    # stretches the spacing. The stack axis must be the normal of the slices.
    reader.SetFileName(files[0])
    reader.ReadImageInformation()
    direction = reader.GetDirection()
    normal = (direction[2], direction[5], direction[8])
    axis = [b - a for a, b in zip(positions[0], positions[-1])]
    length = math.sqrt(sum(v * v for v in axis))
    if length > 0:
        cosine = abs(sum(n * v for n, v in zip(normal, axis))) / length
        if cosine < math.cos(math.radians(0.5)):
            raise SeriesError("DICOM series has tilted slices (gantry tilt)")
    steps = [math.dist(a, b) for a, b in zip(positions, positions[1:])]
    if min(steps) <= 0 or (max(steps) - min(steps)) > 0.05 * (sum(steps) / len(steps)):
        raise SeriesError("DICOM series has missing or unevenly spaced slices")


def _write(source, kind: str, destination: Path, target: str, original: Path) -> None:
    import SimpleITK as sitk
    from sadt_naming import split_extension

    if kind == "file" and split_extension(original.name)[1].lower() == "." + target:
        # Already in this format: keep the bytes, once the header proves it is a scan
        # (and, for a .gz target, that the bytes really are gzip: ITK also opens the plain file).
        reader = sitk.ImageFileReader()
        reader.SetFileName(str(source))
        reader.ReadImageInformation()
        if target.endswith(".gz") and Path(source).read_bytes()[:2] != b"\x1f\x8b":
            raise SeriesError("the file is not compressed although its name says so")
        shutil.copy2(source, destination)
        return
    sitk.ProcessObject_SetGlobalWarningDisplay(False)  # tag warnings can carry patient ids
    try:
        if kind == "dicom":
            _check_slices(source)
            reader = sitk.ImageSeriesReader()
            reader.SetFileNames(source)
            image = reader.Execute()
        else:
            image = sitk.ReadImage(str(source))
    finally:
        sitk.ProcessObject_SetGlobalWarningDisplay(True)
    sitk.WriteImage(image, str(destination), useCompression=True)
    # Not every format keeps everything (GIPL drops the direction and the components, NIfTI
    # makes the direction orthogonal): what was written must say what was read.
    back = sitk.ReadImage(str(destination))
    same = (back.GetNumberOfComponentsPerPixel() == image.GetNumberOfComponentsPerPixel()
            and max(abs(a - b) for a, b in zip(back.GetDirection(), image.GetDirection())) < 1e-4
            and max(abs(a - b) for a, b in zip(back.GetSpacing(), image.GetSpacing())) < 1e-5
            and max(abs(a - b) for a, b in zip(back.GetOrigin(), image.GetOrigin())) < 1e-3 + 1e-6 * max(abs(v) for v in image.GetOrigin()))
    if not same:
        raise SeriesError(f"the {target} format cannot keep this scan's geometry or components")


def convert_tree(scans: Path, output_dir: Path, to: str) -> Path:
    if to not in TARGETS:
        raise ToolInputError(f"Unknown format '{to}'. Available: {', '.join(TARGETS)}.")
    items = find_items(scans)
    output_dir.mkdir(parents=True, exist_ok=True)

    cases, taken, failures = {}, {}, 0
    for position, (case, kind, source, parent) in enumerate(items, start=1):
        progress.report(position, len(items), "scan")
        name = os.path.basename(case)
        relative = os.path.join(parent, f"{name}.{to}")
        original = source if kind == "file" else Path(source[0]).parent
        shown = os.path.relpath(original, scans)
        entry = {"input": scans.name if (not scans.is_dir() or shown == ".") else shown}
        if relative.casefold() in taken:
            # Two inputs would write the same file (a.nrrd and a.nii.gz side by side).
            entry.update(status="skipped: another input has the same name", produced=[])
            failures += 1
            logger.warning("scan %d of %d: skipped, its name is taken", position, len(items))
        else:
            taken[relative.casefold()] = case
            destination = output_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                _write(source, kind, destination, to, original)
                entry.update(status="ok", produced=[relative])
            except Exception as exc:  # one unreadable scan must not cost the rest
                destination.unlink(missing_ok=True)
                if destination.parent != output_dir:
                    try:
                        destination.parent.rmdir()  # only if empty
                    except OSError:
                        pass
                entry.update(status=f"failed: {_reason(exc)}", produced=[])
                failures += 1
                logger.warning("scan %d of %d: conversion failed (%s)", position, len(items), type(exc).__name__)
        cases[case if case not in cases else f"{case}#{position}"] = entry

    done = len(items) - failures
    if not done:
        raise ToolInputError(
            f"None of the {len(items)} scan(s) could be converted. First reason: "
            f"{next(iter(cases.values()))['status']}."
        )
    (output_dir / REPORT_NAME).write_text(json.dumps({
        "tool": "Convert",
        "to": to,
        "summary": {"converted": done, "failed": failures},
        "cases": cases,
    }, indent=2))
    logger.info("Convert: %d of %d scan(s) written as %s", done, len(items), to)
    return output_dir
