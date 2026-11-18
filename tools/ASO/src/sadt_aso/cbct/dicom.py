"""DICOM -> NIfTI conversion for CBCT input.

Ported from `ASO_CBCT_utils/utils.py::convertdicom2nifti`, with the three
things that made it unusable on a server removed:

* it wrote its output into `<input>/NIFTI/`, i.e. into the caller's own data,
  which a later run then re-discovered as input scans. Everything written here
  goes to a scratch directory instead;
* it only looked one level down, so a nested export was invisible;
* its fallback path ran `dicom2nifti.convert_directory`, then took
  `search(output_folder, "nii.gz")[0]` and RENAMED it -- with more than one
  patient already converted, that renames an arbitrary earlier file onto the
  current patient's name. The fallback writes to its own empty directory here,
  so there is nothing else to pick up.
"""

import logging
import os
import shutil
import tempfile

import SimpleITK as sitk

from .. import progress
from ..errors import ToolInputError
from ..markups import is_markups_file

logger = logging.getLogger(__name__)


class DicomStackMissing(RuntimeError):
    """The deployment lacks the fallback reader: the server's fault, never the
    caller's, so it must not be turned into an input error below."""

_INSTALL_HINT = (
    "Reading this DICOM series needs the dicom2nifti package. Install it with "
    "`pip install -r requirements.txt` (see server/README.md)."
)


def holds_a_series(root: str) -> bool:
    """True when anything under `root` is a readable DICOM series.

    Asked rather than declared. DICOM slices routinely carry no extension at
    all, so a clinician could not tell from a file name either -- which is why
    the panel used to put the question to them, and why the answer being wrong
    was a run that failed for a reason nobody could see. ALI's CBCT engine has
    always detected it this way; this is the same question asked in the same
    words.

    Stops at the first series found: a cohort of forty patients does not need
    forty answers to a yes/no question.
    """
    # SimpleITK is imported at module level here; GDCM comes with it.
    reader = sitk.ImageSeriesReader()
    for directory, _subdirs, _names in os.walk(root):
        try:
            if reader.GetGDCMSeriesFileNames(directory):
                return True
        except RuntimeError:
            # GDCM raises on a directory it cannot even scan. That is "no
            # series here", not a failure of the run.
            continue
    return False


def convert_tree(input_root: str, output_root: str) -> str:
    """Convert every DICOM series under `input_root` into `output_root`.

    A directory holding a series becomes `<output_root>/<its relative path>.nii.gz`,
    so a nested export keeps its structure and two patients with the same folder
    name in different parents cannot overwrite each other. Markups files found
    anywhere in the tree are copied across at the same relative position, which
    is what lets semi-automated mode take DICOM input.

    Returns `output_root`. Raises RuntimeError when the tree holds no series.
    """
    os.makedirs(output_root, exist_ok=True)

    # Listed first and converted second, so each series can be logged as
    # "i of n": one unreadable export among forty used to stop the batch with
    # whatever GDCM said and no hint of which series, or how far it had got.
    found = []
    for directory, _, file_names in os.walk(input_root):
        relative = os.path.relpath(directory, input_root)
        for file_name in file_names:
            if is_markups_file(file_name):
                destination = os.path.join(output_root, relative, file_name)
                os.makedirs(os.path.dirname(destination), exist_ok=True)
                shutil.copy2(os.path.join(directory, file_name), destination)

        series = _series_in(directory)
        if series:
            found.append((directory, relative, series))

    if not found:
        raise RuntimeError(
            "No DICOM series found in the input. Send a zip of one folder per "
            "patient, or turn dicom_input off if the input is already NIfTI/NRRD/GIPL."
        )

    progress.emit(None, f"converting {len(found)} DICOM series")
    for index, (directory, relative, series) in enumerate(found, start=1):
        name = os.path.basename(directory) if relative != "." else "scan"
        destination = os.path.join(
            output_root, os.path.dirname(relative) if relative != "." else "", f"{name}.nii.gz"
        )
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        logger.info("DICOM series %d of %d: converting %d slice(s)",
                    index, len(found), len(series))
        try:
            _convert_series(directory, series, destination)
        except DicomStackMissing:
            raise
        except Exception as exc:
            # Both readers refused the series the caller sent, so the fix is
            # theirs: the position says which, the cause says why. The folder
            # is never named -- it is the patient.
            logger.warning(
                "DICOM series %d of %d: conversion failed (%s: %s)",
                index, len(found), type(exc).__name__, exc,
            )
            raise ToolInputError(
                f"DICOM series {index} of {len(found)} could not be converted "
                f"({type(exc).__name__}: {exc})"
            ) from exc
    logger.info("Converted %d DICOM series", len(found))
    return output_root


def _series_in(directory: str) -> tuple:
    reader = sitk.ImageSeriesReader()
    try:
        return tuple(reader.GetGDCMSeriesFileNames(directory))
    except RuntimeError:
        return ()


def _convert_series(directory: str, series: tuple, destination: str) -> None:
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(series)
    # A CBCT export routinely carries tags ITK complains about without the read
    # actually failing; the warnings are noise in the server log and could carry
    # patient identifiers.
    sitk.ProcessObject_SetGlobalWarningDisplay(False)
    try:
        image = reader.Execute()
    except RuntimeError:
        _convert_with_dicom2nifti(directory, destination)
        return
    finally:
        sitk.ProcessObject_SetGlobalWarningDisplay(True)
    sitk.WriteImage(image, destination, useCompression=True)


def _convert_with_dicom2nifti(directory: str, destination: str) -> None:
    """Fallback for series SimpleITK's GDCM reader refuses.

    Imported here rather than at module level: registry.py imports every tool at
    startup, and a server with no DICOM stack must still be able to serve ASO's
    other modes.
    """
    try:
        import dicom2nifti
    except ImportError as exc:  # pragma: no cover - depends on the deployment
        raise DicomStackMissing(f"{_INSTALL_HINT} (missing: dicom2nifti)") from exc

    staging = tempfile.mkdtemp(prefix="dicom2nifti_", dir=os.path.dirname(destination))
    try:
        dicom2nifti.convert_directory(directory, staging, compression=True)
        produced = sorted(
            name for name in os.listdir(staging) if name.lower().endswith(".nii.gz")
        )
        if not produced:
            raise RuntimeError("dicom2nifti wrote no volume for this series")
        shutil.move(os.path.join(staging, produced[0]), destination)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
