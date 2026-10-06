"""DICOM -> NIfTI conversion for CBCT input.

Ported from `AREG_CBCT/AREG_CBCT_utils/utils.py::convertdicom2nifti`, with the
three things that made it unusable on a server removed -- the same three ASO's
port removed, since the two CLIs carried the same function:

* it wrote into `<input>/NIFTI/`, i.e. into the caller's own data, which a
  later run then re-discovered as input scans. Everything goes to a scratch
  directory here;
* it only looked one level down (`os.listdir` of the input folder), so a nested
  export was invisible;
* its fallback ran `dicom2nifti.convert_directory` and then took
  `search(output_folder, "nii.gz")[0]` and RENAMED it -- with more than one
  patient already converted that renames an arbitrary earlier file onto the
  current patient's name. The fallback writes into its own empty directory, so
  there is nothing else to pick up.

AREG's own copy rather than an import of `tools/ASO/src/cbct/dicom.py`:
importing another tool's module at load time makes one tool's missing
dependency take both out of the registry. A shared `file_utils` home for it is
the right answer once a third tool needs it.
"""

import logging
import os
import shutil
import tempfile

import SimpleITK as sitk

from sadt_areg_common.errors import ToolInputError, ToolUnavailableError

logger = logging.getLogger(__name__)

_INSTALL_HINT = (
    "Reading this DICOM series needs the dicom2nifti package. Install it with "
    "`pip install -r requirements.txt` (see server/README.md)."
)


def holds_a_series(root: str) -> bool:
    """True when anything under `root` is a readable DICOM series.

    Asked rather than declared. DICOM slices routinely carry no extension at
    all, so a clinician could not tell from a file name either -- which is why
    the panel used to put the question to them, and why the answer being wrong
    was a run that failed for a reason nobody could see. ASO's CBCT engine
    detects it this way and ALI's always has; this is the same question asked
    in the same words.

    AREG's own copy rather than an import of ASO's, for the reason this
    module's docstring already gives about `convert_tree`: importing another
    tool's module at load time makes one tool's missing dependency take both
    out of the registry.

    Stops at the first series found: a cohort of forty patients does not need
    forty answers to a yes/no question.
    """
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


def convert_tree(input_root: str, output_root: str, label: str = "input",
                 argument: str = "") -> str:
    """Convert every DICOM series under `input_root` into `output_root`.

    A directory holding a series becomes `<output_root>/<its relative
    path>.nii.gz`, so a nested export keeps its structure and two patients with
    the same folder name under different parents cannot overwrite each other.

    `label` is the timepoint ("T1", "T2") and `argument` the request field the
    tree came from: a failure names both and the series' POSITION, never its
    folder, which is a patient's name as often as not.

    Returns `output_root`. Raises ToolInputError when the tree holds no series
    or a series neither reader can open -- the caller's data either way.
    """
    os.makedirs(output_root, exist_ok=True)

    # Found first, converted second, so a failure can say "series 3 of 12".
    found = []
    for directory, subdirs, _file_names in os.walk(input_root):
        subdirs.sort()
        series = _series_in(directory)
        if series:
            found.append((directory, series))

    where = f"'{argument}'" if argument else f"the {label} input"
    if not found:
        raise ToolInputError(
            f"No DICOM series found in {where}. Send one folder per patient, or turn "
            f"'dicom_input' off if the scans are already NIfTI/NRRD/GIPL."
        )

    for index, (directory, series) in enumerate(found, start=1):
        relative = os.path.relpath(directory, input_root)
        name = os.path.basename(directory) if relative != "." else "scan"
        destination = os.path.join(
            output_root, os.path.dirname(relative) if relative != "." else "", f"{name}.nii.gz"
        )
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        _convert_series(directory, series, destination,
                        f"{label} DICOM series {index} of {len(found)} in {where}")

    logger.info("AREG: converted %d %s DICOM series", len(found), label)
    return output_root


def _series_in(directory: str) -> tuple:
    reader = sitk.ImageSeriesReader()
    try:
        return tuple(reader.GetGDCMSeriesFileNames(directory))
    except RuntimeError:
        return ()


def _convert_series(directory: str, series: tuple, destination: str,
                    position: str = "a DICOM series") -> None:
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(series)
    # A CBCT export routinely carries tags ITK complains about without the read
    # actually failing; the warnings are noise in the server log and could carry
    # patient identifiers.
    sitk.ProcessObject_SetGlobalWarningDisplay(False)
    try:
        image = reader.Execute()
    except RuntimeError:
        logger.info("AREG: %s refused by GDCM, trying dicom2nifti", position)
        _convert_with_dicom2nifti(directory, destination, position)
        return
    finally:
        sitk.ProcessObject_SetGlobalWarningDisplay(True)
    sitk.WriteImage(image, destination, useCompression=True)


def _convert_with_dicom2nifti(directory: str, destination: str,
                              position: str = "a DICOM series") -> None:
    """Fallback for series SimpleITK's GDCM reader refuses."""
    try:
        import dicom2nifti
    except ImportError as exc:  # pragma: no cover - depends on the deployment
        # The server's missing package, not the caller's data.
        raise ToolUnavailableError(f"{_INSTALL_HINT} (missing: dicom2nifti)") from exc

    staging = tempfile.mkdtemp(prefix="dicom2nifti_", dir=os.path.dirname(destination))
    try:
        try:
            dicom2nifti.convert_directory(directory, staging, compression=True)
        except Exception as exc:  # noqa: BLE001 -- dicom2nifti raises its own zoo
            lines = [line.strip() for line in str(exc).splitlines() if line.strip()]
            raise ToolInputError(
                f"{position} could not be read by either DICOM reader "
                f"({type(exc).__name__}: {lines[-1] if lines else 'no message'})."
            ) from exc
        produced = sorted(name for name in os.listdir(staging) if name.lower().endswith(".nii.gz"))
        if not produced:
            raise ToolInputError(
                f"{position} could not be read by either DICOM reader "
                f"(dicom2nifti produced no volume)."
            )
        shutil.move(os.path.join(staging, produced[0]), destination)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
