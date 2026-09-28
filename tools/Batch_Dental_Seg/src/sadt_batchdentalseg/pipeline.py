"""BatchDentalSeg -- dental CT/CBCT segmentation with the DentalSegmentator
family of nnUNet models.

Ported from `BATCHDENTALSEG/BATCHDENTALSEGLib/SegmentationWidget.py`. That file
is 2940 lines, and most of it is not this pipeline: a queue table, a RAM
watchdog, killing nnUNet processes a crashed scan left behind, a "free memory"
button, a cool-down between scans, restoring the queue from disk after a
crash. All of it exists because the widget runs inside Slicer on a clinician's
laptop and has to survive being out of memory. Here the queue is a folder
argument, concurrency is the server's business, and a failure is an exception.
None of it is ported.

Deliberately NOT ported, each for a stated reason:

* the runtime model download from GitHub releases. A tool holding patient data
  does not make outbound calls mid-run; the server stages the bundles into
  /DATA/BatchDentalSeg/models and passes the path in.
* the auto-crop. Upstream applies it only when its RAM preflight fails, as a
  mitigation for a laptop, and it changes what the network sees. It is not a
  clinical step and this server has the memory.
* the mirroring resolution (`onResolveMirroring`). It is a button the user
  presses after looking at the result, not part of the automatic pipeline.
* the mesh exports (STL/OBJ/GLTF/VTK). The segmentation is the deliverable
  every downstream tool consumes; surfaces are the obvious next addition.

What the move out of the server changed, beyond dropping `base`/`config`/
`file_utils`: scratch space lives under the caller's `output_dir` and is removed
at the end, `segment()` returns the run report rather than a `SegmentationRun`
(tools no longer call each other), zip extraction is gone because the server
unpacks archives before `run()` is called, and `device` / `tile_step_size` are
arguments rather than server settings.
"""

import json
import logging
import os
from pathlib import Path
import shutil
import time

from . import catalogs, mesh_export, nnunet_runner, progress
from .errors import ToolInputError
from sadt_naming import (VOLUME as SCAN_EXTENSIONS, compressed_extension,
                          split_extension as split_scan_extension)

logger = logging.getLogger(__name__)


# The tool's own name, derived rather than written down. `describe.py` publishes
# `tool_dir.name` as the schema's `name`, so taking it from the same place is
# what keeps a report and the schema from disagreeing -- which they already did:
# this file said "CrownSeg" and "BatchDentalSeg" long after the folders were
# renamed to the underscored convention, and nothing noticed because nothing
# reads the field yet.
#
# `parents[2]` is <tool>/src/<package>/thisfile.py -> <tool>.
TOOL_NAME = Path(__file__).resolve().parents[2].name

# Intermediates go here, under the caller's output directory, and are removed
# before returning. A leading dot keeps them out of the way if a run dies
# before the cleanup.
WORK_DIRNAME = ".batchdentalseg_work"

# What nnUNet expects an input case to be called: one modality, index 0000.
_NNUNET_SUFFIX = "_0000.nii.gz"


# ---------------------------------------------------------------------------
# Input discovery
# ---------------------------------------------------------------------------

def is_previous_output(filename: str, suffix: str) -> bool:
    """True if this file looks like something a previous run wrote.

    Without it, a second run on the same folder feeds the first run's
    segmentations back in as input scans.
    """
    base, _extension = split_scan_extension(os.path.basename(filename))
    if suffix and base.endswith(f"_{suffix}"):
        return True
    # The per-segment files, whose names end in the segment they hold.
    return any(base.endswith(f"_{suffix}_{name.replace(' ', '-')}")
               for model in catalogs.MODELS.values()
               for name in model.labels)


def discover_scans(input_path: str, suffix: str) -> list:
    """Resolve the `scans` argument into a list of scan files.

    One scan, or a folder of them for a batch. Folder scanning is RECURSIVE, so
    a nested cohort is processed whole.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if os.path.isfile(input_path):
        return [input_path]

    scans = []
    for root, _dirs, files in os.walk(input_path):
        for name in sorted(files):
            if not name.lower().endswith(SCAN_EXTENSIONS):
                continue
            if is_previous_output(name, suffix):
                continue
            scans.append(os.path.join(root, name))
    return sorted(scans)


def resolve_model(model_path: str) -> tuple:
    """`(the model this bundle is, the nnUNet folder inside it)`.

    The bundle the caller picked identifies the model: its directory name is
    what the manifest downloaded it as. Both failures are the caller's or the
    deployment's data, not a bug, so they raise ToolInputError.
    """
    name = os.path.basename(str(model_path).rstrip(os.sep))
    try:
        model = catalogs.get(name)
    except KeyError:
        raise ToolInputError(
            f"'{name}' is not a BatchDentalSeg model. This tool knows: "
            f"{catalogs.describe_all()}."
        )

    folder = nnunet_runner.find_model_folder(str(model_path))
    if folder is None:
        raise ToolInputError(
            f"The '{name}' bundle holds no usable nnUNet model (expected "
            f"dataset.json, plans.json and fold_0/{nnunet_runner.CHECKPOINT_NAME}). "
            f"Re-fetch it with `scripts/setup-models.sh --tool BatchDentalSeg`."
        )
    return model, folder


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def _convert_to_nifti(scan_path: str, destination: str) -> None:
    """Real format conversion into the NIfTI file nnUNet reads.

    A read plus a write, never a rename: nnUNet's reader picks its format from
    the extension, so an NRRD renamed to .nii.gz is read as garbage. The voxel
    type is left alone -- nnUNet casts to float32 itself.
    """
    import SimpleITK as sitk

    sitk.WriteImage(sitk.ReadImage(scan_path), destination)


def _match_reference_geometry(mask, reference):
    """Put a predicted mask back onto the original scan's exact grid.

    nnUNet resamples to the model's spacing and back, which can leave a mask
    whose origin differs from its scan in the last float digits -- enough for a
    viewer to draw it offset from the anatomy it describes.
    """
    import numpy as np
    import SimpleITK as sitk

    if (
        mask.GetSize() == reference.GetSize()
        and np.allclose(mask.GetSpacing(), reference.GetSpacing(), atol=1e-4)
    ):
        mask.CopyInformation(reference)
        return mask

    return sitk.Resample(
        mask, reference, sitk.Transform(), sitk.sitkNearestNeighbor, 0, mask.GetPixelID()
    )


def _write_segmentation(image, destination: str) -> str:
    """Write a label volume, always compressed: these are long runs of one
    value, so gzip takes them roughly a hundred times down."""
    import SimpleITK as sitk

    os.makedirs(os.path.dirname(destination), exist_ok=True)
    sitk.WriteImage(image, destination, useCompression=True)
    return destination


def _split_segments(labels, model: catalogs.Model, base: str,
                    extension: str, output_dir: str, suffix: str) -> list:
    """One binary file per label the network actually emitted.

    Only the labels PRESENT in this scan are written: a full UniversalLab run
    would otherwise produce 55 files per patient, most of them empty, and an
    empty mask is indistinguishable from a structure the model failed on.
    """
    import numpy as np
    import SimpleITK as sitk

    array = sitk.GetArrayViewFromImage(labels)
    present = set(int(value) for value in np.unique(array) if value != 0)

    written = []
    for name, value in model.labels.items():
        if value not in present:
            continue
        binary = sitk.GetImageFromArray((array == value).astype(np.uint8))
        binary.CopyInformation(labels)
        safe_name = name.replace(" ", "-").replace("/", "-")
        destination = os.path.join(
            output_dir, f"{base}_{suffix}_{safe_name}{extension}"
        )
        written.append(_write_segmentation(binary, destination))
    return written


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

def segment(
    input_path: str,
    model_path: str,
    output_dir: str,
    separate_segments: bool = False,
    prediction_ID: str = "Seg",
    device: str = "cuda",
    tile_step_size: float = 0.5,
    gpu_resampling: bool = True,
    export_formats=(mesh_export.NIFTI,),
    surface_decimation: int = 0,
) -> dict:
    """Segment every scan under `input_path` with one model bundle.

    `model_path` is the path of the bundle the caller picked; its directory
    name is which of catalogs.MODELS it is.
    """
    import SimpleITK as sitk

    started = time.monotonic()

    # Before any scan is read: an unusable bundle is a property of the
    # deployment, and discovering it inside the loop would report it once per
    # patient.
    model, model_folder = resolve_model(model_path)
    device = nnunet_runner.resolve_device(device)

    scans = discover_scans(input_path, prediction_ID)
    if not scans:
        raise ToolInputError(
            "No scan found in the input. Expected one of "
            f"{', '.join(SCAN_EXTENSIONS)}, or a folder of them."
        )

    input_root = input_path if os.path.isdir(input_path) else os.path.dirname(scans[0])
    output_dir = os.fspath(output_dir)
    work_dir = os.path.join(output_dir, WORK_DIRNAME)
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # One nnUNet call for the whole batch, so the checkpoint is loaded once.
    # The case ids are positional, never derived from the patient's file name:
    # nnUNet writes its output beside its input under the same id, and two
    # scans called scan.nii.gz in different subfolders would collide.
    nnunet_input = os.path.join(work_dir, "nnunet_in")
    nnunet_output = os.path.join(work_dir, "nnunet_out")
    os.makedirs(nnunet_input, exist_ok=True)

    def _describe(path: str) -> str:
        return os.path.relpath(path, input_root) if os.path.isdir(input_root) else os.path.basename(path)

    cases = {}
    failed_conversions = []
    # Only the two ends of this run can be counted. The segmentation between
    # them is ONE nnUNet call over the whole folder, so there is no per-scan
    # position to report inside it -- the bar stops where the reading ends and
    # moves again when the writing starts, rather than inventing a number.
    for index, scan in enumerate(scans):
        progress.report(index + 1, len(scans), "reading scan", end=0.1)
        case_id = f"case_{index:04d}"
        try:
            _convert_to_nifti(scan, os.path.join(nnunet_input, f"{case_id}{_NNUNET_SUFFIX}"))
        except Exception as exc:  # noqa: BLE001 - one unreadable scan must not end the batch
            # Guarded per scan, and deliberately: this loop runs BEFORE
            # inference, so without it one corrupt file in a cohort of forty
            # would abort the whole run before a single scan was segmented.
            logger.exception("BatchDentalSeg: could not read a scan")
            failed_conversions.append(
                {
                    "case_id": case_id,
                    "input": _describe(scan),
                    "status": "failed",
                    "error": f"could not be read ({type(exc).__name__}: {exc})",
                }
            )
            continue
        cases[case_id] = scan

    if not cases:
        raise ToolInputError(
            "None of the input scans could be read. Check the files are valid "
            "medical volumes."
        )

    logger.info("BatchDentalSeg: %d scan(s), model=%s, device=%s", len(cases), model.name, device)
    progress.emit(0.1, f"segmenting {len(cases)} scan(s) in one pass")
    try:
        nnunet_runner.predict_folder(
            model_folder, nnunet_input, nnunet_output, device,
            tile_step_size=tile_step_size, gpu_resampling=gpu_resampling,
        )
    except Exception:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise

    report_cases = {entry["case_id"]: entry for entry in failed_conversions}
    for index, (case_id, scan) in enumerate(cases.items(), start=1):
        progress.report(index, len(cases), "writing scan", start=0.9)
        entry = {"case_id": case_id, "input": _describe(scan)}
        predicted = os.path.join(nnunet_output, f"{case_id}.nii.gz")
        if not os.path.isfile(predicted):
            # Reported per scan rather than raised: one unreadable patient in a
            # cohort of forty must not lose the other thirty-nine.
            entry.update(status="failed", error="nnUNet produced no output for this scan")
            report_cases[case_id] = entry
            continue

        try:
            reference = sitk.ReadImage(scan)
            labels = _match_reference_geometry(sitk.ReadImage(predicted), reference)

            base, extension = split_scan_extension(os.path.basename(scan))
            extension = compressed_extension(extension)
            # The output mirrors the input tree, so two patients whose scans
            # share a file name stay apart.
            relative = (
                os.path.relpath(os.path.dirname(scan), input_root)
                if os.path.isdir(input_root)
                else "."
            )
            scan_output_dir = os.path.normpath(os.path.join(output_dir, relative))

            produced = []
            # `separate_segments` splits the VOLUME, so it follows NIFTI: a
            # caller who asked for meshes only is asking for no volume, and
            # one binary NIfTI per label is still a volume.
            if mesh_export.NIFTI in export_formats:
                produced.append(_write_segmentation(
                    labels,
                    os.path.join(scan_output_dir, f"{base}_{prediction_ID}{extension}"),
                ))
                if separate_segments:
                    produced.extend(_split_segments(
                        labels, model, base, extension, scan_output_dir, prediction_ID
                    ))
            produced.extend(mesh_export.write(
                labels, model, base, scan_output_dir, prediction_ID,
                export_formats, decimation=surface_decimation,
            ))
            entry.update(status="ok", produced=produced)
        except Exception as exc:  # noqa: BLE001 - one bad scan must not end the batch
            logger.exception("BatchDentalSeg: scan failed")
            entry.update(status="failed", error=f"{type(exc).__name__}: {exc}")

        report_cases[case_id] = entry

    succeeded = [entry for entry in report_cases.values()
                 if entry.get("status") == "ok"]
    report = {
        "tool": TOOL_NAME,
        "model": model.name,
        "model_description": model.description,
        # Published with the results: the segmentation is a label volume, and
        # without this table its integers mean nothing to whoever opens it.
        "labels": model.labels,
        "device": device,
        "prediction_ID": prediction_ID,
        "separate_segments": separate_segments,
        # Recorded because it is lossy by default: the input resampling runs at
        # spline order 1 rather than 3, so a result produced this way is not
        # bit-identical to one produced with it off. Whoever reads a
        # segmentation must be able to see which pipeline made it.
        "gpu_resampling": bool(gpu_resampling) and device.startswith("cuda"),
        "tile_step_size": float(tile_step_size),
        # Both recorded for the same reason the resampling is: a mesh is a
        # lossy view of the mask it came from, and whoever opens one has to be
        # able to see how much of it was thrown away.
        "export_formats": list(export_formats),
        "surface_decimation": int(surface_decimation),
        "cases": report_cases,
        "summary": f"{len(succeeded)}/{len(report_cases)} scan(s) segmented",
        "duration_seconds": round(time.monotonic() - started, 2),
    }

    # The nnUNet intermediates sit inside the directory the caller will ship,
    # and one predicted volume per scan is not small.
    shutil.rmtree(work_dir, ignore_errors=True)

    report_path = os.path.join(output_dir, "BatchDentalSeg_report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    return report
