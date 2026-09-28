"""AMASSS -- Automatic Multi-Anatomical Skull Structure Segmentation.

Ported from AMASSS_CLI.py by way of the server-side tool. The business logic
(nnUNet per structure, binary masks, optional multi-label merge, optional
surfaces) is preserved; the CLI envelope is not -- `sys.argv` parsing,
`sys.exit()`, the `<filter-progress>` protocol and the caller-supplied temp
folders are gone, and a failure is an exception.

What the move out of the server changed, beyond dropping `base`/`config`/
`file_utils`:

* Scratch space lives under the caller's `output_dir` and is removed at the
  end, because a tool must not write anywhere else.
* `segment()` returns the run report. The server-side version returned a
  `SegmentationRun` so that another in-process tool could pick up the files
  directly; tools no longer call each other, so that reason is gone.
* Zip extraction is gone. The server unpacks archives before `run()` is called.

See the comments marked "FIX:" for the original CLI's defects corrected here.
"""

import json
import logging
import os
from pathlib import Path
import re
import shutil
import time

from . import nnunet_runner, progress, vtk_export
from .catalog import (
    DEFAULT_MERGE_MODES,
    DEFAULT_STRUCTURES,
    LABEL_COLORS,
    LABELS,
    MERGE_MODES,
    MERGING_ORDER,
    NAMES_FROM_LABELS,
    STRUCTURE_CODES,
)
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
WORK_DIRNAME = ".amasss_work"


# ---------------------------------------------------------------------------
# Input discovery
# ---------------------------------------------------------------------------

def is_previous_output(filename: str, prediction_id: str) -> bool:
    """True if this file looks like a segmentation AMASSS itself produced.

    Without this, running twice on the same folder feeds the first run's
    outputs back in as input scans. Outputs are recognized by the suffixes
    AMASSS actually writes.
    """
    base, _extension = split_scan_extension(os.path.basename(filename))

    if prediction_id and f"_{prediction_id}_" in f"{base}_":
        return True
    if base.endswith("_MERGED"):
        return True
    for code in STRUCTURE_CODES:
        if base.endswith(f"_{code}"):
            return True
    return bool(re.search(r"_Seg(Out)?$", base))


def discover_scans(input_path: str, prediction_id: str) -> list:
    """Resolve the `scans` argument into a list of scan files.

    One scan file, or a folder of them for a batch. A folder is the point: the
    runner pays a process start-up cost plus one checkpoint load per structure,
    so 40 scans have to be one call.

    FIX: folder scanning is RECURSIVE. The Slicer UI counted scans recursively
    while the CLI listed only the top level, so on any nested dataset the UI
    announced N scans and the CLI silently processed a subset.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if os.path.isfile(input_path):
        return [input_path]

    found = []
    previous_outputs = 0
    for root, _dirs, files in os.walk(input_path):
        for name in files:
            if not name.lower().endswith(SCAN_EXTENSIONS):
                continue
            if is_previous_output(name, prediction_id):
                previous_outputs += 1
                continue
            found.append(os.path.join(root, name))

    # How many, never which: a file name is patient metadata, and a tool's log
    # outlives the run that wrote it. The count is what a caller needs here --
    # it says whether the folder was re-fed to a second run.
    if previous_outputs:
        logger.info(
            "Skipping %d file(s) that look like a previous AMASSS output",
            previous_outputs,
        )

    if not found:
        # FIX: the original called sys.exit(1) here, which produces no usable
        # error for a caller that is not a shell.
        raise FileNotFoundError(
            f"No scan found in {input_path}. Supported extensions: "
            f"{', '.join(SCAN_EXTENSIONS)}"
        )
    return sorted(found)


def resolve_models(model_path: str, structures):
    """Map each requested structure to its nnUNet model folder.

    Returns (available, missing).
    """
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Model bundle is not a directory: {model_path}")

    # A bundle may be wrapped in a single top-level folder (a copy of
    # "AMASSS_Models/" rather than of its contents). Descend into it so both
    # layouts work.
    if not any(os.path.isdir(os.path.join(model_path, code)) for code in STRUCTURE_CODES):
        entries = [
            os.path.join(model_path, name)
            for name in sorted(os.listdir(model_path))
            if os.path.isdir(os.path.join(model_path, name)) and name != "__MACOSX"
        ]
        if len(entries) == 1:
            model_path = entries[0]

    available, missing = {}, []
    for code in structures:
        model_folder = nnunet_runner.find_model_folder(model_path, code)
        if model_folder is None:
            logger.warning("No usable model for structure '%s' in %s", code, model_path)
            missing.append(code)
        else:
            available[code] = model_folder

    if not available:
        raise nnunet_runner.ModelNotFoundError(
            f"No nnUNet model found in '{os.path.basename(model_path)}' for any of the "
            f"requested structures ({', '.join(structures)}). Expected "
            f"<bundle>/<CODE>/**/*__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth"
        )
    return available, missing


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _convert_to_nifti(scan_path: str, destination: str) -> None:
    """Real format conversion into the NIfTI file nnUNet will read.

    FIX: the original did `shutil.copy(volume_file, "p_XXX_0000.nii.gz")` --
    an NRRD renamed to .nii.gz, handed to a reader that picks its format from
    the extension. NRRD is Slicer's own default format, so "supported" input
    was in practice only reliable for NIfTI. A read + write actually converts.

    The voxel type is left alone. Casting to float32 first, as this did, was
    not what made the conversion real -- the read and write are -- and on a CBCT
    it doubled the bytes to gzip on the way out and to gunzip on the way back
    in, for 2.4s + 0.4s per scan buying nothing: nnUNet's reader casts to
    float32 itself, and int16 CBCT values are exact in float32 either way.
    """
    import SimpleITK as sitk

    image = sitk.ReadImage(scan_path)
    sitk.WriteImage(image, destination)


def _match_reference_geometry(mask_image, reference):
    """Put a predicted mask back onto the original scan's exact grid."""
    import numpy as np
    import SimpleITK as sitk

    same_geometry = (
        mask_image.GetSize() == reference.GetSize()
        and np.allclose(mask_image.GetSpacing(), reference.GetSpacing())
        and np.allclose(mask_image.GetOrigin(), reference.GetOrigin())
        and np.allclose(mask_image.GetDirection(), reference.GetDirection())
    )
    if not same_geometry:
        mask_image = sitk.Resample(
            mask_image, reference, sitk.Transform(), sitk.sitkNearestNeighbor, 0,
            mask_image.GetPixelID(),
        )
    return sitk.Cast(mask_image, sitk.sitkInt16)


def _write_segmentation(array, reference, output_path: str) -> str:
    import numpy as np
    import SimpleITK as sitk

    image = sitk.GetImageFromArray(array.astype(np.int16))
    image.CopyInformation(reference)
    # useCompression covers the formats whose extension does not already imply
    # it (.nrrd); for a .nii.gz the gz is applied from the name either way.
    # Harmless where compression is already on, so it is not conditional.
    sitk.WriteImage(
        _match_reference_geometry(image, reference), output_path, useCompression=True
    )
    return output_path


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def segment(
    input_path: str,
    model_path: str,
    output_dir: str,
    structures=None,
    merge=None,
    prediction_ID: str = "Pred",
    generate_surface: bool = False,
    surface_smoothing: int = 5,
    surface_decimation: int = 90,
    device: str = "cuda",
    tile_step_size: float = 0.5,
    gpu_resampling: bool = True,
) -> dict:
    """Segment one scan or a batch under `output_dir`, and return the report."""
    started_at = time.monotonic()

    # None means "not specified" and takes the defaults; an EMPTY selection is
    # a different thing entirely -- the caller passed [] -- and must be
    # reported rather than silently turned back into the defaults.
    structures = tuple(DEFAULT_STRUCTURES if structures is None else structures)
    merge = tuple(DEFAULT_MERGE_MODES if merge is None else merge)
    prediction_ID = (prediction_ID or "Pred").strip() or "Pred"

    unknown = [code for code in structures if code not in STRUCTURE_CODES]
    if unknown:
        raise ToolInputError(
            f"Unknown structure code(s): {', '.join(unknown)}. "
            f"Known: {', '.join(STRUCTURE_CODES)}"
        )
    if not structures:
        raise ToolInputError("Select at least one structure to segment.")
    if not merge:
        raise ToolInputError(
            "Select at least one output form (merged and/or separated segmentations)."
        )
    # Same treatment as unknown structure codes above: an unrecognised merge
    # mode must fail HERE, before minutes of inference. Left unvalidated, it
    # matches neither branch of _assemble_scan_outputs and the run ends "ok"
    # with a report and no segmentation files at all.
    unknown_merge = [code for code in merge if code not in MERGE_MODES]
    if unknown_merge:
        raise ToolInputError(
            f"Unknown merge mode(s): {', '.join(unknown_merge)}. "
            f"Known: {', '.join(MERGE_MODES)}"
        )

    output_dir = os.fspath(output_dir)
    work_dir = os.path.join(output_dir, WORK_DIRNAME)
    os.makedirs(work_dir, exist_ok=True)

    device = nnunet_runner.resolve_device(device)

    scans = discover_scans(os.fspath(input_path), prediction_ID)
    models, missing_structures = resolve_models(os.fspath(model_path), structures)
    logger.info(
        "AMASSS: %d scan(s), %d structure(s) available, device=%s",
        len(scans), len(models), device,
    )

    try:
        report = _run(
            scans=scans,
            models=models,
            missing_structures=missing_structures,
            output_dir=output_dir,
            work_dir=work_dir,
            structures=structures,
            merge=merge,
            prediction_ID=prediction_ID,
            generate_surface=generate_surface,
            surface_smoothing=surface_smoothing,
            surface_decimation=surface_decimation,
            device=device,
            tile_step_size=tile_step_size,
            gpu_resampling=gpu_resampling,
            started_at=started_at,
        )
    finally:
        # The intermediates are large -- one predicted volume per scan and per
        # structure -- and they sit inside the directory the caller will ship,
        # so they go whether the run succeeded or not.
        shutil.rmtree(work_dir, ignore_errors=True)

    with open(os.path.join(output_dir, "AMASSS_report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    logger.info(
        "AMASSS finished: %d/%d scan(s) in %.1fs",
        report["summary"]["processed"], report["summary"]["total"],
        report["duration_seconds"],
    )
    return report


def _run(scans, models, missing_structures, output_dir, work_dir, structures, merge,
         prediction_ID, generate_surface, surface_smoothing, surface_decimation,
         device, tile_step_size, gpu_resampling, started_at) -> dict:
    """Everything between the argument checks and the report."""
    # Convert every scan once into the single folder nnUNet reads. Predicting
    # per structure over the whole folder loads each checkpoint once, instead
    # of once per (scan x structure) as the original did.
    nnunet_input = os.path.join(work_dir, "nnunet_input")
    os.makedirs(nnunet_input, exist_ok=True)

    scan_records = []
    # Three phases share the bar. Only the middle one is long, so it is given
    # most of the range; what is exact is the counter in each message, the
    # split between the phases being a weighting and nothing more.
    for index, scan_path in enumerate(scans):
        progress.report(index + 1, len(scans), "reading scan", end=0.1)
        case_id = f"p_{index:03d}"
        record = {
            "case_id": case_id,
            "input": os.path.basename(scan_path),
            "input_path": scan_path,
            "status": "pending",
            "predicted_structures": [],
            "segmentations": [],
            "surfaces": [],
            # The union of the two above, which is what a caller asking "what
            # came out for this case" wants: the typed lists stay because a
            # reader wanting only the meshes should not have to sort by
            # extension.
            "produced": [],
        }
        try:
            _convert_to_nifti(scan_path, os.path.join(nnunet_input, f"{case_id}_0000.nii.gz"))
        except Exception as exc:
            # Position in the batch, never the scan's name -- the same rule the
            # progress call above follows, and it matters more here: a failed
            # run's stderr is copied into the server's own persistent log.
            logger.exception("Could not read scan %d of %d", index + 1, len(scans))
            record["status"] = "failed"
            record["error"] = f"Unreadable input: {exc}"
        scan_records.append(record)

    readable = [record for record in scan_records if record["status"] != "failed"]
    if not readable:
        raise ToolInputError("None of the input scans could be read as a medical volume.")

    # --- Inference: one model load per structure --------------------------
    predictions = {}
    failed_structures = {}
    for structure_index, (code, model_folder) in enumerate(models.items(), start=1):
        # Per STRUCTURE, which is the honest unit here: one nnUNet call covers
        # the whole cohort, so there is no per-scan position to report inside
        # it and interpolating one would invent a number the tool cannot know.
        progress.report(structure_index, len(models), "structure", start=0.1, end=0.9)
        structure_output = os.path.join(work_dir, f"pred_{code}")
        try:
            logger.info("Predicting %s on %s", code, device)
            nnunet_runner.predict_folder(
                model_folder, nnunet_input, structure_output, device,
                tile_step_size=tile_step_size, gpu_resampling=gpu_resampling,
            )
            predictions[code] = structure_output
        except Exception as exc:
            # One structure failing must not lose the others.
            logger.exception("Prediction failed for structure %s", code)
            failed_structures[code] = str(exc)

    if not predictions:
        raise RuntimeError(
            "Every structure failed to predict: "
            + "; ".join(f"{code}: {error}" for code, error in failed_structures.items())
        )

    # --- Assemble per-scan outputs ----------------------------------------
    for index, record in enumerate(readable, start=1):
        progress.report(index, len(readable), "writing scan", start=0.9)
        scan_started = time.monotonic()
        try:
            _assemble_scan_outputs(
                record=record,
                predictions=predictions,
                output_dir=output_dir,
                work_dir=work_dir,
                prediction_ID=prediction_ID,
                merge=merge,
                generate_surface=generate_surface,
                surface_smoothing=surface_smoothing,
                surface_decimation=surface_decimation,
            )
            record["status"] = "ok"
        except Exception as exc:
            # FIX: the original re-raised on the LAST scan only, so a batch
            # could abort at the very end and lose everything already
            # produced. Every failure is recorded; the run continues.
            logger.exception(
                "Failed to assemble outputs for scan %d of %d", index, len(readable)
            )
            record["status"] = "failed"
            record["error"] = str(exc)
        record["duration_seconds"] = round(time.monotonic() - scan_started, 2)
        record.pop("input_path", None)

    processed = [record for record in scan_records if record["status"] == "ok"]
    if not processed:
        raise RuntimeError(
            "AMASSS produced no output for any scan. First error: "
            + str(next((r.get("error") for r in scan_records if r.get("error")), "unknown"))
        )

    return {
        "tool": TOOL_NAME,
        "device": device,
        # Both of these move the segmentation, so a mask is only reproducible
        # alongside them. This records what was ASKED for: a model bundle whose
        # plans pin a non-default resampler opts itself out, and says so in the
        # log.
        "gpu_resampling": bool(gpu_resampling) and device.startswith("cuda"),
        "tile_step_size": float(tile_step_size),
        "prediction_ID": prediction_ID,
        "merge_modes": list(merge),
        "generate_surface": bool(generate_surface),
        "surface_decimation": int(surface_decimation) if generate_surface else None,
        "requested_structures": list(structures),
        "predicted_structures": sorted(predictions, key=lambda c: STRUCTURE_CODES.index(c)),
        # FIX: a structure whose model was missing used to vanish with nothing
        # but a logger.warning -- invisible in a 200-scan batch. It is now
        # reported explicitly, next to the results.
        "structures_without_model": missing_structures,
        "structures_failed": failed_structures,
        "cases": scan_records,
        "summary": {
            "total": len(scan_records),
            "processed": len(processed),
            "failed": len(scan_records) - len(processed),
        },
        "duration_seconds": round(time.monotonic() - started_at, 2),
    }


def _assemble_scan_outputs(record, predictions, output_dir, work_dir, prediction_ID,
                           merge, generate_surface, surface_smoothing,
                           surface_decimation) -> None:
    """Turn one scan's per-structure nnUNet masks into its final files."""
    import numpy as np
    import SimpleITK as sitk

    scan_path = record["input_path"]
    case_id = record["case_id"]
    base, input_extension = split_scan_extension(os.path.basename(scan_path))
    # Never the input's own spelling: a scan sent as an uncompressed .nii must
    # not produce nine uncompressed masks. The masks are label volumes -- long
    # runs of a single value -- so gzip takes them roughly 100x down:
    # uncompressed, a 0.33mm CBCT gives a 191 MB file PER structure, and a
    # nine-structure run wrote 1.75 GB against 14 MB compressed.
    extension = compressed_extension(input_extension)

    reference = sitk.ReadImage(scan_path)

    masks = {}
    for code, structure_output in predictions.items():
        predicted_file = os.path.join(structure_output, f"{case_id}.nii.gz")
        if not os.path.isfile(predicted_file):
            # The case id, not the input's name: `p_003` is the scan's position
            # in the batch and is also what nnUNet read and wrote, so the line
            # is the more useful of the two AND carries no patient metadata.
            logger.warning("No %s prediction for %s", code, case_id)
            continue
        array = sitk.GetArrayFromImage(sitk.ReadImage(predicted_file))
        masks[code] = (array > 0).astype(np.uint8)

    if not masks:
        raise FileNotFoundError(f"nnUNet produced no prediction for {record['input']}")

    record["predicted_structures"] = sorted(masks, key=lambda c: STRUCTURE_CODES.index(c))

    scan_dir = os.path.join(output_dir, f"{base}_{prediction_ID}_SegOut")
    os.makedirs(scan_dir, exist_ok=True)
    surface_temp = os.path.join(work_dir, "surface_tmp")
    os.makedirs(surface_temp, exist_ok=True)

    # Separate files: requested, or unavoidable when there is only one
    # structure (a "merged" volume of one structure is just that structure).
    if "SEPARATE" in merge or len(masks) == 1:
        for code, mask in masks.items():
            output_path = os.path.join(scan_dir, f"{base}_{prediction_ID}_{code}{extension}")
            record["segmentations"].append(
                _write_segmentation(mask, reference, output_path)
            )
            if generate_surface:
                record["surfaces"].append(
                    vtk_export.write_separate_surface(
                        mask=mask,
                        reference=reference,
                        # FIX: the code is passed explicitly instead of being
                        # parsed back out of the file name (original KeyError).
                        structure_code=code,
                        label_colors=LABEL_COLORS,
                        labels=LABELS,
                        temp_dir=surface_temp,
                        smoothing=surface_smoothing,
                        decimation=surface_decimation,
                        output_path=os.path.join(
                            scan_dir, f"{base}_{prediction_ID}_{code}.vtk"
                        ),
                    )
                )

    if "MERGED" in merge and len(masks) > 1:
        shape = next(iter(masks.values())).shape
        merged = np.zeros(shape, dtype=np.int16)
        for code in MERGING_ORDER:
            if code in masks:
                merged = np.where(masks[code] == 1, LABELS[code], merged)

        output_path = os.path.join(scan_dir, f"{base}_{prediction_ID}_MERGED{extension}")
        record["segmentations"].append(_write_segmentation(merged, reference, output_path))
        if generate_surface:
            surface = vtk_export.write_merged_surface(
                merged=merged,
                reference=reference,
                names_from_labels=NAMES_FROM_LABELS,
                label_colors=LABEL_COLORS,
                temp_dir=surface_temp,
                smoothing=surface_smoothing,
                decimation=surface_decimation,
                output_path=os.path.join(scan_dir, f"{base}_{prediction_ID}_MERGED.vtk"),
            )
            if surface:
                record["surfaces"].append(surface)

    # Unreachable while the merge codes are validated in segment(): SEPARATE
    # writes per structure, MERGED writes above, and a lone mask falls back to
    # the separate branch. Kept anyway -- the one time this happened, the run
    # was reported "ok" and the client received an archive holding nothing but
    # the report. A scan with no output must never count as processed.
    record["produced"] = record["segmentations"] + record["surfaces"]
    if not record["segmentations"]:
        raise RuntimeError(
            f"No segmentation was written for {record['input']} (merge modes: {', '.join(merge)})."
        )
