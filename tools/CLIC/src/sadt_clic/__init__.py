"""Segment the impacted canine on a CBCT scan."""

import json
import logging
import os
import time
from pathlib import Path
from typing import Literal

from . import catalogs, progress
from .pipeline import (
    DEFAULT_SCORE_THRESHOLD,
    build_model,
    discover_scans,
    resolve_device,
    segment_volume,
)

logger = logging.getLogger("CLIC")

__all__ = ["run"]

# The DATA folder this tool's weights live in: its own name, since CLIC is one
# tool behind one name. Written rather than derived, like ALI's, because which
# folder holds which bundle is a deployment fact and a wrong guess is a
# directory that is simply not there.
_DATA_NAME = "CLIC"


def _own_models(data_root) -> Path:
    """`<root>/CLIC/models`, or a refusal a caller can act on."""
    if data_root is None:
        raise ValueError(
            "No 'model' given and no data root to look in. Name the checkpoint, "
            "or run this through a server that publishes one."
        )
    return Path(data_root) / _DATA_NAME / "models"


def _sole_checkpoint(folder: Path) -> Path:
    """The one `.pth` in `folder`, refusing both other counts.

    Upstream took `sorted(folder.glob("*.pth"))[0]` -- the alphabetically first
    file, with no check that it was the only one -- so which model vintage ran
    depended on file names. Refusing to guess is the same rule this tool already
    applies to the argument, applied to the folder behind it.
    """
    found = sorted(folder.glob("*.pth"))
    if not found:
        raise ValueError(
            f"No checkpoint in '{folder.name}'. Fetch one with "
            f"`scripts/setup-models.sh --tool CLIC`, or name one."
        )
    if len(found) > 1:
        raise ValueError(
            "Several checkpoints are installed and none was named, so which "
            "weights would run is a guess: "
            + ", ".join(path.name for path in found)
            + ". Name the one you want."
        )
    return found[0]


def _resolve_checkpoint(model, data_root) -> str:
    """The checkpoint to load: the one named, or this tool's own.

    A caller that names a bundle is pinning which weights ran and is obeyed. One
    that names none -- a panel left on "(automatic)", a neighbour asking through
    the supervisor -- gets the checkpoint in this tool's own data folder, and
    only when there is exactly one of them.
    """
    path = Path(model) if model else _own_models(data_root)
    return str(_sole_checkpoint(path) if path.is_dir() else path)


def run(
    scans: Path,
    output_dir: Path,
    # After `output_dir` and optional, which is what lets a panel show
    # "(automatic)" and a supervised call omit it. There is one published
    # checkpoint, so naming it is a step that can only be got wrong.
    model: Path = "",
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    output_suffix: str = "seg",
    device: Literal["cuda", "cpu"] = "cuda",
    *,
    data_root=None,
) -> Path:
    """Segment the impacted canine on a CBCT scan, and say where it sits.

    Args:
        scans: One CBCT (.nii/.nii.gz), or a folder of them for a batch.
            Folders are searched recursively.
        output_dir: Where the segmentations are written. Nothing is written
            outside it.
        model: The Mask R-CNN checkpoint (.pth). Left empty, this tool uses the
            one installed for it, refusing rather than choosing when several
            are: upstream took the alphabetically first file in a folder, so
            which model vintage ran depended on file names.
        score_threshold: Detections scoring below this are not painted. The one
            knob that moves the segmentation, so it is recorded in the report.
        output_suffix: Appended to each scan's name, `<scan>_<suffix>.nii.gz`.
        device: "cuda" or "cpu". CUDA falls back to CPU when no card is visible.

    Returns:
        The output directory, holding one segmentation per scan and
        `CLIC_report.json`. That report carries the label table: the classes
        this network emits ARE the finding -- buccal, bicortical or palatal --
        and an integer nobody can name is not a result.
    """
    started = time.monotonic()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    found = discover_scans(str(scans))
    if not found:
        # A batch that could read nothing is a request problem, not a result:
        # the server maps ValueError to 422 with this message, rather than
        # returning a successful run of zero scans.
        raise ValueError(
            f"No .nii or .nii.gz scan found in '{os.path.basename(str(scans))}'. "
            f"CLIC reads NIfTI volumes; NRRD and MetaImage are not supported."
        )

    checkpoint = _resolve_checkpoint(model, data_root)
    device = resolve_device(device)
    network, classes = build_model(checkpoint, device)
    logger.info(
        "CLIC: %d scan(s), %d class(es), device=%s, score_threshold=%.2f",
        len(found), classes, device, score_threshold,
    )

    report = {
        "tool": "CLIC",
        "model": os.path.basename(checkpoint),
        "classes": classes,
        # Published with the results, like BatchDentalSeg's: the segmentation is
        # a volume of integers, and here those integers are the classification
        # this tool exists to produce.
        "labels": catalogs.labels_for(classes),
        "label_colors": catalogs.colors_for(classes),
        "device": device,
        "score_threshold": score_threshold,
        "cases": {},
    }
    if report["labels"] is None:
        # Named wrongly is worse than not named: a class count these names do
        # not describe still produces a plausible volume, with the surgical
        # approach it implies attached to the wrong anatomy.
        report["labels_note"] = (
            f"This checkpoint has {classes} classes; CLIC's names describe "
            f"{catalogs.CLASSES}. The values are published unnamed rather than "
            f"named wrongly."
        )
    written = []

    # Scans are keyed by their path RELATIVE to the input root, and the output
    # mirrors that tree. Keying on the base name alone made two scans called
    # `scan.nii.gz` in different folders write the same file: one patient's
    # segmentation silently replaced another's, and the report said both had
    # been segmented. A single file is its own root, so it lands directly in
    # `output_dir` under its own name.
    root = Path(scans)
    if root.is_file():
        root = root.parent

    for index, path in enumerate(found, start=1):
        # Position in the batch, never the scan's name: a file name is patient
        # metadata and a progress message is stored and shown.
        progress.report(index, len(found), "scan")
        relative = Path(os.path.relpath(path, root))
        entry = {"input": relative.as_posix()}
        try:
            written.append(
                _segment_one(network, path, output_dir, relative, device,
                             score_threshold, output_suffix, classes, entry)
            )
        except Exception as exc:
            # One unreadable scan must not cost the other 199. Upstream ran one
            # process per scan, so it never had to say this. Position here too,
            # and for a sharper reason than the progress call above: a failed
            # run's stderr is copied into the server's own persistent log.
            logger.exception("CLIC failed on scan %d of %d", index, len(found))
            entry["status"] = "failed"
            entry["reason"] = f"{type(exc).__name__}: {exc}"
        report["cases"][entry["input"]] = entry

    report["summary"] = f"{len(written)}/{len(found)} scan(s) segmented"
    report["duration_seconds"] = round(time.monotonic() - started, 2)

    if not written:
        # The guard counts what was WRITTEN, not what was walked past.
        raise ValueError(
            "CLIC segmented none of the scans it was given. "
            + "; ".join(
                f"{s['input']}: {s.get('reason', 'unknown')}"
                for s in report["cases"].values() if s.get("status") == "failed"
            )
        )

    (output_dir / "CLIC_report.json").write_text(json.dumps(report, indent=2))
    return output_dir


def _segment_one(network, path, output_dir, relative, device, score_threshold,
                 suffix, classes, entry):
    """One scan, in and out. Returns the path written.

    `relative` is the scan's path relative to the input root; the output is
    written at the same place under `output_dir`, so a batch of homonyms in
    different folders keeps them apart.
    """
    import nibabel as nib
    import numpy as np

    image = nib.load(path)
    volume = image.get_fdata(dtype=np.float32)
    if volume.ndim != 3:
        raise ValueError(f"expected a 3D volume, got {volume.ndim}D")

    labels, detections = segment_volume(network, volume, device, score_threshold)

    stem = relative.name
    for extension in (".nii.gz", ".nii"):
        if stem.lower().endswith(extension):
            stem = stem[: -len(extension)]
            break

    destination = output_dir / relative.parent / f"{stem}_{suffix}.nii.gz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(labels, image.affine, image.header), str(destination))

    entry["status"] = "ok"
    entry["slices"] = int(volume.shape[2])
    entry["detections"] = detections
    entry["labels_present"] = sorted(int(v) for v in np.unique(labels) if v)
    # The same values, named. What a reader of this report actually wants to
    # know is "palatal", not "3".
    entry["detected"] = catalogs.names_for(entry["labels_present"], classes)
    # The path under `output_dir`, not the bare file name. `destination` now
    # carries the patient's sub-folder, so `.name` would throw it away and two
    # homonyms would report the identical output -- the very defect the tree
    # mirroring was added to fix.
    entry["produced"] = [destination.relative_to(output_dir).as_posix()]
    if not detections:
        # Reported rather than swallowed: a volume the network found nothing in
        # is a legitimate answer AND the signature of a wrong checkpoint, and
        # only the caller can tell them apart.
        entry["note"] = "no detection cleared the score threshold"
    return str(destination)
