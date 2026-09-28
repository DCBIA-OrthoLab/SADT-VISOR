"""Register two CBCT timepoints with Greedy."""

import json
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Literal

from sadt_areg_common import catalogs, pairing

from . import progress
from .pipeline import (
    binarise_mask,
    check_choices,
    registration_command,
    resample_command,
    run_greedy,
    write_identity_init,
)

logger = logging.getLogger("GreedyReg")

__all__ = ["run"]

# Every token a mask name may carry on top of the patient's own, so
# `P1_T1_MAND_seg.nii.gz` keys to the `P1` its scan keys to. The same set
# `pairing.discover_masks` builds, and applied under the same guard: only to a
# file that says it IS a mask, so a patient legitimately called `MD_01` is not
# read as patient `01`.
_MASK_TOKENS = {
    token for group in catalogs.REGION_TOKENS.values() for token in group
} | set(catalogs.MASK_TOKENS)


def run(
    t1: Path,
    t2: Path,
    output_dir: Path,
    # Straight after the two folders they belong with: sections are published
    # in the order the signature first mentions them, so declaring the two
    # optional INPUTS here puts the collapsed "Advanced" bar under T1 and T2
    # rather than at the foot of the panel, three sections away from the things
    # it holds more of.
    masks: Path = "",
    initial_transforms: Path = "",
    metric: Literal["NCC", "NMI", "SSD"] = "NCC",
    transform_type: Literal["Rigid", "Affine"] = "Rigid",
    output_suffix: str = "registered",
) -> Path:
    """Register each patient's second CBCT onto their first, with Greedy.

    Args:
        t1: Folder of first-timepoint scans -- the FIXED images, the frame
            everything is brought into.
        t2: Folder of second-timepoint scans -- the MOVING images, the ones
            that get resampled. A patient is paired with their T1 by name, up
            to the timepoint token: `P1_T1_scan.nii.gz` pairs with
            `P1_T2.nii.gz`. Anyone present in only one folder is reported
            rather than dropped.
        output_dir: Where the registered volumes and transforms are written.
        masks: One mask per patient, and the metric is then computed inside it
            only. This is how a growing patient is registered on what has NOT
            changed -- the cranial base, typically, which AMASSS segments. The
            mask belongs to the T1 scan, since T1 is the fixed image.
        metric: What the registration optimises. NCC compares local patterns in
            a 4x4x4 window and is the one to keep for two CBCTs of one patient.
            NMI compares the two intensity distributions instead, which is what
            to reach for when the scans were acquired differently enough that
            their grey values no longer correspond. SSD compares voxel values
            directly and needs them on the same scale.
        transform_type: Rigid moves and rotates and nothing else, 6 degrees of
            freedom. Affine adds scaling and shear, 12. For two timepoints of
            one patient rigid is almost always what is wanted: affine can
            absorb real growth into the transform, and that growth is the thing
            a longitudinal study is measuring.
        output_suffix: Appended to each patient's name in the output.
        initial_transforms: A `.mat` transform per patient to start the search
            from, for two timepoints too far apart for the search to find each
            other on its own. Without one the search starts from identity.

    Returns:
        The output directory, holding `<patient>_<suffix>.nii.gz`,
        `<patient>_transform.mat` and `GreedyReg_report.json`.
    """
    started = time.monotonic()
    # Before a folder is walked or a directory created: a metric this tool does
    # not have is a bad request, and answering it with forty identical
    # per-patient failures would hide that.
    check_choices(metric, transform_type)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Paired by the SHARED rule, not by a regex of this tool's own. Upstream
    # read the patient as `^([A-Za-z]+\d+)` -- letters then digits, with
    # nothing between -- which silently skipped every name this repository's
    # own data uses: `C_0001_T1_Or.nii.gz`, `MAMP_0002_T1.nii.gz`,
    # `IC_0005.nii.gz`. A file that matched nothing was not reported, it was
    # dropped from the dict.
    matched = pairing.pair(str(t1), str(t2), output_suffix)
    report = {
        "tool": "GreedyReg",
        "metric": metric,
        "transform_type": transform_type,
        "output_suffix": output_suffix,
        "unmatched": matched.unmatched_report(),
        "cases": {},
    }

    if not matched:
        # The counts alone say the run failed; the NAMES say why, and they are
        # the half a caller can act on -- almost always one folder using a
        # decoration the other does not.
        raise ValueError(
            "No patient appears in both the T1 and the T2 folder. They are paired "
            "by name, up to the timepoint token -- so 'P1_T1_scan.nii.gz' pairs "
            f"with 'P1_T2.nii.gz'. Found {len(matched.t1_only)} T1-only "
            f"({_listed(matched.t1_only)}) and {len(matched.t2_only)} T2-only "
            f"({_listed(matched.t2_only)}) patient(s)."
        )

    mask_by_patient = _discover_masks(str(masks)) if masks else {}
    init_by_patient = _discover_transforms(str(initial_transforms)) if initial_transforms else {}

    # A mask or an initial transform that matched no patient is REPORTED, not
    # dropped. It is the same silence this port exists to remove: a run that
    # quietly registered without the mask it was handed looks, from the
    # outside, exactly like one that used it.
    report["unused_masks"] = sorted(set(mask_by_patient) - set(matched.matched))
    report["unused_initial_transforms"] = sorted(set(init_by_patient) - set(matched.matched))
    for kind in ("unused_masks", "unused_initial_transforms"):
        if report[kind]:
            logger.warning("GreedyReg: %d %s matched no patient", len(report[kind]), kind)

    registered = 0
    for index, (patient, files) in enumerate(matched.matched.items(), start=1):
        # The counter, never the patient key: the key comes from the caller's
        # file names, and a progress message is stored and shown.
        progress.report(index, len(matched.matched), "patient")
        fixed, moving = files["t1"], files["t2"]
        entry = {"t1": os.path.basename(fixed), "t2": os.path.basename(moving)}
        scratch = None
        try:
            # Inside the guard, and with the separators of a nested patient key
            # flattened: `mkdtemp(prefix="greedyreg_sub/A1_")` raises, and it
            # raised OUTSIDE this try -- so one patient in a subfolder took the
            # whole batch down, which is the failure this port exists to end.
            scratch = tempfile.mkdtemp(prefix=f"greedyreg_{patient.replace(os.sep, '_')}_")
            _register_one(
                patient, fixed, moving, output_dir, scratch,
                mask_by_patient.get(patient), init_by_patient.get(patient),
                metric, transform_type, output_suffix, entry,
            )
            registered += 1
        except Exception as exc:
            # Upstream called sys.exit(1) here, so patient 3 failing lost
            # patients 4 to 40 -- and the batch reported nothing about any of
            # them. One patient failing costs one patient. The counter again,
            # and for a sharper reason than the progress call above: a failed
            # run's stderr is copied into the server's own persistent log.
            logger.exception(
                "GreedyReg failed on patient %d of %d", index, len(matched.matched)
            )
            entry["status"] = "failed"
            entry["reason"] = f"{type(exc).__name__}: {exc}"
        finally:
            if scratch:
                shutil.rmtree(scratch, ignore_errors=True)
        report["cases"][patient] = entry

    report["summary"] = {
        "cases": len(report["cases"]),
        "registered": registered,
        "failed": len(report["cases"]) - registered,
    }
    report["duration_seconds"] = round(time.monotonic() - started, 2)

    if not registered:
        raise ValueError(
            "GreedyReg registered none of the pairs it was given. "
            + "; ".join(
                f"{name}: {detail.get('reason', 'unknown')}"
                for name, detail in report["cases"].items()
            )
        )

    (output_dir / "GreedyReg_report.json").write_text(json.dumps(report, indent=2))
    return output_dir


def _listed(keys: list, limit: int = 10) -> str:
    """The first few patient keys, as a sentence fragment."""
    if not keys:
        return "none"
    shown = ", ".join(keys[:limit])
    return shown if len(keys) <= limit else f"{shown}, ..."


def _relative_prefix(root: str, directory: str) -> str:
    """The directory part of a patient key, as `pairing.discover` builds it."""
    relative = os.path.relpath(directory, root)
    return "" if relative == "." else relative


def _discover_masks(root: str) -> dict:
    """{patient key: mask path}, keyed as the scans are.

    `pairing.discover` was used here at first, and it keys on a bare
    `patient_stem`: `A1_mask.nii.gz` became patient `A1_mask` and
    `A1_T1_MAND_seg.nii.gz` became `A1_MAND`, so neither matched the `A1` its
    scan keys to. Every mask AMASSS writes is named that way, which made the
    argument inert -- the run went ahead unmasked and said nothing.
    """
    found: dict = {}
    for directory, _subdirs, names in os.walk(root):
        prefix = _relative_prefix(root, directory)
        for name in sorted(names):
            if name.startswith(".") or not pairing.is_scan_file(name):
                continue
            stem, _extension = pairing.split_scan_extension(name)
            # Only a file that says it IS a mask is allowed to say which
            # structure it covers; see _MASK_TOKENS.
            drop = _MASK_TOKENS if pairing.has_token(stem, catalogs.MASK_TOKENS) else ()
            key = os.path.join(prefix, pairing.patient_stem(name, also_drop=drop))
            found.setdefault(key, os.path.join(directory, name))
    return found


def _discover_transforms(root: str) -> dict:
    """`.mat` files, keyed by the same patient rule the scans are.

    `_transform` is dropped along with the timepoint token, because that is
    what THIS tool names its own transforms: without it `A1_transform.mat` keyed
    to patient `A1_transform`, so feeding one run's transforms back in as
    `initial_transforms` -- the obvious use -- matched nothing and silently
    restarted every patient from identity.
    """
    found: dict = {}
    for directory, _subdirs, names in os.walk(root):
        prefix = _relative_prefix(root, directory)
        for name in sorted(names):
            if not name.lower().endswith(".mat") or name.startswith("."):
                continue
            key = os.path.join(prefix, pairing.patient_stem(name, also_drop=("transform",)))
            found.setdefault(key, os.path.join(directory, name))
    return found


def _register_one(patient, fixed, moving, output_dir, scratch, mask, init,
                  metric, transform_type, suffix, entry) -> None:
    """One pair: affine search, then resample the moving image into the fixed."""
    registered_path = output_dir / f"{patient}_{suffix}.nii.gz"
    transform_path = output_dir / f"{patient}_transform.mat"
    # A patient key carries the directory it was found in, so the output
    # mirrors the input tree -- and greedy will not create that directory. It
    # wrote nothing at all until this line existed.
    registered_path.parent.mkdir(parents=True, exist_ok=True)

    if init:
        entry["initial_transform"] = os.path.basename(init)
    else:
        init = os.path.join(scratch, "init.mat")
        write_identity_init(init)

    if mask:
        entry["mask"] = os.path.basename(mask)
        binarised = os.path.join(scratch, "mask.nii.gz")
        binarise_mask(mask, binarised)
        mask = binarised

    run_greedy(registration_command(
        fixed, moving, str(transform_path), init, metric, transform_type, mask or "",
    ))
    run_greedy(resample_command(fixed, moving, str(registered_path), str(transform_path)))

    entry["status"] = "ok"
    entry["transform_maps"] = "the T2 image -> the T1 frame (what greedy -r consumes)"
    # Relative to the output directory, not just the base name: a nested
    # patient's two files sit in a subfolder and a caller has to be able to
    # find them.
    entry["produced"] = [
        str(registered_path.relative_to(output_dir)),
        str(transform_path.relative_to(output_dir)),
    ]
