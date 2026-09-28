"""Apply a transform to scans, segmentations and landmark files."""

import json
import logging
import os
import time
from pathlib import Path
from typing import Literal

from . import progress
from .pipeline import (
    IMAGE_EXTENSIONS,
    apply_to_landmarks,
    patient_of,
    is_image_file,
    is_landmark_file,
    is_transform_file,
    legacy_file_key,
    looks_like_a_label_map,
    legacy_transform_key,
    read_transform,
    resample,
)

logger = logging.getLogger("AutoMatrix")

__all__ = ["run"]


def run(
    files: Path,
    transforms: Path,
    output_dir: Path,
    same_transform_for_every_patient: bool = False,
    name_output_after_transform: bool = False,
    output_suffix: str = "Reg",
    content: Literal["Automatic", "Scan", "Segmentation"] = "Automatic",
    reference: Path = "",
) -> Path:
    """Apply each patient's transform to their scans, segmentations or landmarks.

    Args:
        files: A scan, a segmentation, a landmark file, or a folder of them.
            Folders are searched recursively.
        transforms: The transforms to apply (.tfm/.mat/.h5/.txt), matched to
            files by patient name. A patient with several transforms has each
            of them applied.
        output_dir: Where the results are written. The input tree is mirrored.
        same_transform_for_every_patient: Apply the one transform given to every
            patient, instead of matching it to a patient by name. What a mirror
            matrix needs, and meaningless with more than one transform.
        name_output_after_transform: Add the transform's name to each output,
            which is what tells two transforms of one patient apart. A patient
            who has several gets it whatever this says, since otherwise each
            output overwrites the last.
        output_suffix: Appended to each output name.
        content: How the voxels are resampled. "Segmentation" uses nearest
            neighbour, so no label is invented; "Scan" interpolates linearly;
            "Automatic" reads it off each file, which is per FILE rather than
            per run and is what a folder holding both needs.
        reference: Optional volume defining the output grid. Without it each
            image keeps its own grid and only its origin moves.

    Returns:
        The output directory, holding the transformed files and
        `AutoMatrix_report.json`.
    """
    started = time.monotonic()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subjects = _discover(str(files))
    if not subjects:
        raise ValueError(
            f"No scan, segmentation or landmark file found in "
            f"'{os.path.basename(str(files))}'."
        )

    by_patient = _discover_transforms(str(transforms))
    if not by_patient:
        raise ValueError(
            f"No transform found in '{os.path.basename(str(transforms))}'. "
            f"Expected one of {', '.join(('.tfm', '.mat', '.h5', '.txt'))}."
        )

    # Everything the legacy module could pair and this port cannot, retried --
    # never a replacement for the rule above, only a second chance for what it
    # left over. See `_pair_like_legacy`.
    paired_by = _pair_like_legacy(
        subjects, by_patient, same_transform_for_every_patient
    )

    reference_image = None
    if reference:
        import SimpleITK as sitk

        reference_image = sitk.ReadImage(str(reference))

    report = {
        "tool": "AutoMatrix",
        "content": content,
        "reference": os.path.basename(str(reference)) if reference else None,
        "output_suffix": output_suffix,
        "cases": {},
        "without_a_transform": sorted(set(subjects) - set(by_patient)),
        "transforms_without_a_file": sorted(set(by_patient) - set(subjects)),
    }
    # Omitted entirely when every pair came from this port's own rule, which is
    # the normal case: a key present here is a key someone may need to explain.
    if paired_by:
        report["paired_by"] = dict(sorted(paired_by.items()))

    written = []
    for index, patient in enumerate(sorted(subjects), start=1):
        # The counter, never the patient key: the key is derived from the
        # caller's file names, and a progress message is stored and shown.
        progress.report(index, len(subjects), "patient")
        matrices = by_patient.get(patient)
        if not matrices:
            # Named rather than skipped in silence: upstream dropped a file
            # whose patient had no transform without a word, so a run could
            # transform 3 of 40 and look complete.
            continue
        # `outputs` keeps its per-file detail -- how many points each move
        # touched -- and `produced` is the flat list of names beside it, which
        # is what every tool of this catalogue now answers.
        entry = {"transforms": [os.path.basename(m) for m in matrices],
                 "outputs": [], "produced": []}
        subject_files = subjects[patient]
        for file_index, path in enumerate(subject_files, start=1):
            for matrix in matrices:
                try:
                    written.append(_apply_one(
                        path, matrix, files, output_dir, reference_image,
                        content,
                        # Several transforms of one patient force the transform
                        # name on whatever the caller asked for: with the flag
                        # off every one of them resolved to the SAME output
                        # name, so each overwrote the last while the report
                        # counted them all as written.
                        name_output_after_transform or len(matrices) > 1,
                        output_suffix, entry,
                    ))
                except Exception as exc:
                    # Both counters, never the file's name: this is a failure
                    # path, and a failed run's stderr is copied into the
                    # server's own persistent log. The report below still names
                    # the file -- it goes back to whoever sent it.
                    logger.exception(
                        "AutoMatrix failed on patient %d of %d, file %d of %d",
                        index, len(subjects), file_index, len(subject_files),
                    )
                    entry.setdefault("failed", []).append(
                        f"{os.path.basename(path)}: {type(exc).__name__}: {exc}"
                    )
        report["cases"][patient] = entry

    report["summary"] = {
        "cases": len(report["cases"]),
        "files_written": len(written),
    }
    report["duration_seconds"] = round(time.monotonic() - started, 2)

    if not written:
        # The reason a caller needs is WHY nothing was written, and the two
        # cases read differently: nothing paired, or everything that paired
        # then failed. Reporting only the counts hid a transform SimpleITK
        # could not read behind "0 file(s) had no transform".
        failures = [
            message
            for detail in report["cases"].values()
            for message in detail.get("failed", [])
        ]
        raise ValueError(
            "AutoMatrix transformed nothing. "
            + (
                "; ".join(failures[:5])
                if failures
                else f"{len(report['without_a_transform'])} file(s) had no transform, "
                     f"{len(report['transforms_without_a_file'])} transform(s) had no file."
            )
        )

    (output_dir / "AutoMatrix_report.json").write_text(json.dumps(report, indent=2))
    return output_dir


def _pair_like_legacy(subjects: dict, by_patient: dict, share_one: bool) -> dict:
    """Pair what this port's own rule left over, the way the legacy module did.

    Two behaviours of SlicerAutomatedDentalTools that `patient_of` does not
    reproduce, and without which all four of the datasets published with that
    module transform nothing:

    1. its substring cut of a file name (`pipeline.legacy_*_key`);
    2. a single transform applied to every patient, with no pairing at all --
       the mirror matrix is used exactly that way, and VFACE drives it eight
       times per run.

    Both are reached ONLY by a patient this port could not pair, and only a
    transform this port did not already give to someone else is offered. A run
    that pairs today therefore keeps its pairs, its outputs and its bytes; this
    can turn a failure into a result and nothing else.

    `by_patient` is extended in place. Returns {patient: which rule paired it},
    for the report -- a patient paired by the normal rule is absent from it,
    that being the default.
    """
    rules: dict = {}
    every_transform = {p for paths in by_patient.values() for p in paths}

    # Asked for outright, so it is not a fallback and does not wait for the
    # normal rule to fail: upstream's own semantics, where one transform file is
    # applied to everyone whatever the names say.
    if share_one and len(every_transform) == 1:
        only = next(iter(every_transform))
        for key in subjects:
            by_patient[key] = [only]
            rules[key] = "asked to share one transform"
        return rules

    unpaired = set(subjects) - set(by_patient)
    if not unpaired:
        return rules

    # 1. Upstream's substring cut, on both sides. Restricted to transforms that
    #    went unpaired, so it can never take one from a patient already matched.
    spare = {key: paths for key, paths in by_patient.items() if key not in subjects}
    if spare:
        wanted: dict = {}
        for key in unpaired:
            for path in subjects[key]:
                legacy = legacy_file_key(os.path.basename(path))
                if legacy:
                    wanted.setdefault(legacy, set()).add(key)
        offered: dict = {}
        for paths in spare.values():
            for path in paths:
                legacy = legacy_transform_key(os.path.basename(path))
                if legacy:
                    offered.setdefault(legacy, []).append(path)
        for legacy, keys in wanted.items():
            matrices = offered.get(legacy)
            if not matrices:
                continue
            for key in keys:
                by_patient[key] = sorted(matrices)
                rules[key] = "legacy file names"
        unpaired -= set(rules)

    # 2. One transform and ONE patient: there is nothing else it could belong
    #    to, so pairing it costs no guess. Upstream is looser -- a single
    #    transform file goes to everyone there, however many patients -- and
    #    that difference is deliberate. Across a COHORT the guess is how one
    #    patient's matrix silently lands on everybody, a failure that looks
    #    exactly like a success, and the legacy module's own callers were bitten
    #    by it. A caller who means it says so with `same_transform_for_every_patient`.
    if unpaired == set(subjects) and len(every_transform) == 1 and len(subjects) == 1:
        only = next(iter(every_transform))
        for key in subjects:
            by_patient[key] = [only]
            rules[key] = "the only transform, applied to every patient"

    return rules


def _walk(root: str, accept) -> dict:
    """{patient: [paths]} for every file under `root` that `accept` keeps.

    A single file is answered as itself, a folder is walked recursively, and
    both spellings key on `patient_of` -- which is what lets a transform be
    matched to the scans it applies to whichever way the caller pointed at it.
    """
    found: dict = {}
    if os.path.isfile(root):
        name = os.path.basename(root)
        if accept(name):
            found.setdefault(patient_of(name), []).append(root)
        return found

    for directory, _subdirs, names in os.walk(root):
        for name in sorted(names):
            if name.startswith(".") or not accept(name):
                continue
            found.setdefault(patient_of(name), []).append(
                os.path.join(directory, name)
            )
    return found


def _discover(root: str) -> dict:
    """{patient: [paths]} for everything transformable under `root`."""
    return _walk(root, lambda name: is_image_file(name) or is_landmark_file(name))


def _discover_transforms(root: str) -> dict:
    """{patient: [transform paths]}, keyed by the same rule the files are."""
    return _walk(root, is_transform_file)


def _with_tail(stem: str, tail: str) -> str:
    """`stem_tail`, or `stem` when there is no tail to add."""
    return f"{stem}_{tail}" if tail else stem


def _apply_one(path, matrix, input_root, output_dir, reference, content,
               name_after_transform, suffix, entry) -> str:
    """One file through one transform. Returns the path written."""
    import SimpleITK as sitk

    transform = read_transform(matrix)
    name = os.path.basename(path)

    # Joined from the parts that exist, so an empty `output_suffix` gives
    # `P1_T1.nii.gz` rather than `P1_T1_.nii.gz` -- and, with the transform
    # named too, `P1_T1__P1_CBReg.nii.gz`.
    tail = "_".join(
        part
        for part in (suffix, Path(matrix).stem if name_after_transform else "")
        if part
    )

    relative = os.path.relpath(path, str(input_root)) if os.path.isdir(str(input_root)) else name
    destination = output_dir / relative
    destination.parent.mkdir(parents=True, exist_ok=True)

    if is_landmark_file(name):
        stem = name[: -len(".mrk.json")]
        destination = destination.parent / f"{_with_tail(stem, tail)}.mrk.json"
        moved = apply_to_landmarks(path, transform, str(destination))
        entry["outputs"].append({"file": destination.name, "points_moved": moved})
        entry["produced"].append(destination.name)
        return str(destination)

    for extension in IMAGE_EXTENSIONS:
        if name.lower().endswith(extension):
            stem, tail_extension = name[: -len(extension)], extension
            break
    else:
        stem, tail_extension = os.path.splitext(name)

    destination = destination.parent / f"{_with_tail(stem, tail)}{tail_extension}"
    image = sitk.ReadImage(path)

    # A caller that named the content is believed; "Automatic" is read off the
    # file. Recorded in the report either way, because which interpolator ran is
    # not visible in the result and is the difference between a label map that
    # survived and one that grew labels nobody segmented.
    written_entry = {"file": destination.name}
    if content == "Automatic":
        is_segmentation = looks_like_a_label_map(image)
        written_entry["detected"] = "segmentation" if is_segmentation else "scan"
    else:
        is_segmentation = content == "Segmentation"

    sitk.WriteImage(resample(image, transform, reference, is_segmentation), str(destination))
    entry["outputs"].append(written_entry)
    return str(destination)
