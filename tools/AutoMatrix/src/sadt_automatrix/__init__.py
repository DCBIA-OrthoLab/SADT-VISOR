"""Apply a transform to scans, segmentations and landmark files."""

import json
import logging
import os
import re
import time
from collections import Counter
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
    itk_cause,
    legacy_file_key,
    looks_like_a_label_map,
    legacy_transform_key,
    read_transform,
    resample,
)

# Named after the module, not the tool: the server shows the records of the
# loggers under the tool's own package, and "AutoMatrix" is not one of them.
logger = logging.getLogger(__name__)

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

    patients = sorted(subjects)
    total = len(patients)
    logger.info(
        "AutoMatrix: %d patient(s), %d file(s), %d transform(s); content %s, %s",
        total, sum(len(paths) for paths in subjects.values()),
        len({p for paths in by_patient.values() for p in paths}), content,
        "onto a reference grid" if reference else "each image on its own grid",
    )

    # Counted and placed up front, rather than one line per patient inside the
    # loop: upstream dropped a file whose patient had no transform without a
    # word, so a run could transform 3 of 40 and look complete -- and forty
    # lines saying the same thing would bury the one that differs.
    skipped = [i for i, patient in enumerate(patients, start=1)
               if not by_patient.get(patient)]
    if skipped:
        logger.warning(
            "%d of %d patients have no transform and are skipped "
            "(%d file(s), positions %s); %d transform(s) matched no file",
            len(skipped), total,
            sum(len(subjects[patients[i - 1]]) for i in skipped),
            _positions(skipped),
            len(set(by_patient) - set(subjects)),
        )

    reference_image = None
    if reference:
        import SimpleITK as sitk

        progress.emit(None, "reading the reference volume")
        try:
            reference_image = sitk.ReadImage(str(reference))
        except (RuntimeError, OSError) as exc:
            # The argument named, not only a path: the server redacts the path
            # out of the reason, and what is left must still say which input
            # the caller has to fix. A ValueError, because that input is theirs.
            raise ValueError(
                f"the 'reference' volume could not be read: {itk_cause(exc)}"
            ) from exc

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
    # (exception, the message with the caller's names taken out), one per
    # file and transform that failed: the first is what a refusal chains from,
    # and the cleaned messages are what the most common failure is counted on.
    failures = []
    attempted = 0
    for index, patient in enumerate(patients, start=1):
        # The counter, never the patient key: the key is derived from the
        # caller's file names, and a progress message is stored and shown.
        progress.report(index, total, "patient")
        matrices = by_patient.get(patient)
        if not matrices:
            # Already counted, with its position, in the warning above.
            continue
        # `outputs` keeps its per-file detail -- how many points each move
        # touched -- and `produced` is the flat list of names beside it, which
        # is what every tool of this catalogue now answers.
        entry = {"transforms": [os.path.basename(m) for m in matrices],
                 "outputs": [], "produced": []}
        subject_files = subjects[patient]
        for file_index, path in enumerate(subject_files, start=1):
            where = f"patient {index} of {total}, file {file_index} of {len(subject_files)}"
            for matrix in matrices:
                attempted += 1
                step = {"name": "reading the transform"}

                def at(name, where=where, step=step, fraction=(index - 1) / total):
                    # The last progress message is what the server's diagnosis
                    # quotes as the stage a failed run was in, so each risky
                    # step announces itself -- by position, never by name.
                    step["name"] = name
                    progress.emit(fraction, f"{where}: {name}")

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
                        output_suffix, entry, at,
                    ))
                except Exception as exc:
                    # Positions and the exception, never the file's name: the
                    # message is cleaned of the paths and the patient key this
                    # step knows. The report below still names the file -- it
                    # goes back to whoever sent it.
                    cleaned = _without_names(itk_cause(exc), path, matrix, patient)
                    failures.append((exc, cleaned))
                    logger.warning(
                        "%s: %s failed (%s: %s)",
                        where, step["name"], type(exc).__name__, cleaned,
                    )
                    entry.setdefault("failed", []).append(
                        f"{os.path.basename(path)}: {type(exc).__name__}: {exc}"
                    )
                    continue
                last = entry["outputs"][-1]
                if last.get("points_moved") == 0:
                    # Written, and identical to its input: no control point
                    # was defined with three coordinates. Said here because the
                    # output otherwise reads as a moved landmark file.
                    logger.warning(
                        "%s: the landmark file has no point to move "
                        "(none defined with 3 coordinates); written unchanged",
                        where,
                    )
        report["cases"][patient] = entry

    report["summary"] = {
        "cases": len(report["cases"]),
        "files_written": len(written),
    }
    report["duration_seconds"] = round(time.monotonic() - started, 2)

    if not written:
        _refuse_nothing_transformed(failures, attempted, total, len(skipped), report)

    logger.log(
        logging.WARNING if failures or skipped else logging.INFO,
        "%d of %d files transformed, %d failed; %d of %d patients skipped "
        "for want of a transform",
        len(written), attempted, len(failures), len(skipped), total,
    )
    (output_dir / "AutoMatrix_report.json").write_text(json.dumps(report, indent=2))
    return output_dir


def _refuse_nothing_transformed(failures, attempted, patients, skipped, report):
    """Raise for a run that wrote nothing, chained to its first cause.

    Two cases that read differently. Nothing paired: the names the caller gave
    matched nothing, which is theirs to fix -- a ValueError, with the counts.
    Everything that paired failed: the most common failure leads, because the
    server cuts a long reason and the cause has to survive the cut.

    That second case is a ValueError only when EVERY failure was one: a
    transform or a volume that could not be read, a markups file that is not
    JSON, a matrix with no inverse -- each of them the caller's input, and
    `_apply_one` raises exactly those as ValueError. Anything else (a resample
    or a write that failed) is this tool's, so the run is a RuntimeError and
    the server answers it as its own fault rather than the caller's.
    """
    if not failures:
        raise ValueError(
            f"AutoMatrix transformed nothing: {skipped} of {patients} patients "
            f"in 'files' matched no transform in 'transforms'; "
            f"{len(report['without_a_transform'])} file(s) had no transform, "
            f"{len(report['transforms_without_a_file'])} transform(s) had no file."
        )

    kinds = Counter(f"{type(exc).__name__}: {cleaned}" for exc, cleaned in failures)
    common, count = kinds.most_common(1)[0]
    if len(common) > 200:
        common = common[:197] + "..."
    message = (
        f"0 of {attempted} files transformed; most common failure: {common} "
        f"({count} of {attempted})"
    )
    first = failures[0][0]
    every_one_the_callers = all(isinstance(exc, ValueError) for exc, _ in failures)
    raise (ValueError if every_one_the_callers else RuntimeError)(message) from first


def _positions(indices, shown: int = 10) -> str:
    """`2, 5, 9`, or the first `shown` of them and how many more."""
    head = ", ".join(str(i) for i in indices[:shown])
    rest = len(indices) - shown
    return f"{head} and {rest} more" if rest > 0 else head


def _without_names(text: str, *names) -> str:
    """`text` with the given paths, their file names and a patient key replaced.

    SimpleITK quotes the full path of a file it could not read, and a path holds
    the name the caller gave the file -- which may be a patient's. A log line
    and a refusal are kept after the run, so neither may carry one. Longest
    first, so a path is replaced whole before its basename is looked for.
    """
    candidates = set()
    for name in names:
        name = str(name)
        if not name:
            continue
        candidates.add(name)
        candidates.add(os.path.basename(name))
    for name in sorted(candidates, key=len, reverse=True):
        if os.sep in name or "." in name:
            text = text.replace(name, "<file>")
        else:
            # A patient key is a bare token, so only a whole one is replaced:
            # `P1` must not eat the start of `P12` nor of an ITK class name.
            text = re.sub(
                rf"(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])", "<patient>", text
            )
    return text


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
               name_after_transform, suffix, entry, at=lambda name: None) -> str:
    """One file through one transform. Returns the path written.

    `at(step)` is called before each step, so a failure can say which one it
    was in. What the caller sent and could not be read is raised as a
    ValueError naming the argument it came in; anything else propagates as it
    is, and `run` tells the two apart that way.
    """
    import SimpleITK as sitk

    at("reading the transform")
    try:
        transform = read_transform(matrix)
    except (RuntimeError, FileNotFoundError) as exc:
        raise ValueError(f"a 'transforms' file could not be read: {exc}") from exc
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
        at("moving the landmark points")
        # A markups file that is not JSON, and a matrix with no inverse, both
        # arrive here as ValueError already -- the caller's input either way.
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
    at("reading the volume")
    try:
        image = sitk.ReadImage(path)
    except RuntimeError as exc:
        raise ValueError(f"a 'files' volume could not be read: {itk_cause(exc)}") from exc

    # A caller that named the content is believed; "Automatic" is read off the
    # file. Recorded in the report either way, because which interpolator ran is
    # not visible in the result and is the difference between a label map that
    # survived and one that grew labels nobody segmented.
    written_entry = {"file": destination.name}
    if content == "Automatic":
        at("telling scan from segmentation")
        is_segmentation = looks_like_a_label_map(image)
        written_entry["detected"] = "segmentation" if is_segmentation else "scan"
    else:
        is_segmentation = content == "Segmentation"

    at("resampling")
    resampled = resample(image, transform, reference, is_segmentation)
    at("writing the output")
    sitk.WriteImage(resampled, str(destination))
    entry["outputs"].append(written_entry)
    return str(destination)
