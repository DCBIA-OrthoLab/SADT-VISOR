"""The CBCT half of ASO: discover scans, recentre them, register on a
reference, write the oriented volume, its landmarks and its transform.

Ported from `ASO_CBCT/PRE_ASO_CBCT/PRE_ASO_CBCT.py` (the recentring) and
`ASO_CBCT/SEMI_ASO_CBCT/SEMI_ASO_CBCT.py` (the registration driver). Both
CLIs' envelopes are gone: no `<filter-progress>` prints, no `time.sleep`
progress theatre, no `sys.exit`, no writing into the caller's input tree.

The two automation modes share this whole file. Semi-automated reads the
landmarks the caller sent; fully-automated is handed predicted ones. That is
the only difference, and it is a parameter.
"""

import logging
import os
import re

import numpy as np
import SimpleITK as sitk

from .. import markups
from sadt_naming import VOLUME as SCAN_EXTENSIONS, compressed_extension
from . import icp

logger = logging.getLogger(__name__)

# Suffixes previous runs (of ASO, AMASSS or ALI) leave on a file name, stripped
# to recover the patient they belong to. Longest first so "_Scanreg" is not cut
# short by "_Scan".
#
# "_T1"/"_T2" are deliberately NOT here: they mark two timepoints of the same
# subject, which are two scans to orient, not one. Collapsing them dropped the
# second scan and merged both landmark sets into the survivor.
PATIENT_SUFFIXES = (
    "_lm_Pred", "_Scanreg", "_MERGED", "_scan", "_Scan", "_Or", "_OR", "_lm",
)

# The same markers, PUBLISHED. `describe.py` reads this name out of the source
# and puts it in the schema, so a server that has to work out which results
# belong to one patient -- to replay a chain for the cases a clinician marked
# and leave the rest alone -- can do it without holding a table of its own.
# It knows no dental tool, and this is how that stays true.
#
# Only the STABLE part of each marker. What this tool appends is
# `_{output_suffix}` and what ALI appends is `_lm_{prediction_ID}`, both
# caller-controlled: `_Or` is a default, `_lm` is the part that does not
# move. Cutting at the earliest match handles the rest.
OUTPUT_SUFFIXES = PATIENT_SUFFIXES

# One separator character, for asking whether a suffix match ends where a token
# ends. The same four the sibling engines split names on -- `ios/pipeline.py`'s
# `_SPLIT` and the shared `sadt_areg_common.pairing` both use `_ - .`, and
# whitespace is added because a name a clinician typed can contain it and a
# space closes a word as plainly as an underscore does.
_SEPARATOR = re.compile(r"[_\-.\s]")


def _token_aligned_index(stem: str, suffix: str) -> int:
    """Where `suffix` occurs in `stem` as WHOLE tokens, or -1.

    `str.find` alone was the defect this exists to remove. Every entry of
    `PATIENT_SUFFIXES` opens on an underscore, so a raw `find` is already
    anchored on its left -- and anchored nowhere on its right, which is what
    let `_OR` match inside `_ORTHO`, `_Scan` inside `_Scanner`, and `_lm`
    inside `_lmk`:

        SMITH_ORTHO_Scan.nii.gz  ->  SMITH
        SMITH_ORLANDO_Scan.nii.gz -> SMITH

    Two subjects whose names differ only after the false match collapse into
    one key, and `discover` then keeps one scan and gives it both landmark
    sets -- silently, in a run that reports success.

    A match therefore has to end on a token boundary: the end of the stem, or a
    separator. The left side is checked rather than assumed, so an entry added
    to the table without a leading separator would still be matched as a token
    and not as a substring.

    The same rule, and the same helper, as the shared
    `sadt_areg_common.pairing._token_aligned_index`, where it was fixed first.
    Copied rather than imported: ASO does not depend on that package today and
    adding the dependency is a bigger decision than this fix.
    """
    start = stem.find(suffix)
    while start >= 0:
        end = start + len(suffix)
        opens_on_a_boundary = (
            start == 0
            or bool(_SEPARATOR.match(suffix[0]))
            or bool(_SEPARATOR.match(stem[start - 1]))
        )
        closes_on_a_boundary = end == len(stem) or bool(_SEPARATOR.match(stem[end]))
        if opens_on_a_boundary and closes_on_a_boundary:
            return start
        start = stem.find(suffix, start + 1)
    return -1


def patient_stem(filename: str) -> str:
    """The patient a file belongs to, from its name alone.

    Only ever compared against other files in the SAME directory (see
    `discover`), so two patients whose names collapse to the same stem in
    different folders stay separate.

    A suffix is truncated, not deleted: everything it introduces goes with it,
    so `P1_Scan_reoriented.nii.gz` keys to `P1` and not to `P1_reoriented`. It
    has to match on token boundaries -- see `_token_aligned_index` -- and a
    stem that BEGINS with one is left alone, because there is no patient in
    front of it to keep.
    """
    stem = filename
    for extension in SCAN_EXTENSIONS:
        if stem.lower().endswith(extension):
            stem = stem[: -len(extension)]
            break
    else:
        for extension in markups.MARKUPS_EXTENSIONS:
            if stem.lower().endswith(extension):
                stem = stem[: -len(extension)]
                break
        else:
            stem = os.path.splitext(stem)[0]

    for suffix in PATIENT_SUFFIXES:
        index = _token_aligned_index(stem, suffix)
        if index > 0:
            stem = stem[:index]
    return stem


def is_previous_output(filename: str, suffix: str) -> bool:
    """True if this file looks like something a previous ASO run wrote.

    Running twice on the same folder must not orient an already-oriented scan.
    `patient1_Or.nii.gz` sorts BEFORE `patient1_scan.nii.gz`, so without this a
    second run would silently pick the first run's output as its input -- and a
    previous run's `patient1_lm_Or.mrk.json` would overwrite the caller's own
    landmarks during the merge.
    """
    stem = filename
    for extension in SCAN_EXTENSIONS + markups.MARKUPS_EXTENSIONS + (".tfm",):
        if stem.lower().endswith(extension):
            stem = stem[: -len(extension)]
            break
    return stem.endswith(f"_{suffix}") or stem.endswith(f"_{suffix}_transform")


def _group_by_patient(input_root: str, output_suffix: str) -> dict:
    """Every file under `input_root`, bucketed by the patient its NAME implies.

    Shared by `discover` and `orphan_markups` so the two can never disagree
    about what belongs to whom -- which is the whole point of reporting the
    leftovers: a second implementation would eventually call a different file
    an orphan than the one discovery dropped.
    """
    patients: dict = {}
    for directory, _, file_names in os.walk(input_root):
        relative = os.path.relpath(directory, input_root)
        prefix = "" if relative == "." else relative
        for file_name in sorted(file_names):
            if file_name.startswith("."):
                continue
            key = os.path.join(prefix, patient_stem(file_name))
            entry = patients.setdefault(
                key, {"scans": [], "old_scans": [], "markups": [], "old_markups": []}
            )
            path = os.path.join(directory, file_name)
            previous = is_previous_output(file_name, output_suffix)
            if file_name.lower().endswith(SCAN_EXTENSIONS):
                entry["old_scans" if previous else "scans"].append(path)
            elif markups.is_markups_file(file_name):
                entry["old_markups" if previous else "markups"].append(path)
    return patients


def orphan_markups(input_root: str, output_suffix: str = "Or") -> dict:
    """{patient key it was filed under: [markups paths]} for every landmark
    file that matched no scan.

    `discover` drops these groups, and used to drop them in complete silence.
    The caller was then told "no landmark file (.mrk.json) alongside this
    scan" about a scan whose landmark file was sitting next to it -- named
    just differently enough that `patient_stem` put the two in separate
    buckets (`P1_CBCT.nii.gz` beside `P1.mrk.json`). Nothing said so: not the
    report, not the logs.

    A run that produces nothing has to be able to say why, so the leftovers
    are collected and named. Files a previous run wrote are not leftovers --
    they are deliberately set aside -- so only the fresh ones count.
    """
    return {
        key: entry["markups"]
        for key, entry in _group_by_patient(input_root, output_suffix).items()
        if entry["markups"] and not (entry["scans"] or entry["old_scans"])
    }


def discover(input_root: str, output_suffix: str = "Or") -> dict:
    """{patient key: {"scan": path, "markups": [paths]}} for one input tree.

    The key is the patient's path RELATIVE to the input root, not its base
    name: `GetPatients` keyed on the base name, so two scans called
    `scan.nii.gz` in different subfolders became one patient. The output keeps
    the input's tree.

    A patient may have several markups files (fully-automated runs produced one
    per landmark group); they are merged in memory, where `MergeJson` rewrote
    the caller's input folder and DELETED the sources.

    Files that look like a previous run's output are set aside and used only if
    a patient has nothing else, so re-running on an output folder works while a
    folder holding both an original and its orientation uses the original.
    """
    patients = _group_by_patient(input_root, output_suffix)

    found = {}
    for key, entry in patients.items():
        # A patient with several scans in one folder keeps the first in sorted
        # order; the rest would be re-orientations of the same subject and are
        # not silently mixed in.
        scans = entry["scans"] or entry["old_scans"]
        if not scans:
            continue
        found[key] = {
            "scan": scans[0],
            "markups": entry["markups"] or entry["old_markups"],
        }
    return found


def load_landmarks(paths: list) -> dict:
    """Merge every markups file a patient has into one landmark dict.

    Later files win on a repeated label, which only happens when a caller sends
    overlapping groups. An unreadable file is skipped with a warning giving its
    position and the failure's type -- never its name, and never the data.
    """
    merged: dict = {}
    for index, path in enumerate(paths, start=1):
        try:
            merged.update(markups.load_landmarks(path))
        except (ValueError, OSError) as exc:
            # The TYPE and not the message, which is the unobvious half: the
            # ValueError `markups.load_landmarks` raises quotes the file's own
            # name, so passing `exc` through would put it back. The type is the
            # diagnosis anyway -- JSONDecodeError, ValueError and
            # FileNotFoundError are three different mistakes.
            logger.warning(
                "Skipping markups file %d of %d: %s",
                index, len(paths), type(exc).__name__,
            )
    return merged


def load_reference(reference_dir: str) -> dict:
    """The reference landmark set, from the first markups file in the bundle.

    The reference SCAN is not read. The original loaded it
    (`gold_image = sitk.ReadImage(gold_file)`) and never used the result, while
    requiring one to be present -- so a reference bundle holding just the
    landmarks it actually registers against was rejected with an IndexError.
    """
    for directory, _, file_names in sorted(os.walk(reference_dir)):
        for file_name in sorted(file_names):
            if markups.is_markups_file(file_name) and not file_name.startswith("."):
                landmarks = markups.load_landmarks(os.path.join(directory, file_name))
                if landmarks:
                    return landmarks
    raise ValueError(
        "The reference bundle holds no readable landmark file (.mrk.json). A CBCT "
        "reference is one already-oriented case's landmarks."
    )


def recenter(image: sitk.Image) -> tuple:
    """Move a volume's field-of-view centre onto the physical origin.

    Returns `(recentred image, the translation that was applied)`. This is the
    whole of what `PRE_ASO_CBCT` still does -- its `model_folder`, `SmallFOV`
    and `temp_folder` arguments were read and never used once the learned
    orientation step was removed.
    """
    center = np.array(
        image.TransformContinuousIndexToPhysicalPoint(
            (np.array(image.GetSize()) / 2.0).tolist()
        )
    )
    translation = sitk.TranslationTransform(3)
    translation.SetOffset((-center).tolist())

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image)
    resampler.SetTransform(translation.GetInverse())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetSize([int(size) for size in image.GetSize()])
    resampler.SetOutputOrigin((np.array(image.GetOrigin()) - center).tolist())
    return resampler.Execute(image), translation


def prepare(scan_path: str, destination: str = None) -> tuple:
    """Read a scan and recentre it. Returns `(centred image, translation)`.

    With `destination` set the centred volume is also written there -- which is
    what the fully-automated mode needs, since the landmark tool runs on the
    centred scans, not the originals.
    """
    centered, translation = recenter(sitk.ReadImage(scan_path))
    if destination:
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        sitk.WriteImage(centered, destination, useCompression=True)
    return centered, translation


def center_landmarks(landmarks: dict, translation: sitk.Transform) -> dict:
    """Move landmarks into the centred volume's space.

    Recentring resamples the volume so a feature at physical point q ends up at
    q + offset. The landmarks have to follow, or the registration would compare
    a centred scan against uncentred points -- which is the state the original
    semi-automated CLI ran in, since it recentred nothing and then read a `.tfm`
    that only the fully-automated chain ever produced.
    """
    offset = np.array(translation.GetOffset())
    return {name: point + offset for name, point in landmarks.items()}


def orient_patient(
    centered: sitk.Image,
    pre_transform: sitk.Transform,
    source_landmarks: dict,
    reference_landmarks: dict,
    requested: list,
    output_dir: str,
    relative_key: str,
    suffix: str,
    extension: str,
    max_triplets: int,
    seed: int,
) -> dict:
    """Orient one centred CBCT and write its three outputs. Returns a report entry.

    Raises `icp.RegistrationError` when this patient cannot be registered; the
    caller records that and moves on to the next one.
    """
    registration = icp.register(
        source_landmarks,
        reference_landmarks,
        requested,
        pre_transform=pre_transform,
        max_triplets=max_triplets,
        seed=seed,
    )

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(centered)
    resampler.SetTransform(registration.resample_transform)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    oriented = resampler.Execute(centered)

    relative_dir, patient = os.path.split(relative_key)
    destination = os.path.join(output_dir, relative_dir)
    os.makedirs(destination, exist_ok=True)

    scan_output = os.path.join(destination, f"{patient}_{suffix}{compressed_extension(extension)}")
    sitk.WriteImage(oriented, scan_output, useCompression=True)

    landmark_output = os.path.join(destination, f"{patient}_lm_{suffix}.mrk.json")
    markups.write_landmarks(registration.landmarks, landmark_output)

    transform_output = os.path.join(destination, f"{patient}_{suffix}_transform.tfm")
    sitk.WriteTransform(registration.full_transform, transform_output)

    return {
        "status": "ok",
        "landmarks_used": registration.used,
        "landmarks_dropped": registration.dropped,
        "produced": sorted(
            os.path.relpath(path, output_dir)
            for path in (scan_output, landmark_output, transform_output)
        ),
    }
