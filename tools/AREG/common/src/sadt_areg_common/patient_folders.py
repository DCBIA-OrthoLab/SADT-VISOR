"""Read a cohort kept as one folder per patient.

Every AREG engine reads its two inputs as two folders, one per type: `t1` and
`t2`, or `ios` and `cbct`. Cohorts are often kept the other way round, one
folder per patient holding both (`<patient>/T1/` and `<patient>/T2/`, or
`<patient>/IOS/` and `<patient>/CBCT/`), and asking for two folders then meant
reorganising the data by hand before every run.

This module does that reorganising, and nothing else: it sorts each patient's
files into one folder per type, under names that carry the patient's folder
name, and hands the engine those folders. Everything downstream -- the
pairing, the supervised tools, the output folders -- is then exactly what a
per-type request runs, which is what makes the two layouts give the same
answer.

The patient IS its folder. A file's own name is not trusted to say whose it is:
`<patient>/T1/scan.nii.gz` is that patient's T1 whatever it is called, which
is the reason to keep data this way in the first place.
"""

import os
import shutil
import tempfile

from . import catalogs, pairing
from .errors import ToolInputError

_TIMEPOINTS = {"t1": "T1", "t2": "T2"}


def timepoint_of(relative: str):
    """"T1", "T2" or None for a file at `relative` inside a patient's folder.

    Read from a folder on the way named after the timepoint, and from a
    timepoint word in the file's own name; either is enough. Two different
    answers is no answer: a file in `T1/` whose name says T2 is left out, not
    guessed at.
    """
    *folders, name = relative.replace("\\", "/").split("/")
    by_folder = {_TIMEPOINTS[f.lower()] for f in folders if f.lower() in _TIMEPOINTS}
    by_name = {_TIMEPOINTS[t] for t in pairing.tokens(pairing.split_scan_extension(name)[0])
               if t in _TIMEPOINTS}
    found = by_folder | by_name
    return found.pop() if len(found) == 1 else None


def by_timepoint(accept, skip=()):
    """A `classify` for the two longitudinal engines: "t1", "t2" or None.

    `accept(name)` says which files the engine reads at all; a file whose name
    holds a word in `skip` is left out -- a segmentation kept beside its scan
    is not a second T1.
    """
    skip = set(skip)

    def classify(patient, relative):
        name = relative.replace("\\", "/").rsplit("/", 1)[-1]
        if not accept(name):
            return None
        if skip & set(pairing.tokens(pairing.split_scan_extension(name)[0])):
            return None
        timepoint = timepoint_of(relative)
        return timepoint.lower() if timepoint else None

    return classify


def jaw_of(name: str):
    """"U", "L" or None, from the jaw word in a mesh's name."""
    jaws = {catalogs.JAW_TOKENS[t] for t in pairing.tokens(os.path.splitext(name)[0])
            if t in catalogs.JAW_TOKENS}
    if jaws == {catalogs.JAW_UPPER}:
        return "U"
    if jaws == {catalogs.JAW_LOWER}:
        return "L"
    return None


def patients_in(root: str) -> list:
    """The patient folders of `root`, sorted. Hidden entries are not patients."""
    if not root or not os.path.isdir(root):
        raise ToolInputError(
            "'patients' must be a folder holding one sub-folder per patient."
        )
    found = sorted(
        entry for entry in os.listdir(root)
        if not entry.startswith(".") and os.path.isdir(os.path.join(root, entry))
    )
    if not found:
        raise ToolInputError(
            "'patients' holds no sub-folder. In this layout each patient has a "
            "folder of its own, holding both of that patient's inputs."
        )
    return found


def files_of(folder: str) -> list:
    """Every file under one patient's folder, as paths relative to it, sorted."""
    found = []
    for directory, directories, names in os.walk(folder):
        directories[:] = sorted(d for d in directories if not d.startswith("."))
        for name in sorted(names):
            if not name.startswith("."):
                found.append(os.path.relpath(os.path.join(directory, name), folder))
    return found


def place(source: str, destination: str) -> None:
    """Put `source` at `destination` without copying a scan when it can be helped.

    A hard link costs nothing and is a real file to every reader, which a
    symbolic link is not to a tool that refuses to follow one. It needs both
    paths on one filesystem; a copy is the fallback.
    """
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def extension_of(name: str) -> str:
    """The whole extension, `.nii.gz` included."""
    _stem, extension = pairing.split_scan_extension(name)
    if extension:
        return extension
    return os.path.splitext(name)[1]


def sort(root: str, work_dir: str, roles: tuple, classify, rename) -> dict:
    """Sort a per-patient cohort into one folder per role under `work_dir`.

    `roles` names the folders made (`("t1", "t2")`). `classify(patient,
    relative)` gives the role of one file, or None to leave it out; `rename(
    patient, role, relative, index)` gives the name it is filed under, `index`
    counting that patient's files of that role from 1. Two files given one name
    are refused rather than one silently replacing the other.

    Returns `{role: folder}`. A patient with no file for a role is left for the
    engine to report as unpaired, the way it reports one in a per-type
    request.
    """
    folders = {role: os.path.join(work_dir, role) for role in roles}
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)

    filed = 0
    for patient in patients_in(root):
        counts = dict.fromkeys(roles, 0)
        for relative in files_of(os.path.join(root, patient)):
            role = classify(patient, relative)
            if role not in folders:
                continue
            counts[role] += 1
            destination = os.path.join(folders[role], rename(patient, role, relative, counts[role]))
            if os.path.exists(destination):
                # Counts, never the names: a patient folder is the caller's own
                # identifier, and this message reaches the server's log.
                raise ToolInputError(
                    f"Two files of one patient would be read as the same "
                    f"{role.upper()} input. Keep one file per input, or name "
                    "them apart (for an arch, by its jaw: U or L)."
                )
            place(os.path.join(root, patient, relative), destination)
            filed += 1
    if not filed:
        raise ToolInputError(
            "No usable file in any patient folder of 'patients'."
        )
    return folders


def resolve(layout: str, patients, given: dict, output_dir, classify, rename) -> tuple:
    """`({role: folder}, scratch)`: the per-type folders an engine reads.

    `given` is the engine's own per-type arguments (`{"t1": ..., "t2": ...}`),
    read as they are in the per-type layout. In the per-patient one they are
    ignored and `patients` is sorted into a scratch folder beside
    `output_dir` -- beside it, not in it, so nothing of it is handed back, and
    on the filesystem the inputs were unpacked on, so the files are linked
    rather than copied. `scratch` is that folder, for the caller to remove, or
    None.
    """
    if layout == catalogs.LAYOUT_BY_PATIENT:
        parent = os.path.dirname(os.path.abspath(str(output_dir))) if output_dir else None
        scratch = tempfile.mkdtemp(prefix="areg-patients-", dir=parent)
        try:
            return sort(str(patients or ""), scratch, tuple(given), classify, rename), scratch
        except BaseException:
            shutil.rmtree(scratch, ignore_errors=True)
            raise
    if layout != catalogs.LAYOUT_BY_TYPE:
        raise ToolInputError(
            f"Unknown input layout {layout!r}: choose one of {', '.join(catalogs.LAYOUT_CHOICES)}."
        )
    missing = [role for role, value in given.items() if not str(value or "").strip()]
    if missing:
        raise ToolInputError(
            f"'{missing[0]}' is empty. Send one folder per type, or choose "
            f"\"{catalogs.LAYOUT_BY_PATIENT}\" and send the folder of patients."
        )
    return {role: str(value) for role, value in given.items()}, None
