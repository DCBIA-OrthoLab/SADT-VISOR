"""AREG_IOSCBCT -- register an intraoral scan onto a CBCT of the same patient.

NOT longitudinal, unlike its two siblings: AREG_CBCT and AREG_IOS take a
baseline and a follow-up of one modality, this takes ONE timepoint imaged two
ways. Upstream's own test set says so -- `P001_T2_U.vtk` beside
`P_0001_T2.nii.gz`, both T2 -- which is why the arguments are `ios` and `cbct`
rather than `t1` and `t2`.

**It has no engine of its own, and that is the design.** No torch, no pytorch3d,
no nnUNet: the landmarks come from ALI_CBCT and ALI_IOS, the tooth labels from
Crown_Seg and the orientation from ASO, each in its own virtualenv, reached
through the supervisor. Containing both stacks would mean one environment
holding torch 2.8 for one half and 2.11 for the other -- the exact defect the
AREG and ALI splits removed. What is left here is geometry.
"""

import functools
import os
import tempfile
from pathlib import Path
from typing import Literal

import shutil

from sadt_areg_common import catalogs, pairing, patient_folders
from sadt_areg_common.errors import ToolInputError

from . import pipeline
from .dispatch import main


# What each path argument can read, for the client's file dialog and for the
# server's refusal of an upload that contradicts it. DERIVED from the tables
# the pipeline already registers against -- a second spelling of them would be
# a second thing to keep in step.
#
# Without this the intraoral picker offered a folder of CBCT volumes, and the
# mistake surfaced as "No patient has both an intraoral scan and a CBCT" a
# minute into the run rather than at the click.
ACCEPTS = {
    "ios": pipeline.SURFACE_EXTENSIONS,
    "cbct": pairing.SCAN_EXTENSIONS,
    "ios_landmarks": pipeline.LANDMARK_EXTENSIONS,
    "cbct_landmarks": pipeline.LANDMARK_EXTENSIONS,
}


def _role(patient: str, relative: str):
    """"ios" for a mesh, "cbct" for a volume, None for anything else.

    By the file's type alone: the two modalities are never the same kind of
    file, so a patient folder needs no `IOS/` and `CBCT/` sub-folders -- though
    it may have them.
    """
    name = relative.replace("\\", "/").rsplit("/", 1)[-1]
    if name.lower().endswith(pipeline.SURFACE_EXTENSIONS):
        return "ios"
    if pairing.is_scan_file(name):
        return "cbct"
    return None


def _timepoint_in(relative: str):
    """The timepoint a file inside a patient folder names, in its own name or
    in a sub-folder (`T1/scan.nii.gz`), or None."""
    return pipeline.timepoint_of(os.path.join("patient", relative), "patient")


def _patient_timepoints(root) -> dict:
    """`{patient folder: {timepoints}}` over the files this tool reads.

    What `_filed_as` needs to know whether a patient holds more than one
    timepoint, which one file cannot tell it.
    """
    found = {}
    if not str(root or "").strip() or not os.path.isdir(str(root)):
        return found
    for patient in patient_folders.patients_in(str(root)):
        points = found.setdefault(patient, set())
        for relative in patient_folders.files_of(os.path.join(str(root), patient)):
            if _role(patient, relative):
                points.add(_timepoint_in(relative))
    return found


def _filed_as(patient: str, role: str, relative: str, index: int,
              timepoints=None) -> str:
    """`<patient>_U.stl`, `<patient>_CBCT.nii.gz`: the names the pairing reads.

    The patient's folder name comes first, so `pipeline.patient_name` gives it
    back as the output folder; the jaw stays, since that is how each arch finds
    its landmarks. Patients are matched on the DIGITS of their names, so a
    folder name needs one.

    The timepoint stays too when the patient's folder holds more than one
    (`timepoints`, from `_patient_timepoints`): `<patient>_T1_CBCT.nii.gz`
    beside `<patient>_T2_CBCT.nii.gz`. Renamed without it, a T1 and a T2 were
    two files under one name and the run was refused. A folder holding one
    timepoint keeps the names, and the outputs, it always had.
    """
    if not any(c.isdigit() for c in patient):
        raise ToolInputError(
            "A patient folder's name carries no number. Patients are matched on "
            "the digits of their names: name each folder with its patient's "
            "number (e.g. patient001)."
        )
    name = relative.replace("\\", "/").rsplit("/", 1)[-1]
    extension = patient_folders.extension_of(name)
    point = _timepoint_in(relative)
    if point and len((timepoints or {}).get(patient, ())) > 1:
        patient = f"{patient}_{point.upper()}"
    if role == "cbct":
        return f"{patient}_CBCT{extension}"
    jaw = patient_folders.jaw_of(name)
    return f"{patient}_{jaw or f'IOS{index}'}{extension}"


def _as_folders(folders: dict, output_dir: Path) -> tuple:
    """`(folders, scratch)` with a lone FILE given for a side put in a folder.

    The panel lets either side be one file -- the CBCT of the patient at hand
    beside a folder of intraoral scans -- and everything downstream walks a
    folder: `os.walk` over a file yields nothing, so a run given one CBCT
    answered "No CBCT scan found". The file is linked into a scratch folder
    beside `output_dir`, as `patient_folders.resolve` does for its layout, and
    `scratch` is that folder for the caller to remove, or None.
    """
    files = {role: path for role, path in folders.items() if os.path.isfile(path)}
    if not files:
        return folders, None
    parent = os.path.dirname(os.path.abspath(str(output_dir))) if str(output_dir) else None
    scratch = tempfile.mkdtemp(prefix="areg-single-", dir=parent)
    folders = dict(folders)
    for role, path in files.items():
        folder = os.path.join(scratch, role)
        os.makedirs(folder)
        target = os.path.join(folder, os.path.basename(path))
        try:
            os.link(path, target)
        except OSError:
            shutil.copy2(path, target)
        folders[role] = folder
    return folders, scratch


def run(
    # First, because it decides which of the next three the panel asks for.
    # Spelled out because `Literal` takes literals only; a test asserts it
    # against `catalogs.LAYOUT_CHOICES`.
    input_layout: Literal["One folder per type", "One folder per patient"] = "One folder per type",
    # Not required any more, and not because they are optional: which of
    # `ios`/`cbct` or `patients` is needed depends on `input_layout`, and a
    # field the panel marks required in the layout that does not read it would
    # refuse to apply. `patient_folders.resolve` refuses what is missing.
    ios: Path = "",
    cbct: Path = "",
    patients: Path = "",
    output_dir: Path = "",
    # "From the data" is the default and names no mode: `dispatch.derive_automation`
    # reads it off the request -- both landmark sets supplied is Registration,
    # which predicts nothing. Whether to orient the CBCT first is `orient_cbct_first`
    # below, because no folder can answer it. Naming a mode is an override.
    automation: Literal[
        "From the data", "Registration", "Semi-Automated", "Fully-Automated"
    ] = "From the data",
    ios_landmarks: Path = "",
    cbct_landmarks: Path = "",
    cbct_reference: Path = "",
    landmark_model: Path = "",
    ios_landmark_model: Path = "",
    crown_model: Path = "",
    max_dist: float = 0.0,
    # The one thing about this mode that a folder cannot answer, asked as what it
    # is instead of hidden inside a three-valued `automation`: orienting the CBCT
    # into a standard frame before the cross-modality registration is a clinical
    # decision, not a property of the files. On by default -- it is what
    # "Fully-Automated" did, and the frame the rest of the chain expects.
    #
    # Read only when no landmarks are supplied: with both sets in hand there is
    # nothing to predict and nothing to orient for.
    orient_cbct_first: bool = True,
    output_suffix: str = "Reg",
    # Nothing ticked by default: the CBCT is the heaviest file of a run, and
    # each of these is for a reader who asked for it. Spelled out because
    # `Literal` takes literals only; a test pins it to `catalogs`.
    outputs: list[Literal["Registration landmarks (IOS + CBCT)", "Orientation landmarks (ASO)",
                          "CBCT scan", "Transform (.tfm)"]] = [],
    *,
    sup=None,
) -> Path:
    """Register an intraoral scan onto a CBCT of the same patient.

    Args:
        input_layout: How the files are organised: one folder for each type
            of file (T1 and T2, or intraoral scans and CBCT) holding every
            patient, or one folder for each patient holding all of that
            patient's files. Rest the pointer on each option for an example.
        patients: The folder of patient folders, in the one-folder-per-patient
            layout. Each patient's folder name, with its number, names its
            results.
        ios: The intraoral surfaces (.vtk/.stl): one file, or a folder searched
            recursively. Upper and lower arches are registered separately and
            matched to their landmarks by the jaw token in the name.
        cbct: The CBCT volumes: one file, or a folder. Paired to the intraoral
            scans on the digits of the patient identifier, the two modalities
            being named by different conventions, and on the timepoint (T1, T2)
            when the names give one, so each timepoint is registered onto its
            own CBCT.
        output_dir: Where the registered meshes, their 4x4 matrices and
            `AREG_report.json` are written, one folder per patient named as
            its files name it (`case_042`). When the CBCT was oriented first,
            the oriented scan the meshes are registered onto and its `.tfm`
            are written beside them. Nothing is written outside it.
        automation: Registration takes both landmark sets and predicts nothing --
            no other tool is called and no GPU is needed. Semi-Automated labels
            the crowns and predicts both sets. Fully-Automated orients the CBCT
            first, which needs a reference.
        ios_landmarks: Registration mode. Your own intraoral landmarks.
        cbct_landmarks: Registration mode. Your own CBCT landmarks.
        cbct_reference: A reference bundle to orient the CBCT volumes onto
            instead of the Frankfurt horizontal frame ASO holds. Left empty --
            the normal case -- ASO resolves its own.
        landmark_model: The CBCT landmark bundle, for the modes that predict.
        ios_landmark_model: The intraoral landmark bundle.
        crown_model: The crown-labelling checkpoint, for the modes that label.
        max_dist: How far a point may be from its nearest neighbour and still
            count as an ICP correspondence, in millimetres. 0 uses 1.5, which is
            upstream's.
        orient_cbct_first: Put the CBCT in the Frankfurt horizontal and
            mid-sagittal frame before the cross-modality registration. One frame
            only, unlike the CBCT-to-CBCT engine which offers two: the landmarks
            the rest of this chain reads are defined in this one, so another
            would silently change what they mean. Read only when no landmarks
            are supplied, there being nothing to orient for once they are.
        output_suffix: Added to each output name, e.g. `scan_Reg.vtk`.
        outputs: What to hand back beside the registered meshes, all in the
            CBCT's space so it opens on top of them. Registration landmarks
            (IOS + CBCT): each mesh's own, moved with it, and the CBCT
            landmarks used. Orientation landmarks (ASO): the landmarks ASO
            oriented the CBCT on, when the run oriented it. CBCT scan:
            the CBCT the meshes were registered onto. Transform (.tfm): each
            registration as a `.tfm`. Each patient's `scene.json` lists what was written.

    Returns:
        The output directory.
    """
    output_dir = Path(output_dir)
    timepoints = (_patient_timepoints(patients)
                  if input_layout == catalogs.LAYOUT_BY_PATIENT else {})
    folders, scratch = patient_folders.resolve(
        input_layout, patients, {"ios": ios, "cbct": cbct}, output_dir, _role,
        functools.partial(_filed_as, timepoints=timepoints),
    )
    single = None
    try:
        folders, single = _as_folders(folders, output_dir)
        main(
            ios=folders["ios"], cbct=folders["cbct"], output_dir=output_dir,
            automation=automation,
            ios_landmarks=ios_landmarks, cbct_landmarks=cbct_landmarks,
            cbct_reference=cbct_reference, landmark_model=landmark_model,
            ios_landmark_model=ios_landmark_model, crown_model=crown_model,
            max_dist=max_dist, output_suffix=output_suffix,
            orient_cbct_first=orient_cbct_first, outputs=outputs, sup=sup,
        )
    finally:
        for folder in (scratch, single):
            if folder:
                shutil.rmtree(folder, ignore_errors=True)
    return output_dir
