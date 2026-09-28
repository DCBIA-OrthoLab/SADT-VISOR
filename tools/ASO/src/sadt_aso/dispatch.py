"""ASO -- Automated Standardized Orientation.

Ported from the Slicer extension's `ASO/` module and its four CLI modules
(`ASO_CBCT/{PRE,SEMI}_ASO_CBCT`, `ASO_IOS/{PRE,SEMI}_ASO_IOS`). One tool, two
engines, four modes:

|            | Semi-Automated                        | Fully-Automated                       |
|------------|---------------------------------------|---------------------------------------|
| **CBCT**   | landmarks you send, ICP onto a gold   | landmarks predicted first (see        |
|            | landmark set                          | `_predict_landmarks`), then the same  |
|            |                                       | ICP                                   |
| **IOS**    | landmarks you send, ICP per jaw       | tooth centroids of an already         |
|            |                                       | segmented mesh, ICP per jaw           |

The Slicer envelope is gone entirely: no `<filter-progress>` prints, no
`time.sleep` progress theatre, no `sys.exit`, no log file the client polls, no
`*Error.txt` beside the results, and nothing written into the caller's input
tree.

**Fully-automated CBCT calls another tool, and calls it from the middle.** The
landmark tool runs on the RECENTRED scans -- that is what the Slicer chain did,
`PRE_ASO_CBCT` before `ALI_CBCT` -- so it cannot be run before ASO and its
result handed in. The recentring is a pure metadata change, so it *ought* to be
reorderable, and it very nearly is; `ALI`'s `physical_position` takes the
absolute value of the origin, which does not commute with moving it (see issue
#11). Rather than bet on that, the order is kept exactly as it was and the
landmark tool is reached through the supervisor, at the point it always ran.

`sup` is keyword-only and unannotated, which is what marks it as the supervisor
rather than an argument: `describe.py` keeps it out of the schema, and a client
never sends it. It is duck-typed -- nothing here imports a supervisor type,
because doing so would need a package shared with the server, which is the
coupling this repository exists to remove.
"""

import json
import logging
import os
import shutil

from . import catalogs
from . import markups
from . import progress
from sadt_naming import split_extension as split_scan_extension
from .cbct import dicom
from .cbct import pipeline as cbct_pipeline
from .errors import SupervisorRequired, ToolInputError
from .ios import pipeline as ios_pipeline

logger = logging.getLogger(__name__)

REPORT_NAME = "ASO_report.json"

# Intermediates live here, under the output directory the caller owns, and are
# removed before `orient` returns. A surviving `.aso_work/` means a run crashed.
WORK_DIRNAME = ".aso_work"

# The tool asked for the landmarks of a fully-automated CBCT run. A string, not
# a dynamic attribute on the supervisor: a typo in a string is greppable and
# the call graph stays inspectable, where `sup.ALI(...)` is an AttributeError
# fifteen minutes into a job.
LANDMARK_TOOL = "ALI_CBCT"



class OrientationRun:
    """Result of `orient()`: where the files are, and what actually happened.

    A run is reported per patient, so a batch where three scans out of forty
    could not be registered says which three and why -- the original wrote a
    `<name>Error.txt` into the output folder and carried on, which read as a
    successful run with a few odd text files in it.
    """

    def __init__(self, output_dir: str, report: dict):
        self.output_dir = output_dir
        self.report = report

    @property
    def patients(self) -> dict:
        return self.report["cases"]

    @property
    def succeeded(self) -> list:
        return [
            key for key, entry in self.patients.items() if entry.get("status") == "ok"
        ]

    @property
    def output_files(self) -> list:
        return [
            os.path.join(self.output_dir, relative)
            for entry in self.patients.values()
            for relative in entry.get("produced", [])
        ]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def orient(
    input_path: str,
    reference_path: str,
    output_dir: str,
    modality: str,
    automation: str,
    cbct_landmarks=None,
    ios_teeth=None,
    ios_landmark_types=None,
    ios_jaws=None,
    ios_occlusion: str = catalogs.OCCLUSION_INDEPENDENT,
    landmarks_path: str = "",
    landmark_model: str = "",
    dicom_input: bool = False,
    output_suffix: str = "Or",
    max_triplets: int = 2500,
    seed: int = 0,
    sup=None,
) -> OrientationRun:
    """Orient every case under `input_path` onto `reference_path`.

    Every cross-argument rule is checked BEFORE any file is read: a request that
    cannot work has to come back in a second, not after minutes of registration.
    """
    modality, automation = str(modality), str(automation)
    occlusion = str(ios_occlusion or catalogs.OCCLUSION_INDEPENDENT)
    suffix = (output_suffix or "Or").strip() or "Or"
    if os.sep in suffix or (os.altsep and os.altsep in suffix):
        raise ToolInputError("'output_suffix' is a name fragment, not a path.")

    if modality == catalogs.MODALITY_CBCT:
        requested = _selected(cbct_landmarks, catalogs.CBCT_LANDMARK_CHOICES, "cbct_landmarks")
        _check_cbct(automation, requested, landmarks_path, landmark_model, sup)
        selection = {"cbct_landmarks": requested}
    elif modality == catalogs.MODALITY_IOS:
        teeth = _selected(ios_teeth, catalogs.TOOTH_CHOICES, "ios_teeth")
        types = _selected(
            ios_landmark_types, catalogs.IOS_LANDMARK_TYPE_CHOICES, "ios_landmark_types"
        )
        jaws = _selected(ios_jaws, catalogs.JAW_CHOICES, "ios_jaws")
        _check_ios(automation, teeth, types, jaws, occlusion)
        selection = {"ios_teeth": teeth, "ios_landmark_types": types, "ios_jaws": jaws}
    else:
        raise ToolInputError(
            f"Unknown 'modality' {modality!r}. Expected one of: "
            f"{', '.join(catalogs.MODALITY_CHOICES)}"
        )

    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    work_dir = os.path.join(output_dir, WORK_DIRNAME)
    os.makedirs(work_dir, exist_ok=True)

    report = {
        "modality": modality,
        "automation": automation,
        "output_suffix": suffix,
        "reference": os.path.basename(str(reference_path).rstrip(os.sep)),
        "cases": {},
    }

    try:
        input_root = _as_directory(input_path, os.path.join(work_dir, "input"))
        reference_root = _as_directory(reference_path, os.path.join(work_dir, "reference"))
        # Either a bundle, or the folder that HOLDS the bundles. The server
        # fills a hidden hosted-model argument with the whole of
        # `DATA/ASO/models/`, so both shapes arrive here and the difference is
        # visible: a bundle carries markups, a folder of bundles does not.
        if not _is_reference_bundle(reference_root):
            reference_root = choose_reference(
                reference_root,
                modality,
                selection["cbct_landmarks"] if modality == catalogs.MODALITY_CBCT else [],
            )
            report["reference"] = os.path.basename(reference_root.rstrip(os.sep))

        if modality == catalogs.MODALITY_CBCT:
            _run_cbct(
                input_root=input_root,
                reference_root=reference_root,
                automation=automation,
                requested=selection["cbct_landmarks"],
                landmarks_path=landmarks_path,
                landmark_model=landmark_model,
                dicom_input=dicom_input,
                output_dir=output_dir,
                work_dir=work_dir,
                suffix=suffix,
                max_triplets=max_triplets,
                seed=seed,
                report=report,
                sup=sup,
            )
        else:
            _run_ios(
                input_root=input_root,
                reference_root=reference_root,
                automation=automation,
                teeth=selection["ios_teeth"],
                landmark_types=selection["ios_landmark_types"],
                jaws=selection["ios_jaws"],
                occlusion=occlusion,
                output_dir=output_dir,
                suffix=suffix,
                max_triplets=max_triplets,
                seed=seed,
                report=report,
            )
    finally:
        # Extracted inputs, converted DICOM, the centred volumes and whatever
        # the landmark tool wrote. Removed whether or not the run succeeded, so
        # what is left under output_dir is results and nothing else.
        shutil.rmtree(work_dir, ignore_errors=True)

    _summarize(report)
    with open(os.path.join(output_dir, REPORT_NAME), "w") as handle:
        json.dump(report, handle, indent=2)

    return OrientationRun(output_dir, report)


# ---------------------------------------------------------------------------
# Argument rules
# ---------------------------------------------------------------------------

def _selected(value, choices: dict, argument: str) -> list:
    """A multichoice argument's value, in declaration order.

    None means the argument was omitted and its declared defaults apply. An
    unknown name raises rather than being dropped: `Literal` is published, not
    enforced -- the runner calls `run(**params)` from a JSON object -- so a
    stale client naming an option that no longer exists must be told, not handed
    a narrower run than it asked for.
    """
    if value is None:
        return [name for name, on in choices.items() if on]
    unknown = sorted(set(value) - set(choices))
    if unknown:
        raise ToolInputError(
            f"Unknown option(s) in '{argument}': {', '.join(unknown)}. "
            f"Known: {', '.join(choices)}."
        )
    wanted = set(value)
    return [name for name in choices if name in wanted]


def _check_cbct(
    automation: str, landmarks: list, landmarks_path: str, landmark_model: str, sup
) -> None:
    if len(landmarks) < 3:
        raise ToolInputError(
            f"CBCT orientation registers on at least 3 landmarks; "
            f"{len(landmarks)} are selected in 'cbct_landmarks'."
        )
    if automation != catalogs.AUTOMATION_FULLY or landmarks_path:
        return

    # Checked before the input is even read. Neither order changes what fails,
    # but this one changes what the caller is told: with no way to reach the
    # landmark tool, "send landmarks or use Semi-Automated" is the answer
    # whatever they put in 'landmark_model'.
    if sup is None:
        raise SupervisorRequired(
            f"Fully-Automated CBCT predicts the landmarks with the '{LANDMARK_TOOL}' tool, "
            f"and nothing here can run it: no supervisor was supplied. Send the landmarks "
            f"yourself in 'landmarks' (a folder of .mrk.json files, which is what "
            f"'{LANDMARK_TOOL}' produces), or use Semi-Automated mode."
        )


def _no_landmarks_reason(key: str, markups_paths: list, orphans: list) -> str:
    """Why a Semi-Automated patient has no landmarks -- three situations the
    one message used to cover with the only wording that could be wrong.

    "No landmark file alongside this scan" was emitted whenever the merged
    dict came out empty, which happens when the file is absent, when it was
    matched but unreadable (`load_landmarks` only logs that), and when a file
    IS in the folder but under a name that put it in another bucket. Telling
    a user their file is missing while it sits next to the scan is the worst
    of the three, and it was the likeliest.
    """
    if markups_paths:
        names = ", ".join(sorted(os.path.basename(path) for path in markups_paths))
        return (
            f"the landmark file(s) found for this scan ({names}) hold no usable "
            f"control point -- check the log for the file that was skipped"
        )
    if orphans:
        return (
            f"no landmark file matched this scan. {len(orphans)} landmark file(s) in "
            f"the input matched no scan at all ({', '.join(orphans)}): a scan and its "
            f"landmarks are paired by name, up to a trailing "
            f"{', '.join(cbct_pipeline.PATIENT_SUFFIXES)}. Rename them so both sides "
            f"share the same stem -- e.g. '{key}_scan.nii.gz' with '{key}_lm.mrk.json'"
        )
    return (
        "no landmark file (.mrk.json) alongside this scan. Semi-Automated mode "
        "registers on landmarks you provide, so they must travel with the scans -- "
        "send the whole FOLDER rather than the single scan file, pass them in "
        "'landmarks', or use Fully-Automated mode to have them predicted"
    )


def choose_reference(models_root: str, modality: str, requested: list) -> str:
    """The reference bundle inside `models_root`, chosen by what it carries.

    There is no choice here for a clinician to make, and that is the point. A
    reference defines the target frame through what it CARRIES, and the two
    published CBCT bundles carry disjoint landmark sets -- Frankfurt Horizontal
    + Midsagittal has Ba/S/N/RPo/LPo/ROr/LOr, Occlusal + Midsagittal has
    ANS/IF/PNS/UL6O/UR1O/UR6O. So the selection already says which one applies;
    asking again could only produce a pairing that fails every patient
    separately, which is the failure `_check_selection_against_reference` exists
    to explain.

    The original extension asked for a FOLDER -- somewhere on the clinician's
    own disk, with a button offering to download one of the two. Here the
    bundles are on the server, so the folder question has no answer left to
    give.

    Each modality recognises its own shape, the way every engine in this project
    recognises its own weights: a CBCT reference carries markups, an IOS one
    carries labelled surfaces. A bundle of neither -- ALI's `.pth` files sit in
    the same folder -- is not a reference and is never offered.
    """
    markups_bundles, surface_bundles = {}, []
    for name in sorted(os.listdir(models_root)):
        directory = os.path.join(models_root, name)
        if not os.path.isdir(directory):
            continue
        # Surfaces decide, not the absence of markups: the published intraoral
        # bundle carries BOTH -- `Upper_gold.vtk` beside `Upper_gold.json` --
        # and classifying it by its markups would file the one IOS reference
        # under CBCT. A CBCT bundle carries markups beside a VOLUME, never a
        # mesh, so "has a mesh" separates them with nothing left over.
        if _holds_surfaces(directory):
            surface_bundles.append(name)
            continue
        landmarks = _reference_landmarks(directory)
        if landmarks:
            markups_bundles[name] = landmarks

    if modality == catalogs.MODALITY_IOS:
        return _only_one(models_root, surface_bundles, "intraoral")

    wanted = set(requested)
    fits = sorted(name for name, labels in markups_bundles.items()
                  if wanted and wanted <= set(labels))
    if len(fits) == 1:
        return os.path.join(models_root, fits[0])
    if not markups_bundles:
        raise ToolInputError(
            "No CBCT reference bundle on this server. Stage one with "
            "`setup-models.sh --tool ASO`."
        )
    offered = "; ".join(
        "'{}' carries {}".format(name, ", ".join(sorted(labels)))
        for name, labels in sorted(markups_bundles.items())
    )
    if not fits:
        raise ToolInputError(
            "No reference on this server carries every landmark you selected "
            "({}). {}. Select landmarks one of them supports.".format(
                ", ".join(sorted(wanted)) or "none", offered)
        )
    raise ToolInputError(
        "Several references carry the landmarks you selected, so which target "
        "frame to orient onto is ambiguous: {}. Name one in 'reference'.".format(
            ", ".join(fits))
    )


def _only_one(models_root: str, names: list, what: str) -> str:
    """The single bundle of its kind, or a refusal naming what was found."""
    if len(names) == 1:
        return os.path.join(models_root, names[0])
    if not names:
        raise ToolInputError(
            "No {} reference bundle on this server. Stage one with "
            "`setup-models.sh --tool ASO`.".format(what)
        )
    raise ToolInputError(
        "Several {} reference bundles are staged, so which one to orient onto "
        "is ambiguous: {}. Name one in 'reference'.".format(what, ", ".join(sorted(names)))
    )


def _reference_landmarks(directory: str) -> list:
    """The labels the first markups file in `directory` holds, or []."""
    for where, _subdirs, names in sorted(os.walk(directory)):
        for name in sorted(names):
            if markups.is_markups_file(name) and not name.startswith("."):
                try:
                    landmarks = markups.load_landmarks(os.path.join(where, name))
                except Exception:  # noqa: BLE001 - unreadable is "not a bundle"
                    continue
                if landmarks:
                    return list(landmarks)
    return []


def _is_reference_bundle(directory: str) -> bool:
    """Whether this directory IS a reference, rather than the folder of them.

    Answered on the TOP LEVEL only, and that is the whole trick: a bundle keeps
    its markups or its meshes right there, while `DATA/ASO/models/` keeps only
    directories. Asked recursively the models folder would look like a bundle,
    because one of its children is.
    """
    try:
        names = os.listdir(directory)
    except OSError:
        return False
    return any(
        markups.is_markups_file(name) or name.lower().endswith((".vtk", ".vtp", ".stl"))
        for name in names
        if os.path.isfile(os.path.join(directory, name))
    )


def _holds_surfaces(directory: str) -> bool:
    """Whether anything under `directory` is a surface mesh.

    What an IOS reference is: the engine takes tooth centroids off a labelled
    mesh rather than reading landmarks from a file, so a bundle carrying no
    markups at all is exactly right there and would be nothing anywhere else.
    """
    for _where, _subdirs, names in os.walk(directory):
        if any(name.lower().endswith((".vtk", ".vtp", ".stl")) for name in names):
            return True
    return False


def _check_selection_against_reference(requested: list, reference_landmarks: dict) -> None:
    """Refuse a selection the reference cannot support, before reading a scan.

    A reference defines the target frame through the landmarks it carries, and
    the two published bundles carry DISJOINT sets: Frankfurt Horizontal +
    Midsagittal has Ba/S/N/RPo/LPo/ROr/LOr, Occlusal + Midsagittal has
    ANS/IF/PNS/UL6O/UR1O/UR6O. The schema's defaults are the first set, so
    picking the second one without changing the selection means every landmark
    is dropped as "not in the reference" and every patient fails separately with
    "0 usable landmarks" -- a batch of forty identical failures for one wrong
    choice made in one place.

    Nothing knows which reference will be picked when the schema is published,
    but both are known the moment the call arrives. So it says so once, and
    names what the reference actually offers.
    """
    usable = [name for name in requested if name in reference_landmarks]
    if len(usable) >= cbct_pipeline.icp.MIN_LANDMARKS:
        return
    raise ToolInputError(
        f"Only {len(usable)} of the selected landmarks exist in this reference, and "
        f"{cbct_pipeline.icp.MIN_LANDMARKS} are needed. The reference defines: "
        f"{', '.join(sorted(reference_landmarks))}. Select landmarks from that list in "
        f"'cbct_landmarks', or pick a reference built on the ones you selected."
    )


def _check_ios(automation: str, teeth: list, types: list, jaws: list, occlusion: str) -> None:
    if not jaws:
        raise ToolInputError("Select at least one jaw in 'ios_jaws'.")
    if occlusion not in catalogs.DRIVING_JAW:
        raise ToolInputError(
            f"Unknown 'ios_occlusion' mode '{occlusion}'. Expected one of: "
            f"{', '.join(catalogs.DRIVING_JAW)}"
        )
    driving = catalogs.DRIVING_JAW[occlusion]
    if driving and driving not in jaws:
        raise ToolInputError(
            f"'ios_occlusion' asks the {driving} jaw to drive the other one, but "
            f"{driving} is not selected in 'ios_jaws'."
        )

    per_jaw = catalogs.split_by_jaw(teeth)
    # Only the jaws that are actually registered need teeth. With one jaw
    # driving the other, the driven one is moved by the driver's transform and
    # needs none of its own.
    registered = [driving] if driving else list(jaws)

    if automation == catalogs.AUTOMATION_FULLY:
        for jaw in registered:
            count = len(per_jaw[jaw])
            if count not in (3, 4):
                raise ToolInputError(
                    f"Fully-Automated IOS aligns each jaw from 3 or 4 teeth spread "
                    f"across the arch; {count} are selected for the {jaw} jaw in "
                    f"'ios_teeth'."
                )
        return

    if not types:
        raise ToolInputError(
            "Semi-Automated IOS registers on landmarks; select at least one in "
            "'ios_landmark_types'."
        )
    keys = catalogs.landmark_keys_by_jaw(teeth, types)
    for jaw in registered:
        if len(keys[jaw]) < 3:
            raise ToolInputError(
                f"Semi-Automated IOS registers on at least 3 landmarks per jaw; "
                f"the {jaw} jaw has {len(keys[jaw])} "
                f"({len(per_jaw[jaw])} teeth x {len(types)} landmark types)."
            )


# ---------------------------------------------------------------------------
# Calling the landmark tool
# ---------------------------------------------------------------------------

def _predict_landmarks(
    centered_root: str, landmark_model: str, requested: list, work_dir: str, sup
) -> dict:
    """Predict landmarks for every centred scan, through the supervisor.

    Returns `{patient key: {landmark: position}}`, keyed like
    `cbct.pipeline.discover`. The results never touch the caller's input tree.

    Asked for by NAME, not by region: ASO registers on seven landmarks
    straddling two of the landmark tool's regions, so asking by region would run
    58 agents to use 7 -- and one agent is a full two-scale walk of the volume.
    """
    if sup is not None and hasattr(sup, "progress"):
        sup.progress(0.2, f"predicting landmarks with {LANDMARK_TOOL}")

    # Only the landmarks are asked for. Which weights place them is that tool's
    # business, and it finds its own -- this used to compose a path into ALI's
    # data folder, which meant ASO holding a name for its neighbour's storage.
    # `landmark_model` survives as an OVERRIDE: a caller pinning a bundle is
    # recording which weights ran, and is obeyed.
    # No `output_dir`: where a supervised tool writes is the supervisor's, the
    # same way it is the server's over HTTP. Pointing it into this run's scratch
    # is what used to delete the predicted landmarks before anyone could ask to
    # keep them -- `keep_intermediate` collects from the supervisor's own folder.
    produced = sup.run(
        LANDMARK_TOOL,
        input=centered_root,
        landmarks=list(requested),
        **({"model": landmark_model} if landmark_model else {}),
    )
    # A tool returns a Path, or a dict of named ones. The landmark tool returns
    # its output directory; a dict is accepted so a future one naming its
    # outputs does not break the call.
    if isinstance(produced, dict):
        produced = next(iter(produced.values()))
    if not produced:
        raise ToolInputError(
            "'{}' returned no output directory, so there are no predicted "
            "landmarks to read.".format(LANDMARK_TOOL)
        )
    return _collect(str(produced))


# A tool's own run report sits beside its results and is NOT a markups file --
# but `is_markups_file` accepts a bare `.json`, because some markups files are
# written that way. Without this the report is probed on every fully-automated
# run and skipped with a warning that reads like data loss.
_NOT_LANDMARKS = ("run_report.json", REPORT_NAME)


def _collect(output_dir: str) -> dict:
    """Merge every markups file the landmark tool produced, per patient.

    Keys mirror `cbct.pipeline.discover`: the patient's path relative to the
    results root. The landmark tool may write one file per landmark GROUP, so
    several files can belong to one patient and are merged.
    """
    from . import markups

    predictions: dict = {}
    for directory, _, file_names in os.walk(output_dir):
        relative = os.path.relpath(directory, output_dir)
        prefix = "" if relative == "." else relative
        for file_name in sorted(file_names):
            if file_name in _NOT_LANDMARKS:
                continue
            if not markups.is_markups_file(file_name) or file_name.startswith("."):
                continue
            key = os.path.join(prefix, cbct_pipeline.patient_stem(file_name))
            try:
                found = markups.load_landmarks(os.path.join(directory, file_name))
            except (ValueError, OSError) as exc:
                # No name and no message: these files are named after the
                # caller's scans, and the ValueError `markups.load_landmarks`
                # raises quotes that name. There is no position to give either
                # -- this is a walk of a tree, not a pass over a batch -- so
                # the failure's type is the whole diagnosis.
                logger.warning(
                    "Skipping a landmark file the landmark tool wrote: %s",
                    type(exc).__name__,
                )
                continue
            predictions.setdefault(key, {}).update(found)
    return predictions


# ---------------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------------

def _run_cbct(
    input_root, reference_root, automation, requested, landmarks_path, landmark_model,
    dicom_input, output_dir, work_dir, suffix, max_triplets, seed, report, sup,
) -> None:
    # Asked of the DATA, not of the caller. DICOM slices routinely carry no
    # extension, so a clinician could not tell from a file name either -- and
    # answering wrong produced a run that failed for a reason nobody could see.
    # `dicom_input` remains an OVERRIDE for a caller who knows better than the
    # detector, which is why it stays in the signature and leaves the panel.
    if dicom_input or dicom.holds_a_series(input_root):
        input_root = dicom.convert_tree(input_root, os.path.join(work_dir, "dicom_nifti"))

    reference_landmarks = cbct_pipeline.load_reference(reference_root)
    report["requested_landmarks"] = list(requested)
    report["reference_landmarks"] = sorted(reference_landmarks)

    patients = cbct_pipeline.discover(input_root, suffix)
    if not patients:
        raise ToolInputError(
            "No CBCT scan found in the input. Expected one of "
            f"{', '.join(cbct_pipeline.SCAN_EXTENSIONS)}, sent as a file or a folder."
        )

    # After discovery, before a single scan is read: with nothing to orient,
    # "there are no scans here" is the useful answer, and complaining about the
    # landmark selection first would bury it.
    _check_selection_against_reference(requested, reference_landmarks)

    # A supplied landmark folder makes a fully-automated run behave like a
    # semi-automated one: the points are the caller's, wherever they came from.
    # That is what lets this tool be used standalone, with no supervisor and no
    # repository checkout -- see README.md.
    supplied = _as_directory(landmarks_path, os.path.join(work_dir, "landmarks_in")) \
        if landmarks_path else None
    by_patient = _collect(supplied) if supplied else None

    # Decided PER PATIENT, from what each one came with. A landmark file beside
    # a scan, or in the folder the caller supplied, is landmarks to register on;
    # a scan with none is one to predict for. The caller used to say which, and
    # saying it wrong was silent -- a fully-automated run over landmarks already
    # on disk predicted them again, and a semi-automated one over scans with
    # none failed patient by patient with "no landmarks".
    #
    # `automation` survives as an OVERRIDE: writing Fully-Automated when
    # landmarks exist is asking for them to be ignored, which is a legitimate
    # thing to want and is now the only thing this argument does.
    forced = automation == catalogs.AUTOMATION_FULLY

    def _needs_prediction(key):
        # A folder the caller SUPPLIED wins over everything, including the mode.
        # That precedence is what lets this tool run standalone -- predict the
        # landmarks yourself, pass the folder, no supervisor needed -- and
        # reversing it would make `Fully-Automated` throw away points a caller
        # went out of their way to hand over.
        if by_patient is not None:
            return not by_patient.get(key)
        if forced:
            return True
        return not patients[key]["markups"]

    # Only where predicting is POSSIBLE. With no supervisor there is no landmark
    # tool to ask, and a patient with no landmarks then fails on its own terms --
    # "no landmarks for this patient", which is true and actionable -- instead of
    # taking the batch down inside a call that was never going to work.
    to_predict = ({key for key in patients if _needs_prediction(key)}
                  if sup is not None else set())
    fully = bool(to_predict)
    report["landmark_source"] = (
        LANDMARK_TOOL if len(to_predict) == len(patients)
        else ("supplied" if supplied else "alongside the scans") if not to_predict
        else "mixed"
    )

    # Landmark files that matched no scan. Collected even on a run that
    # succeeds: "39 of your 40 patients were oriented" and "the 40th one's
    # landmarks are in the folder under a name nothing matched" are the same
    # sentence, and only this makes the second half sayable.
    # Landmarks that matched no scan. A supplied folder holds landmarks only,
    # so `orphan_markups` -- which calls anything without a scan beside it an
    # orphan -- would call all of them orphans; there, matching is by patient
    # key against the scans that were actually found.
    if fully:
        orphans = []
    elif supplied:
        orphans = sorted(set(_collect(supplied)) - set(patients))
    else:
        orphans = sorted(
            os.path.basename(path)
            for paths in cbct_pipeline.orphan_markups(input_root, suffix).values()
            for path in paths
        )
    report["unmatched_markups"] = orphans

    centered_root = os.path.join(work_dir, "centered")

    # Phase 1 -- recentre. The landmark tool runs on centred scans (that is what
    # the Slicer chain did, PRE_ASO_CBCT before ALI_CBCT), so in fully-automated
    # mode they have to reach disk before it is called.
    prepared = {}
    # Three phases share the bar, and each is given the slice it occupies so
    # the second does not send it back to zero. The boundaries are where the
    # landmark waypoint below already put them: recentring up to 0.2, the
    # landmark tool from there, registration on the tail. What is exact is the
    # counter in the message; the split between phases is a weighting.
    for index, (key, entry) in enumerate(sorted(patients.items()), start=1):
        progress.report(index, len(patients), "centring scan", end=0.2)
        _, extension = split_scan_extension(os.path.basename(entry["scan"]))
        destination = (
            os.path.join(
                centered_root,
                f"{key}{cbct_pipeline.compressed_extension(extension)}",
            )
            if key in to_predict
            else None
        )
        try:
            image, translation = cbct_pipeline.prepare(entry["scan"], destination)
        except RuntimeError as exc:
            report["cases"][key] = {"status": "failed", "reason": str(exc)}
            continue
        prepared[key] = {
            # Kept in RAM only when it is used next -- a predicted patient's
            # scan is read from disk by the landmark tool instead.
            "image": None if key in to_predict else image,
            "translation": translation,
            "extension": extension,
            "centered_path": destination,
            "landmarks": None,
        }

    # Phase 2 -- landmarks, either the caller's (moved into the centred space)
    # or predicted ones (already in it, because the tool ran on the centred
    # scans).
    # ONE call for every scan that needs predicting, not one per patient: the
    # landmark tool loads its agents once and walks a folder, so a cohort costs
    # what a cohort costs rather than N times a single scan.
    predictions = (
        _predict_landmarks(centered_root, landmark_model, requested, work_dir, sup)
        if to_predict else {}
    )
    for key, entry in prepared.items():
        if key in to_predict:
            # Already in the centred space: the tool ran on the centred scans.
            entry["landmarks"] = predictions.get(key, {})
            continue
        # The caller's, moved into the centred space -- they describe the
        # ORIGINAL volume, wherever they came from, and the registration
        # compares them against a centred one.
        found = (
            by_patient.get(key, {})
            if by_patient is not None
            else cbct_pipeline.load_landmarks(patients[key]["markups"])
        )
        entry["landmarks"] = cbct_pipeline.center_landmarks(found, entry["translation"])

    # Phase 3 -- register and write. It starts where the landmark tool left
    # off, which is only where semi-automated ends its own phase 1: there is no
    # prediction between them, so the bar must not skip the slice it never used.
    registration_start = 0.6 if fully else 0.2
    for index, (key, entry) in enumerate(sorted(prepared.items()), start=1):
        progress.report(index, len(prepared), "orienting patient", start=registration_start)
        if not entry["landmarks"]:
            report["cases"][key] = {
                "status": "failed",
                "reason": (
                    "no predicted landmarks for this scan"
                    if fully
                    else _no_landmarks_reason(
                        key,
                        [] if supplied else patients[key]["markups"],
                        orphans,
                    )
                ),
            }
            continue
        image = entry["image"]
        if image is None:
            import SimpleITK as sitk  # local: only the fully-automated path re-reads

            image = sitk.ReadImage(entry["centered_path"])
        try:
            report["cases"][key] = cbct_pipeline.orient_patient(
                centered=image,
                pre_transform=entry["translation"],
                source_landmarks=entry["landmarks"],
                reference_landmarks=reference_landmarks,
                requested=requested,
                output_dir=output_dir,
                relative_key=key,
                suffix=suffix,
                extension=entry["extension"],
                max_triplets=max_triplets,
                seed=seed,
            )
        except cbct_pipeline.icp.RegistrationError as exc:
            report["cases"][key] = {"status": "failed", "reason": str(exc)}
        entry["image"] = None  # a batch must not hold every volume in RAM


def _run_ios(
    input_root, reference_root, automation, teeth, landmark_types, jaws, occlusion,
    output_dir, suffix, max_triplets, seed, report,
) -> None:
    # Which form each jaw needs is decided per jaw now, so the reference is
    # loaded with both and each branch says what it is missing.
    reference = ios_pipeline.load_reference(reference_root)

    patients = ios_pipeline.discover(input_root, suffix)
    patients = {key: entry for key, entry in patients.items() if _has_surface(entry)}
    if not patients:
        raise ToolInputError(
            "No intra-oral mesh found in the input. Expected one of "
            f"{', '.join(ios_pipeline.surfaces.SURFACE_EXTENSIONS)}, named so a "
            f"token says which jaw it is (e.g. 'P1_U_Seg.vtk')."
        )

    report["requested_teeth"] = list(teeth)
    # Both are reported: which one a jaw used is decided per jaw, so a cohort
    # can legitimately have registered some on landmarks and some on centroids,
    # and a reader has to be able to see what was asked for either way.
    report["requested_landmark_types"] = list(landmark_types)

    selected_teeth = catalogs.split_by_jaw(teeth)
    landmark_keys = catalogs.landmark_keys_by_jaw(teeth, landmark_types)
    driving = catalogs.DRIVING_JAW[occlusion]

    # One cache for the whole cohort: the reference bundle is the same two
    # files for every patient, and it was being parsed once per jaw per
    # patient. A patient's own meshes are dropped when that patient is done,
    # inside orient_patient's own `finally`.
    cache = ios_pipeline.FileCache()

    for index, (key, entry) in enumerate(sorted(patients.items()), start=1):
        progress.report(index, len(patients), "orienting patient")
        report["cases"][key] = ios_pipeline.orient_patient(
            jaws=entry,
            reference=reference,
            automation=automation,
            selected_teeth=selected_teeth,
            landmark_keys=landmark_keys,
            wanted_jaws=jaws,
            driving_jaw=driving,
            output_dir=output_dir,
            relative_key=key,
            suffix=suffix,
            max_triplets=max_triplets,
            seed=seed,
            cache=cache,
        )

    # Unconditional now. It only fires when EVERY jaw failed for want of tooth
    # labels, which can only happen on the centroid path -- there is no mode to
    # test for any more.
    _reject_if_nothing_was_labelled(report["cases"])


def _has_surface(entry: dict) -> bool:
    return any(jaw["surface"] for jaw in entry.values())


# What `ios_pipeline.segment_unlabelled` puts in the reason, matched to tell a
# wrong-mode request apart from a bad mesh.
_UNLABELLED = "carries no tooth labels"


def _reject_if_nothing_was_labelled(patients: dict) -> None:
    """Turn "not one of your meshes is segmented" into an input error.

    The distinction matters, and only these two cases are treated differently.
    No labelled mesh at all is a cohort that cannot be oriented at all, and
    saying so is more use than an empty output folder. Some meshes labelled and others not
    is a data problem: those are recorded per patient and the rest of the batch
    is kept, because "one of your forty meshes was bad" is not a reason to
    return nothing.

    Checked after the loop rather than before it, so no mesh is read twice: an
    unlabelled one fails the moment its label array is probed, before any
    registration work.
    """
    reasons = [
        jaw.get("reason", "")
        for entry in patients.values()
        for jaw in entry.get("jaws", {}).values()
    ]
    if not reasons or not all(_UNLABELLED in reason for reason in reasons):
        return
    raise ToolInputError(
        "None of these meshes carries landmarks, so each was oriented by its "
        "tooth centroids -- and none carries a per-point label array named one "
        f"of {', '.join(ios_pipeline.markups.LABEL_ARRAY_NAMES)} either, so there "
        "are no centroids to take. Run 'Crown_Seg' over them first, or put a "
        "landmark file beside each mesh."
    )


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------

def _as_directory(path: str, destination: str) -> str:
    """A directory holding the input, whatever shape it arrived in.

    No archive is unpacked here: the server extracts a `.zip` before `run()` is
    called, with the bomb cap and the single-root strip this function used to
    apply itself.

    A single file is still linked into a directory of its own rather than used
    from where it landed, because its neighbours are not necessarily part of the
    same input -- a caller may well have staged the scan and the reference side
    by side.
    """
    path = str(path)
    if os.path.isdir(path):
        return path
    if not os.path.exists(path):
        raise ToolInputError(f"Path not found: {os.path.basename(path)}")

    os.makedirs(destination, exist_ok=True)
    linked = os.path.join(destination, os.path.basename(path))
    try:
        os.link(path, linked)
    except OSError:
        shutil.copy2(path, linked)
    return destination


def _summarize(report: dict) -> None:
    statuses = [entry.get("status") for entry in report["cases"].values()]
    report["summary"] = {
        "cases": len(statuses),
        "oriented": statuses.count("ok"),
        "failed": statuses.count("failed"),
    }
    logger.info(
        "%s %s: %d/%d oriented",
        report["modality"],
        report["automation"],
        report["summary"]["oriented"],
        report["summary"]["cases"],
    )
