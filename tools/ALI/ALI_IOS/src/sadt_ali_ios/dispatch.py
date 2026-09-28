"""Everything ALI_IOS does around inference: discovery, the landmark selection
and the run report. `engine.py` only has to know how to place landmarks.

ALI used to be one tool choosing an engine from the data. Splitting it in two
moved that question out of the run and into the request: this tool is the
intraoral engine, and an input holding CBCT volumes is refused by name rather
than half-processed.

No DICOM here, unlike ALI_CBCT: a DICOM series is a volume by definition, so
detecting one is that tool's business. It also needs itk, which this
environment does not carry.

**A mesh with no tooth labels is segmented mid-run rather than refusing the
batch.** The landmark networks are pointed at teeth, so a mesh that carries no
label array is nothing they can be asked about -- and a clinician sending a
folder of intraoral scans has no reason to know that `Crown_Seg` exists, let
alone which half of their cohort has already been through it. So the meshes
that already work are processed FIRST and their landmarks are on disk before a
second of segmentation is spent; the rest go through `Crown_Seg` and follow.
That ordering is the whole design: a run cancelled, timed out or killed half
way still leaves the cheap results behind.

It is the exception CONTRIBUTING.md allows, for the reason it allows it: the
server cannot chain the two beforehand without segmenting the meshes that did
not need it and without losing that ordering. With NO supervisor -- a direct
call, a checkout, an older server -- nothing changes: the batch is refused
exactly as it was, by the check that names the tool to run.
"""

import json
import logging
import os
import shutil
import time

from sadt_ali_common import markups
from sadt_ali_common.discovery import (
    IOS,
    SURFACE_EXTENSIONS,
    VOLUME_EXTENSIONS,
    WORK_DIRNAME,
    classify,
    keyed,
)

from .errors import ToolInputError, ToolUnavailableError
from . import catalog as ios_catalog

logger = logging.getLogger(__name__)


REPORT_NAME = "run_report.json"

# The tool asked for tooth labels when a mesh arrives without them. A string,
# not a dynamic attribute on the supervisor: a typo in a string is greppable and
# `describe.py` reads it out of this file to publish the schema's `calls`, where
# `sup.Crown_Seg(...)` is an AttributeError forty minutes into a batch.
CROWN_TOOL = "Crown_Seg"

# Under the run's own work dir: what was handed to Crown_Seg, what it handed
# back, and the meshes renamed for the engine to read. All three go with the
# work dir when the run ends.
TO_SEGMENT_DIRNAME = "to_segment"
SEGMENTED_DIRNAME = "segmented"


class Input:
    """What the input turned out to hold, and how to run it.

    `scans` is a list of `(absolute path, key)` pairs. The key is the path
    relative to the input root and is what identifies a scan everywhere
    afterwards -- in the report and in the output tree. Keying by BASE NAME,
    as the original did, meant two patients called `scan.nii.gz` in different
    subfolders silently overwrote each other twice over: once in the working
    dictionary, once in the flat output folder.
    """

    def __init__(self, mode: str, scans: list):
        self.mode = mode
        self.scans = scans



# ---------------------------------------------------------------------------
# Discovery and mode detection
# ---------------------------------------------------------------------------


def detect(input_path: str, work_dir: str) -> Input:
    """List the intraoral surfaces to process, and refuse anything else.

    Before the split this decided WHICH engine ran. It no longer does: this
    tool is the intraoral engine, so the question is only whether the caller
    sent surfaces. Volumes are named rather than ignored -- silently processing
    the meshes of a mixed folder and dropping the scans is the failure that
    looks like success.

    `work_dir` is unused here and kept so both tools' dispatchers have the same
    shape; ALI_CBCT needs it to convert DICOM.

    No archive is unpacked here. The server extracts a `.zip` before `run()`
    is called -- with the bomb cap and the single-root strip this function used
    to apply itself -- so what arrives is always a real file or directory.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path not found: {input_path}")

    root = input_path

    if os.path.isfile(root):
        lower = root.lower()
        if lower.endswith(SURFACE_EXTENSIONS):
            return Input(IOS, [(root, os.path.basename(root))])
        if lower.endswith(VOLUME_EXTENSIONS):
            raise ToolInputError(
                f"'{os.path.basename(root)}' is a CBCT volume. This tool places "
                f"landmarks on intraoral surfaces; run ALI_CBCT on volumes."
            )
        raise ToolInputError(
            f"'{os.path.basename(root)}' is not an intraoral surface "
            f"({', '.join(SURFACE_EXTENSIONS)})."
        )

    volumes, surfaces = classify(root)

    if volumes and not surfaces:
        raise ToolInputError(
            f"This input holds {len(volumes)} CBCT scan(s) and no intraoral surface. "
            f"Run ALI_CBCT on volumes."
        )
    if volumes:
        raise ToolInputError(
            f"This input mixes {len(volumes)} CBCT scan(s) and {len(surfaces)} intraoral "
            f"surface(s). Send them as two batches, to ALI_CBCT and ALI_IOS respectively."
        )

    if surfaces:
        return Input(IOS, keyed(surfaces, root))

    raise ToolInputError(
        f"No intraoral surface ({', '.join(SURFACE_EXTENSIONS)}) found in the input."
    )


# ---------------------------------------------------------------------------
# Tooth labels, on the meshes that arrived without them
# ---------------------------------------------------------------------------

def split_by_labels(meshes: list) -> tuple:
    """`(labelled, unlabelled)`, both `(path, key)` lists, order preserved.

    `engine.require_labels` asks the same question of the whole batch and
    refuses it; this asks it of each mesh, which is what lets the two halves be
    treated differently. Reading a mesh's point-data array NAMES is what it
    costs -- never their values, so nothing about the patient is read to decide
    this.
    """
    from . import surface

    labelled, unlabelled = [], []
    for path, key in meshes:
        try:
            found = surface.label_array_name(surface.read_surface(path)) is not None
        except Exception:  # noqa: BLE001 - unreadable is not labelled; see below
            # A mesh this cannot even read is put with the ones needing labels
            # rather than raised here. One corrupt file in a cohort of forty
            # then costs that file -- `Crown_Seg` fails it alone and says why,
            # and the run returns landmarks for the other thirty-nine. Raising
            # would lose them all, on mesh 3 of 40, before a landmark was
            # placed.
            logger.warning("ALI_IOS: a surface could not be read; sending it to '%s'",
                           CROWN_TOOL)
            found = False
        (labelled if found else unlabelled).append((path, key))
    return labelled, unlabelled


def _stage(source: str, destination: str) -> None:
    """Put `source` where Crown_Seg will find it, without copying it if we can.

    An intraoral scan is tens of megabytes and a cohort is forty of them, so the
    meshes are linked rather than copied. Copying is the fallback for a
    filesystem with no symlinks, not the plan. Absolute target: the callee runs
    with its own working directory.
    """
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    try:
        os.symlink(os.path.abspath(source), destination)
    except OSError:  # no symlink support, or one already there
        shutil.copy2(source, destination)


def _segment(sup, meshes: list, model_path: str, device: str, work_dir: str) -> tuple:
    """Ask `Crown_Seg` for tooth labels; return `(labelled, {key: reason})`.

    `meshes` and the first return value are both `(path, key)` pairs, and the
    KEY is what survives the round trip: a mesh comes back as a different file
    in a different tree, and the report, the output tree and the markups file
    still have to name the patient's own scan.

    The meshes are staged UNDER THEIR KEYS so Crown_Seg's own report -- which is
    keyed by the path relative to what it was given -- comes back keyed the same
    way. Reading that report is what makes the seam narrow: this knows the
    callee's published schema and the report it documents, and nothing else
    about it.

    `model` is the directory this tool was handed, passed straight on. The
    server fills a hosted-model argument with the whole of `DATA/<tool>/models/`
    and each engine recognises its own weights inside; that is exactly what
    Crown_Seg's `find_checkpoint` now does, so the same directory answers for
    both. A caller who PINNED a bundle instead gets that bundle passed on, and
    if the crown checkpoint is not in it Crown_Seg says so -- reaching outside
    the directory the server named would be this tool resolving paths, which is
    the one thing it must not do.

    Everything is written under this run's own work dir rather than under
    `sup.tmp`: it is removed on every path out of `identify`, crash included,
    and it puts the labelled mesh on the same filesystem as the name it is
    renamed onto below -- which is what makes that a rename and not a copy.
    """
    staged_root = os.path.join(work_dir, TO_SEGMENT_DIRNAME)
    for path, key in meshes:
        _stage(path, os.path.join(staged_root, key))

    # No `output_dir`: where a supervised tool writes is the supervisor's, the
    # same way it is the server's over HTTP. Pointing it into this run's scratch
    # is what used to delete the labelled meshes before anyone could ask to keep
    # them -- `keep_intermediate` collects from the supervisor's own folder.
    produced = sup.run(
        CROWN_TOOL,
        meshes=staged_root,
        model=model_path,
        # This deployment's device, not Crown_Seg's own default: a CPU-only
        # server would otherwise have every supervised call ask for CUDA and
        # fall back with a warning.
        device=device,
    )
    # A tool returns a Path, or a dict of named ones.
    if isinstance(produced, dict):
        produced = next(iter(produced.values()))

    records = _crown_records(str(produced))
    labelled, failed = [], {}
    for _path, key in meshes:
        record = records.get(key) or {}
        produced_files = record.get("produced") or []
        output = produced_files[0] if produced_files else None
        if not output or not os.path.isfile(output):
            # One mesh the segmentation could not label costs that mesh, never
            # the batch: the rest of the cohort is still worth an hour of GPU.
            failed[key] = record.get("error") or (
                f"'{CROWN_TOOL}' produced no labelled mesh for it"
            )
            continue
        labelled.append((_rename_to_input(output, key, work_dir), key))

    # Trust, then verify. A mesh reported as segmented that carries no label
    # array is a contradiction, and the engine's own check would answer it by
    # refusing the WHOLE batch with a message telling the caller to run
    # Crown_Seg -- which is exactly what just happened. Demoted to that one
    # mesh's own failure instead, so the rest of the cohort still runs.
    labelled, without_labels = split_by_labels(labelled)
    for _path, key in without_labels:
        failed[key] = f"'{CROWN_TOOL}' returned a mesh carrying no tooth-label array"
    return labelled, failed


def _crown_records(output_dir: str) -> dict:
    """Crown_Seg's per-mesh report, keyed as this tool keys its scans.

    Missing or unreadable is not fatal here -- every mesh then reports "no
    labelled mesh", which is true and is what the caller needs to know. A
    KeyError in the middle of reading a sibling's report would say nothing at
    all about the meshes that did work.
    """
    try:
        with open(os.path.join(output_dir, REPORT_NAME), encoding="utf-8") as handle:
            return json.load(handle).get("cases") or {}
    except (OSError, ValueError):
        logger.warning("'%s' wrote no readable run report", CROWN_TOOL)
        return {}


def _rename_to_input(produced: str, key: str, work_dir: str) -> str:
    """Move a labelled mesh back onto the name its input had.

    Crown_Seg names its output `<stem>_Seg.vtk`, and the engine names the
    markups file after the mesh it read -- so without this, a mesh segmented on
    the fly produces `arch_Seg_lm_Pred.mrk.json` where the same mesh sent
    already labelled produces `arch_lm_Pred.mrk.json`. ASO and AREG pair
    landmarks to scans BY THAT STEM, so the two paths have to be
    indistinguishable in the output; how a mesh got its labels is the report's
    to say, not the file name's.

    `.vtk` whatever the input was: that is what Crown_Seg writes, and it is what
    the engine reads back.
    """
    destination = os.path.join(
        work_dir, SEGMENTED_DIRNAME, os.path.dirname(key),
        os.path.splitext(os.path.basename(key))[0] + ".vtk",
    )
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    # Both sides live under the work dir, so this is a rename and not a copy.
    os.replace(produced, destination)
    return destination


def _unprocessed(key: str, reason: str) -> dict:
    """A scan record for a mesh that never reached the engine.

    Same shape as the engine's own records, so a client reading the report has
    one thing to read rather than two.
    """
    return {
        "input": os.path.basename(key),
        "status": "failed",
        "error": reason,
        "landmarks_found": [],
        "landmarks_failed": {},
        "jaws_without_model": {},
        "produced": [],
        "duration_seconds": 0.0,
    }


def _merge(reports: list) -> dict:
    """The two passes' reports as one.

    Only `cases` is genuinely merged. Everything else -- the mode, the device,
    the networks the bundle could serve, the checkpoints it could not read --
    is a property of the BUNDLE and the request, identical in both passes, so
    it is taken from the first rather than combined into a list that would
    always hold the same value twice. `summary` and `duration_seconds` are
    recomputed by the caller over the whole run.
    """
    merged = dict(reports[0])
    cases = {}
    for report in reports:
        cases.update(report["cases"])
    merged["cases"] = cases
    return merged


# ---------------------------------------------------------------------------
# Narrowing a run to named landmarks
# ---------------------------------------------------------------------------

def _keep_only(report: dict, landmarks) -> None:
    """Drop every landmark the caller did not name from what was written.

    This is deliberately AFTER the run rather than inside the engine. One
    forward pass covers a (network, jaw, tooth) and emits every channel of that
    network at once, so predicting UR1O predicts UR1MB and UR1DB with it at no
    extra cost: naming landmarks narrows what is WRITTEN, never what is
    computed. What a selection does save is whole passes, and that is decided
    before the run by `catalog.networks_for` -- by the time the engine is
    called there is nothing left for it to narrow.

    Each file is rewritten through the writer that produced it, so a filtered
    file is indistinguishable from one a run of exactly this selection would
    have written, control-point ids included.
    """
    wanted = set(landmarks)
    for record in report.get("cases", {}).values():
        kept_files = []
        for path in record.get("produced", []):
            if _filter_markups(path, wanted):
                kept_files.append(path)
            else:
                # Nothing selected survives on this scan -- an upper-arch
                # selection against a mandible, say. An empty markups file
                # opens in Slicer as an empty node, which reads as a run that
                # went wrong; the scan's own record is where "nothing here"
                # belongs.
                os.remove(path)
        record["produced"] = kept_files
        record["landmarks_found"] = [
            label for label in record.get("landmarks_found", []) if label in wanted
        ]
        degraded = record.get("landmarks_degraded")
        if degraded:
            record["landmarks_degraded"] = {
                label: note for label, note in degraded.items() if label in wanted
            }


def _filter_markups(path: str, wanted: set) -> list:
    """Rewrite one markups file keeping only `wanted`; return what it kept."""
    with open(path, encoding="utf-8") as handle:
        content = json.load(handle)

    points = [
        point
        for point in content["markups"][0]["controlPoints"]
        if point["label"] in wanted
    ]
    if not points:
        return []

    markups.write(
        {point["label"]: point["position"] for point in points},
        path,
        # A caveat belongs to its point and travels with it: dropping the
        # neighbours it was written beside must not silently make a degraded
        # landmark look like a clean one.
        {
            point["label"]: point["description"]
            for point in points
            if point.get("description")
        },
    )
    return [point["label"] for point in points]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def identify(
    input_path: str,
    model_path: str,
    output_dir: str,
    ios_networks=None,
    landmarks=None,
    prediction_ID: str = "Pred",
    device: str = "cuda",
    sup=None,
) -> dict:
    """Place landmarks on whatever this input holds; return the run report.

    Everything is written under `output_dir`: one markups file per scan, in the
    input's own tree, plus `run_report.json`. Intermediates go in
    `<output_dir>/.ali_work/` and are removed before returning.

    In two passes when the batch is mixed: the meshes that already carry tooth
    labels, then the ones `Crown_Seg` labelled for them. See the module
    docstring for why that order and not the other.
    """
    started_at = time.monotonic()

    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    work_dir = os.path.join(output_dir, WORK_DIRNAME)
    os.makedirs(work_dir, exist_ok=True)

    prediction_ID = (prediction_ID or "Pred").strip() or "Pred"

    try:
        # Walking a cohort and converting DICOM are minutes of work on a large
        # batch, and used to happen in complete silence -- so a run looked hung
        # before it had even started. Counts only, never a file name.
        logger.info("ALI_IOS: inspecting the input")
        detected = detect(input_path, work_dir)
        logger.info("ALI_IOS: %d surface(s)", len(detected.scans))

        # Imported here, not at module level: the engine pulls torch and
        # pytorch3d, and CI imports this package on every PR to publish the
        # schema. That must not cost a CUDA stack.
        from . import engine as ios_engine

        # An explicit landmark list REPLACES the families rather than narrowing
        # them, exactly as it does in ALI_CBCT: a caller naming the points it
        # needs must not also have to leave the right families ticked, and
        # narrowing would silently drop landmarks for one that set both.
        selected, unknown = ios_catalog.resolve_landmarks(landmarks)
        if landmarks and not selected:
            raise ToolInputError(
                f"None of the landmarks named under 'landmarks' exists: "
                f"{', '.join(sorted(set(landmarks)))}. This tool places "
                f"{len(ios_catalog.LANDMARKS)} landmarks, e.g. "
                f"{', '.join(ios_catalog.LANDMARKS[:3])}."
            )

        if selected:
            networks = ios_catalog.networks_for(selected)
        else:
            networks = ios_catalog.network_codes(ios_networks)
            if not networks:
                # The cross-argument rule the schema cannot express.
                raise ToolInputError(
                    f"Select at least one landmark family under 'networks' "
                    f"({', '.join(ios_catalog.NETWORK_NAMES)}), or name the "
                    f"points you want under 'landmarks'."
                )

        if sup is None:
            # No way to reach another tool, so nothing changes: the batch is
            # refused exactly as it always was, by the check that names the tool
            # to run. Asked of the WHOLE batch before it is split, so the
            # message counts the whole batch and every mesh is read once.
            ios_engine.require_labels(detected.scans)
            labelled, unlabelled = detected.scans, []
        else:
            labelled, unlabelled = split_by_labels(detected.scans)
            if unlabelled:
                logger.info(
                    "ALI_IOS: %d of %d surface(s) carry no tooth labels; the labelled "
                    "ones run first, the rest go through '%s'",
                    len(unlabelled), len(detected.scans), CROWN_TOOL,
                )

        pass_arguments = dict(
            model_path=model_path,
            networks=networks,
            prediction_ID=prediction_ID,
            output_dir=output_dir,
            device=device,
        )
        reports, errors = [], []

        def run_pass(meshes: list) -> None:
            """One engine pass, allowed to fail without sinking the other.

            `predict_landmarks` raises when NO mesh of the batch it was given
            produced a landmark. With two passes that must not end the run: a
            cohort whose pre-labelled half is unusable still has landmarks to
            return for the half segmented here, and the other way round. It is
            recorded and re-raised below only if nothing at all was produced.

            The two typed errors are not per-batch and are re-raised at once: a
            missing pytorch3d is a property of the venv, and a bundle with no
            weights for the selected networks a property of the request. Both
            give the identical answer for the other pass, so running it would
            cost a second failure and tell nobody anything.
            """
            try:
                reports.append(ios_engine.predict_landmarks(meshes=meshes, **pass_arguments))
            except (ToolUnavailableError, ToolInputError):
                raise
            except Exception as exc:  # noqa: BLE001 - recorded, re-raised below
                logger.exception("ALI_IOS: no landmark on any of %d mesh(es)", len(meshes))
                errors.append(exc)

        # The labelled meshes FIRST. They need nothing this run does not already
        # have, so their landmarks are on disk before a second of segmentation is
        # spent -- which is what a cancelled or timed-out run keeps.
        if labelled:
            run_pass(labelled)

        segmented, segmentation_failures = [], {}
        if unlabelled:
            # Before spending a segmentation on meshes nothing could then read.
            # `predict_landmarks` does this itself, but a batch where NOTHING
            # arrived labelled has not reached it yet -- so a venv without
            # pytorch3d would segment the whole cohort and only then say the
            # landmark engine was never installed. Idempotent: the pass below
            # calls it again.
            ios_engine.check_dependencies()
            if hasattr(sup, "progress"):
                # The share of the batch already behind us. The engine's own
                # per-mesh events restart at zero for the second pass, which it
                # cannot know it is (see `progress.report`'s start/end bounds,
                # which only the caller of the loop could pass).
                sup.progress(
                    len(labelled) / float(len(detected.scans)),
                    f"labelling {len(unlabelled)} mesh(es) with {CROWN_TOOL}",
                )
            try:
                segmented, segmentation_failures = _segment(
                    sup, unlabelled, model_path, device, work_dir
                )
            except Exception as exc:  # noqa: BLE001 - see below
                # The whole call failed -- no checkpoint in the bundle, the
                # engine's extra not installed there, the process killed. That
                # is not a reason to throw away the landmarks the first pass
                # already wrote, so it is reported against every mesh it was
                # asked about and the run ends with what it has.
                logger.exception("'%s' could not label %d mesh(es)", CROWN_TOOL, len(unlabelled))
                reason = f"{type(exc).__name__}: {exc}"
                segmentation_failures = {key: reason for _path, key in unlabelled}
            if segmented:
                run_pass(segmented)

        if not reports:
            if errors:
                raise errors[0]
            # Nothing carried labels and nothing could be given any, so the
            # segmentation is the whole story and its reason is what travels.
            raise RuntimeError(
                "ALI_IOS produced no landmarks: none of the {} mesh(es) carried tooth "
                "labels and '{}' could not label them. First reason: {}".format(
                    len(detected.scans), CROWN_TOOL,
                    next(iter(segmentation_failures.values()), "unknown"),
                )
            )

        report = _merge(reports)
        if selected:
            _keep_only(report, selected)
        # What drove the run, and only that: the families are left at their
        # default when landmarks were named, so reporting them would show a
        # selection the caller never made. `networks` above stays as the engine
        # wrote it -- it says which passes actually ran, which is a fact about
        # the run rather than a selection.
        for key, reason in sorted(segmentation_failures.items()):
            report["cases"][key] = _unprocessed(key, reason)
        processed = sum(1 for record in report["cases"].values() if record["status"] == "ok")
        report["summary"] = {
            "total": len(report["cases"]),
            "processed": processed,
            "failed": len(report["cases"]) - processed,
        }
        # Which meshes did not arrive ready, by key. Always present, so a client
        # reads one field rather than testing whether it exists: empty is the
        # normal case and means every mesh already carried its labels.
        report["segmented_on_the_fly"] = sorted(key for _path, key in segmented)
        report["landmarks_selected"] = list(selected)
        if unknown:
            # Named rather than dropped in silence: a stale client asking for a
            # landmark this tool no longer places gets a shorter answer than it
            # asked for, and nothing else would say so.
            report["landmarks_unknown"] = list(unknown)
    finally:
        # The intermediates are large -- converted DICOM, and every scan
        # preprocessed at two spacings. Removed whether or not the run
        # succeeded, and never from inside the output tree the caller keeps.
        shutil.rmtree(work_dir, ignore_errors=True)

    report["tool"] = "ALI_IOS"
    # So the report says which weights ran even when nobody read the argument.
    # `setdefault`: the engine names the bundle the checkpoints actually came
    # from, which is the better answer whenever it has one. This is the
    # fallback for a run that never reached the engine.
    report.setdefault("model_bundle", os.path.basename(str(model_path).rstrip(os.sep)))
    report["output_dir"] = output_dir
    report["duration_seconds"] = round(time.monotonic() - started_at, 2)

    # Named `run_report.json` because that is what the Slicer module reads to
    # tell "the model bundle has no such landmark" from "the agent did not
    # converge on this scan" -- two failures that look identical in the scene
    # and need opposite fixes.
    with open(os.path.join(output_dir, REPORT_NAME), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    logger.info(
        "ALI_IOS finished: %d/%d scan(s) in %.1fs",
        report["summary"]["processed"],
        report["summary"]["total"],
        report["duration_seconds"],
    )
    return report
