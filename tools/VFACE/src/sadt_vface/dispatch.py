"""Everything VFACE does around the measurements themselves.

The run is a chain, and the chain is what this file is:

    resample -> orient (per frame) -> segment (per frame) -> the second scan
             -> register (per region) -> landmarks -> measure -> classify

**The second scan is where the two studies part.** An asymmetry assessment has
no follow-up: it mirrors the patient's own oriented scan across the mid-sagittal
plane and registers the patient against themselves. A longitudinal study has a
real second timepoint, resampled and oriented into the same frames. Everything
after that step is identical, which is why one tool does both.

Three modes say how far down the chain the caller's scans already are, and each
one starts where the last leaves off:

    Full pipeline            raw CBCTs; everything below runs
    File already Oriented    scans already in the two frames; skips the
                             resample and the orientation
    File already Registered  scans and transforms already paired; measures only

The regions are worked in whichever FRAME they are defined in -- the mandible
against the cranial base, the maxilla against the occlusal plane -- and a
measurement of one region is never read in the other's frame.
"""

import json
import logging
import os
import shutil
import time

from . import aq3dc, catalogs, classify, features, landmarks as landmark_files
from . import discovery, progress, resample, tools
from .errors import ToolInputError

logger = logging.getLogger(__name__)

REPORT_NAME = "VFACE_report.json"
WORK_DIRNAME = ".vface_work"

MEASUREMENTS_DIRNAME = "Measurements"
CLASSIFICATION_DIRNAME = "Classification"
FEATURE_TABLE_NAME = "PostProcess_Measurements.xlsx"
CLASSIFICATION_NAME = "Classification.xlsx"

# How much of the bar each stage takes, relative to the others. A forty-patient
# run is an hour of other tools, and the share each takes is roughly what these
# say; a bar that jumped from 10% to 90% would be worse than none. Orienting and
# registering are the long stages -- ASO runs ALI_CBCT inside it, AREG_CBCT runs
# elastix per patient -- and measuring is arithmetic. A weighting, not a
# measurement.
#
# Only the stages a run actually makes are laid out (see `_plan`), so a mode
# that skips the orientation, or a request that asks for no measurement, does
# not leave a jump where the skipped stage would have been.
STAGE_WEIGHTS = {
    "resample": 8,
    "orient": 20,
    # Sorting scans that came already oriented, in place of "orient".
    "sort": 1,
    "segment": 15,
    "register": 20,
    "landmarks": 15,
    "measure": 7,
    "classify": 3,
    # Two Batch_Dental_Seg calls per region, then a distance map per patient.
    "heat_maps": 15,
}
# The second scan is a mirror for an asymmetry assessment -- an AutoMatrix call
# per frame -- and a whole second resample and orientation for a longitudinal
# study, which is why the two weigh so differently.
SECOND_WEIGHTS = {
    catalogs.STUDY_ASYMMETRY: 5,
    "longitudinal": STAGE_WEIGHTS["resample"] + STAGE_WEIGHTS["orient"],
}
# Within the landmark stage: an ALI_CBCT call is a full search per landmark, an
# AutoMatrix call moves files.
LANDMARK_CALL_WEIGHT = 4
MATRIX_CALL_WEIGHT = 1
# What came before the first stage -- reading the measurement lists -- keeps
# the sliver of the bar it always had.
LEAD = 0.02


def _plan(mode: str, study: str, wants_measurements: bool, wants_heat_maps: bool) -> dict:
    """{stage: (start, end)} for the stages this run makes, in the order it
    makes them, tiling LEAD..1.

    The waypoints used to be fixed numbers, and the heat maps reused the
    classification's 0.97 -- so a run asking only for heat maps sat at 55% for
    the whole registration and then leapt to 97% for its longest remaining
    stage. Laying out only what runs is what makes each stage's slice honest.
    """
    stages = []
    if mode == catalogs.MODE_FULL:
        stages += ["resample", "orient"]
    else:
        stages.append("sort")
    if mode != catalogs.MODE_REGISTERED:
        stages += ["segment", "second", "register"]
    if wants_measurements:
        stages += ["landmarks", "measure", "classify"]
    if wants_heat_maps:
        stages.append("heat_maps")

    weights = [
        SECOND_WEIGHTS.get(study, SECOND_WEIGHTS["longitudinal"]) if stage == "second"
        else STAGE_WEIGHTS[stage]
        for stage in stages
    ]
    return dict(zip(stages, tools.split_span((LEAD, 1.0), weights)))


def _stage(sup, plan: dict, key: str, message: str) -> tuple:
    """Say where the run has got to, through whichever channel exists, and
    return the stage's span for the calls it makes.

    The supervisor's when there is one -- a chain's events land in one file,
    already ordered -- and the tool's own otherwise, so a standalone run
    reports just the same.
    """
    span = plan[key]
    if sup is not None and hasattr(sup, "progress"):
        sup.progress(span[0], message)
    else:
        progress.emit(span[0], message)
    logger.info("VFACE: %s", message)
    return span


def _log(sup, message: str, level: str = "info", user: bool = False) -> None:
    """`sup.log` when there is a supervisor, the progress file's log otherwise.

    Never a file name: the line reaches the operator page, and with `user`
    the clinician's panel.
    """
    if sup is not None and hasattr(sup, "log"):
        sup.log(message, level=level, user=user)
    else:
        progress.log(message, level=level, user=user)


def _folder(work_dir: str, *parts) -> str:
    destination = os.path.join(work_dir, *parts)
    os.makedirs(destination, exist_ok=True)
    return destination


def _split_by_frame(scans_dir: str, work_dir: str) -> dict:
    """Scans already oriented, sorted into the frame each was oriented into.

    A file names its frame: ASO appends the suffix VFACE asked it for, `CB_Or`
    or `MAX_Or`. Upstream copies them into per-frame folders and so does this,
    because every step after reads one frame at a time.

    A scan naming neither frame is left out and reported. Guessing would put a
    maxillary scan in the cranial base's folder, where it would be registered
    against the wrong masks and reported as a success.
    """
    found = {frame: _folder(work_dir, "oriented", frame) for frame in catalogs.FRAMES}
    placed = {frame: 0 for frame in found}
    unplaced = []

    # The transforms travel with the scans. They are what lets one frame's
    # landmarks be expressed in the other's, so a mode that took the scans and
    # left the `.tfm` files behind could orient nothing and derive nothing --
    # and would say so three steps later, about the wrong step.
    everything = discovery.find_scans(scans_dir) + discovery.find_by_extension(
        scans_dir, (".tfm",)
    )
    for path in sorted(everything):
        name = os.path.basename(path)
        frames = [frame for frame, entry in catalogs.FRAMES.items()
                  if entry["suffix"].lower() in name.lower()]
        if len(frames) != 1:
            unplaced.append(os.path.relpath(path, scans_dir))
            continue
        shutil.copy2(path, os.path.join(found[frames[0]], name))
        if not name.lower().endswith(".tfm"):
            placed[frames[0]] += 1

    missing = [frame for frame, count in placed.items() if count == 0]
    if missing:
        raise ToolInputError(
            "The scans do not say which frame they were oriented into. Expected "
            f"names carrying {' and '.join(catalogs.FRAMES[f]['suffix'] for f in missing)}, "
            f"which is what the orientation appends. Found {len(unplaced)} scan(s) "
            "naming neither."
        )
    return {"folders": found, "unplaced": unplaced}


def _orient(sup, scans_dir: str, work_dir: str, reference_of: dict,
            landmark_model: str, span=None) -> dict:
    """The cohort in both frames, one ASO call each, each in half of `span`."""
    oriented = {}
    spans = tools.split_span(span, [1] * len(catalogs.FRAMES))
    for (frame, entry), frame_span in zip(catalogs.FRAMES.items(), spans):
        reference = reference_of.get(frame)
        if not reference:
            raise ToolInputError(
                f"Orienting into the {entry['reference']} frame needs its reference "
                f"case, and none was named."
            )
        oriented[frame] = tools.orient_scans(
            sup, scans_dir, reference, entry["landmarks"], entry["suffix"],
            landmark_model, label=frame, span=frame_span,
        )
    return oriented


def _second_timepoint(sup, study: str, oriented: dict, mirror_reference: str,
                      t2: str, work_dir: str, reference_of: dict,
                      landmark_model: str, report: dict, span=None) -> dict:
    """What the baseline is compared against, in each frame.

    The patient's own mirror for an asymmetry assessment, the follow-up for a
    longitudinal study. Both come back as `{frame: folder}`, which is all the
    registration needs to know.
    """
    if study == catalogs.STUDY_ASYMMETRY:
        if not mirror_reference:
            raise ToolInputError(
                "An asymmetry assessment compares a patient against their own "
                "mirror, and 'mirror_reference' names the transform that makes it."
            )
        spans = tools.split_span(span, [1] * len(oriented))
        return {
            frame: tools.mirror(sup, folder, mirror_reference, content="Scan",
                                label=f"scans-{frame}", span=frame_span)
            for (frame, folder), frame_span in zip(oriented.items(), spans)
        }

    if not t2:
        raise ToolInputError(
            "A longitudinal study compares two timepoints, and no 't2' was sent. "
            f"Use '{catalogs.STUDY_ASYMMETRY}' to compare a patient against their "
            "own mirror instead."
        )
    # The follow-up goes through the same two stages the baseline did, in the
    # same proportions: a resample, then an orientation per frame.
    _resample_span, orient_span = tools.split_span(
        span, [STAGE_WEIGHTS["resample"], STAGE_WEIGHTS["orient"]]
    )
    resampled = resample.resample_cohort(t2, _folder(work_dir, "t2_resampled"),
                                         report=report)
    return _orient(sup, resampled, work_dir, reference_of, landmark_model,
                   span=orient_span)


def _register(sup, regions, oriented: dict, second: dict, masks: dict,
              span=None) -> dict:
    """Each region registered in its own frame. Returns `{region: folder}`.

    One AREG_CBCT call per region, each in an equal share of `span`.
    """
    registered = {}
    for region, region_span in zip(regions, tools.split_span(span, [1] * len(regions))):
        entry = catalogs.REGION_TABLE[region]
        frame = entry["frame"]
        registered[region] = tools.register(
            sup, oriented[frame], second[frame], entry["areg"], masks[frame],
            label=region, span=region_span,
        )
    return registered


def _landmarks(sup, oriented: dict, registered: dict, measurements: dict,
               work_dir: str, landmark_model: str, mirror_reference: str,
               study: str, second: dict, report: dict, span=None) -> tuple:
    """Both timepoints' landmarks, per frame and per region.

    Predicted ONCE, in the cranial base frame, and carried into the maxillary
    one by a rigid transform. Searching a second time costs minutes per patient
    and makes the same anatomical point land in two slightly different places
    depending on which run found it.
    """
    wanted = sorted({
        name
        for region_measurements in measurements.values()
        for name in aq3dc.landmarks_needed(region_measurements)
    })
    if not wanted:
        raise ToolInputError("The measurement lists name no landmark to find.")

    # The calls this stage makes, in the order it makes them: the baseline's
    # search, then the comparison set per frame -- mirrored, or searched again
    # on the follow-up -- then one AutoMatrix move per region. Each gets its
    # own slice of the stage, weighted by what it costs.
    asymmetry = study == catalogs.STUDY_ASYMMETRY
    compared_weight = MATRIX_CALL_WEIGHT if asymmetry else LANDMARK_CALL_WEIGHT
    frames = [catalogs.FRAME_CRANIAL_BASE, catalogs.FRAME_MAXILLA]
    slices = tools.split_span(
        span,
        [LANDMARK_CALL_WEIGHT]
        + [compared_weight] * (len(frames) if asymmetry else len(second))
        + [MATRIX_CALL_WEIGHT] * len(registered),
    )
    baseline_span = slices[0]
    compared_spans = slices[1:len(slices) - len(registered)]
    region_spans = slices[len(slices) - len(registered):]

    padded = landmark_files.pad_scans(
        oriented[catalogs.FRAME_CRANIAL_BASE], _folder(work_dir, "padded"), report=report
    )
    baseline = {
        catalogs.FRAME_CRANIAL_BASE: tools.predict_landmarks(
            sup, padded, wanted, landmark_model, label="t1", span=baseline_span,
        )
    }
    baseline[catalogs.FRAME_MAXILLA] = landmark_files.derive_into_frame(
        baseline[catalogs.FRAME_CRANIAL_BASE],
        oriented[catalogs.FRAME_CRANIAL_BASE],
        oriented[catalogs.FRAME_MAXILLA],
        oriented[catalogs.FRAME_MAXILLA],
        wanted, _folder(work_dir, "landmarks", catalogs.FRAME_MAXILLA), report=report,
    )

    # The comparison set: the same points seen the other way. For an asymmetry
    # assessment they are the baseline's own, mirrored; for a longitudinal study
    # they are the follow-up's, predicted on it.
    if study == catalogs.STUDY_ASYMMETRY:
        compared = {
            # "Automatic", not "Scan": `content` chooses how VOXELS are
            # resampled, and these folders hold markups. AutoMatrix reads it per
            # FILE, which is what a folder of landmarks needs.
            frame: tools.mirror(sup, folder, mirror_reference, content="Automatic",
                                label=f"landmarks-{frame}", span=frame_span)
            for (frame, folder), frame_span in zip(baseline.items(), compared_spans)
        }
    else:
        # Padded too, and for the same reason. Searching the baseline with room
        # around it and the follow-up without would place the border landmarks
        # of one timepoint and not the other -- and every measurement between
        # them is a difference, so a systematic gap on one side IS the answer.
        compared = {
            frame: tools.predict_landmarks(
                sup,
                landmark_files.pad_scans(
                    folder, _folder(work_dir, "padded_t2", frame), report=report
                ),
                wanted, landmark_model, label=f"t2-{frame}", span=frame_span,
            )
            for (frame, folder), frame_span in zip(second.items(), compared_spans)
        }

    # And then moved by the registration, so both sets sit in one frame. One
    # call per region: AutoMatrix pairs each matrix with its patient by name
    # when it is handed a folder, and the registration writes one matrix per
    # patient per region.
    per_region = {
        region: tools.apply_transforms(
            sup, compared[catalogs.REGION_TABLE[region]["frame"]], folder,
            label=f"registered-{region}", span=region_span,
        )
        for (region, folder), region_span in zip(registered.items(), region_spans)
    }
    return baseline, per_region


def _measure(regions, baseline: dict, compared: dict, measurements: dict,
             output_dir: str, report: dict) -> dict:
    """One measurement table per region, written where a reader can open them."""
    destination = _folder(output_dir, MEASUREMENTS_DIRNAME)
    stats = {}
    for region in regions:
        frame = catalogs.REGION_TABLE[region]["frame"]
        rows = aq3dc.compute_cohort(
            landmark_files.read_cohort(baseline[frame]),
            landmark_files.read_cohort(compared[region]),
            measurements[region], report=report,
        )
        if not rows:
            raise ToolInputError(
                f"No measurement could be made on the {region}. The per-patient "
                "reasons are in the run report."
            )
        aq3dc.write_table(rows, os.path.join(
            destination, f"Measurements_{_short(region)}.xlsx"
        ))
        stats[_short(region)] = features.to_stats(rows)
    return stats


def _short(region: str) -> str:
    """The three-letter name the feature columns and the output files use.

    Read from `catalogs.REGION_CODES` rather than restated here: the panel shows
    the same codes beside its check boxes, and a region described in two places
    is a region that gets renamed in one of them.
    """
    return catalogs.REGION_CODES[region]


# Which DATA folder this tool's bundles live in, and the name of each. Written
# rather than derived: which folder serves which tool is a deployment fact, and
# these are the names the manifest unpacks those archives to.
#
# Resolved here rather than asked of the caller, for the reason the three AREG
# engines resolve theirs: `models/` holds all six bundles at once, so an unnamed
# argument arrived as that FOLDER -- which is not a bundle -- and the run died
# after the segmentation and the orientation had already been paid for. None of
# them is a clinical choice either: one classifier, one mirror transform, one
# reference per frame.
#
# `landmark_model` is absent on purpose. ALI_CBCT resolves its own weights from
# the same data root, so the empty string it gets here is the right answer, not
# an omission.
_DATA_NAME = "VFACE"
_BUNDLES = {
    "segmentation_model": "AMASSS_Models",
    "classifier_model": "VFACE_classifier",
    "cranial_base_reference": "CBCT_Gold_Frankfurt_Horizontal_Midsagittal_Plane",
    "maxilla_reference": "CBCT_Gold_Occlusal_Midsagittal_Plane",
    "mirror_reference": "Mirror_matrix",
    "measurements": "DefaultList",
    "feature_template": "DefaultList",
}


def _own_bundle(data_root, argument):
    """The bundle this deployment publishes for `argument`, or ""."""
    name = _BUNDLES.get(argument)
    if not data_root or name is None:
        return ""
    candidate = os.path.join(str(data_root), _DATA_NAME, "models", name)
    return candidate if os.path.isdir(candidate) else ""


def main(t1, output_dir, mode=None, study=None, outputs=None, regions=None,
         t2=None, measurements=None, feature_template=None,
         registration_transforms=None,
         cranial_base_reference=None, maxilla_reference=None,
         mirror_reference=None, segmentation_model=None, landmark_model=None,
         classifier_model=None, surface_model=None, sup=None,
         data_root=None):
    """Validate, run the chain the mode asks for, and write the report."""
    started_at = time.monotonic()
    # What the caller named wins; what it left empty this deployment fills. The
    # checks below then see a real bundle, and their message stays about what is
    # missing from the DEPLOYMENT rather than about a field the panel no longer
    # shows. `feature_template` is a file inside its bundle, not the bundle.
    segmentation_model = segmentation_model or _own_bundle(data_root, "segmentation_model")
    classifier_model = classifier_model or _own_bundle(data_root, "classifier_model")
    cranial_base_reference = cranial_base_reference or _own_bundle(
        data_root, "cranial_base_reference")
    maxilla_reference = maxilla_reference or _own_bundle(data_root, "maxilla_reference")
    mirror_reference = mirror_reference or _own_bundle(data_root, "mirror_reference")
    measurements = measurements or _own_bundle(data_root, "measurements")
    if not feature_template:
        bundle = _own_bundle(data_root, "feature_template")
        candidate = os.path.join(bundle, "features.xlsx") if bundle else ""
        feature_template = candidate if os.path.isfile(candidate) else ""
    mode = str(mode or catalogs.MODE_FULL)
    study = str(study or catalogs.STUDY_ASYMMETRY)
    outputs = str(outputs or catalogs.OUTPUT_QUANTITATIVE)
    regions = list(regions or catalogs.REGIONS)

    for value, allowed, what in ((mode, catalogs.MODES, "mode"),
                                 (study, catalogs.STUDIES, "study"),
                                 (outputs, catalogs.OUTPUTS, "outputs")):
        if value not in allowed:
            raise ToolInputError(
                f"'{value}' is not a {what} this tool has. It offers: {', '.join(allowed)}."
            )
    unknown = [region for region in regions if region not in catalogs.REGION_TABLE]
    if unknown:
        raise ToolInputError(
            f"{unknown} name no region this tool measures. It offers: "
            f"{', '.join(catalogs.REGIONS)}."
        )

    output_dir = os.path.abspath(str(output_dir))
    os.makedirs(output_dir, exist_ok=True)
    work_dir = _folder(output_dir, WORK_DIRNAME)

    report = {
        "mode": mode, "study": study, "outputs": outputs, "regions": regions,
    }

    try:
        _run(t1=str(t1), output_dir=output_dir, work_dir=work_dir, mode=mode,
             study=study, outputs=outputs, regions=regions,
             t2=str(t2) if t2 else "",
             measurements=str(measurements) if measurements else "",
             feature_template=str(feature_template) if feature_template else "",
             registration_transforms=(
                 str(registration_transforms) if registration_transforms else ""
             ),
             reference_of={
                 catalogs.FRAME_CRANIAL_BASE: str(cranial_base_reference or ""),
                 catalogs.FRAME_MAXILLA: str(maxilla_reference or ""),
             },
             mirror_reference=str(mirror_reference or ""),
             segmentation_model=str(segmentation_model or ""),
             landmark_model=str(landmark_model or ""),
             classifier_model=str(classifier_model or ""),
             surface_model=str(surface_model or ""),
             report=report, sup=sup)
    finally:
        report["duration_seconds"] = round(time.monotonic() - started_at, 2)
        with open(os.path.join(output_dir, REPORT_NAME), "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        shutil.rmtree(work_dir, ignore_errors=True)

    logger.info("VFACE: finished in %.1fs", report["duration_seconds"])
    return output_dir


def _run(t1, output_dir, work_dir, mode, study, outputs, regions, t2,
         measurements, feature_template, registration_transforms, reference_of,
         mirror_reference,
         segmentation_model, landmark_model, classifier_model, surface_model,
         report, sup):
    """The chain proper, once the arguments are known to be usable."""
    wants_measurements = catalogs.wants_measurements(outputs)
    wants_heat_maps = catalogs.wants_heat_maps(outputs)

    # Everything the mode needs is checked BEFORE a volume is read: a request
    # that cannot work has to come back in a second, not after an hour.
    if mode == catalogs.MODE_FULL:
        tools.require(sup, "ASO", "The full pipeline")
    if mode != catalogs.MODE_REGISTERED:
        tools.require(sup, "AMASSS", "A mode that registers")
        tools.require(sup, "AREG_CBCT", "A mode that registers")
        if study == catalogs.STUDY_ASYMMETRY:
            tools.require(sup, "AutoMatrix", "An asymmetry assessment")
        if not segmentation_model:
            raise ToolInputError(
                "Registering needs masks around the regions measured, and "
                "'segmentation_model' names the bundle that makes them."
            )
    if wants_measurements:
        tools.require(sup, "ALI_CBCT", "Measuring")
        tools.require(sup, "AutoMatrix", "Measuring")
        if not measurements:
            raise ToolInputError(
                "'measurements' names the folder holding one measurement list per "
                "region -- one file per CB, MAND and MAX -- and none was sent."
            )
        # `landmark_model` is NOT required. Which weights the landmark tool
        # predicts with is that tool's business: ALI_CBCT resolves its own from
        # the deployment's data folder, and ASO -- which reaches it for the
        # orientation -- asks only for a supervisor. Demanding a name here made
        # VFACE hold a name for its neighbour's storage, and it is the reason
        # AREG_CBCT dropped the same requirement. An explicit one is still obeyed.
    if wants_heat_maps:
        tools.require(sup, "Batch_Dental_Seg", "Heat maps")
        if not surface_model:
            raise ToolInputError(
                "Heat maps are drawn on segmented surfaces, and 'surface_model' "
                "names the bundle that makes them."
            )

    lists = _measurement_lists(measurements, regions) if wants_measurements else {}
    plan = _plan(mode, study, wants_measurements, wants_heat_maps)

    # --- the scans, in the two frames ---------------------------------------
    if mode == catalogs.MODE_FULL:
        _stage(sup, plan, "resample", "putting the cohort on one voxel grid")
        resampled = resample.resample_cohort(t1, _folder(work_dir, "resampled"),
                                             report=report)
        span = _stage(sup, plan, "orient",
                      "orienting into the cranial base and maxillary frames")
        oriented = _orient(sup, resampled, work_dir, reference_of, landmark_model,
                           span=span)
    else:
        _stage(sup, plan, "sort", "sorting the oriented scans by frame")
        split = _split_by_frame(t1, work_dir)
        oriented = split["folders"]
        if split["unplaced"]:
            report["scans_without_a_frame"] = split["unplaced"]
            # The run goes on without them; the clinician who sent them would
            # otherwise learn it only from the report. A count, never a name.
            _log(sup, f"{len(split['unplaced'])} scan(s) name neither frame and are "
                      "left out", level="warning", user=True)

    # --- the masks, the second scan, the registration ------------------------
    if mode != catalogs.MODE_REGISTERED:
        span = _stage(sup, plan, "segment", "segmenting the bone each region registers on")
        # Only the frames some region actually registers on. Both frames are
        # oriented whatever was asked for -- the landmarks are predicted in the
        # cranial base frame and derived into the maxillary one, so the
        # measurement path needs both -- but a frame no region registers on has
        # no structures to segment, and AMASSS rightly refuses an empty list
        # with "Select at least one structure to segment". A request for the
        # cranial base alone died there, after both orientations had been paid
        # for.
        to_segment = [
            (frame, folder, structures)
            for frame, folder in oriented.items()
            for structures in [catalogs.structures_for(frame, regions)]
            if structures
        ]
        masks = {
            frame: tools.segment_masks(
                sup, folder, segmentation_model, structures, label=frame,
                span=frame_span,
            )
            for (frame, folder, structures), frame_span
            in zip(to_segment, tools.split_span(span, [1] * len(to_segment)))
        }
        span = _stage(sup, plan, "second",
                      "building the scan each patient is compared against")
        second = _second_timepoint(sup, study, oriented, mirror_reference, t2,
                                   work_dir, reference_of, landmark_model, report,
                                   span=span)
        span = _stage(sup, plan, "register", "registering each region")
        registered = _register(sup, regions, oriented, second, masks, span=span)
    else:
        # The caller registered already, so the transforms are theirs to supply.
        # Taking them from `t1` would have read the ORIENTATION transforms
        # sitting beside the scans as though they were registrations -- the
        # landmarks would have been moved by the wrong matrix, and every
        # measurement would have come out of a run that reported success.
        if not registration_transforms:
            raise ToolInputError(
                f"'{catalogs.MODE_REGISTERED}' measures a registration somebody "
                "else made, so it needs it: 'registration_transforms' is the folder "
                "of per-patient transforms that registration produced, one subfolder "
                f"per region ({', '.join(catalogs.REGIONS)})."
            )
        second = dict(oriented)
        registered = {
            region: os.path.join(registration_transforms, region)
            if os.path.isdir(os.path.join(registration_transforms, region))
            else registration_transforms
            for region in regions
        }

    if not catalogs.wants_measurements(outputs):
        report["measurements"] = "not asked for"
    else:
        span = _stage(sup, plan, "landmarks",
                      "finding the landmarks the measurements are made on")
        baseline, compared = _landmarks(
            sup, oriented, registered, lists, work_dir, landmark_model,
            mirror_reference, study, second, report, span=span,
        )
        _stage(sup, plan, "measure", "measuring")
        stats = _measure(regions, baseline, compared, lists, output_dir, report)

        _stage(sup, plan, "classify", "reading the measurements as a classification")
        _classify(stats, feature_template, classifier_model, output_dir, report)

    if wants_heat_maps:
        from . import heatmap

        # Its own stage, after the classification rather than on top of it:
        # two segmentations per region are the longest thing left in the run.
        span = _stage(sup, plan, "heat_maps", "drawing the heat maps")
        heatmap.draw_cohort(sup, oriented, registered, regions, surface_model,
                            output_dir, work_dir, report, span=span)
        if report.get("heat_maps_unpaired") or report.get("heat_maps_failed"):
            # Some patient has a map missing for some region. Counted, never
            # named: the per-pair detail is in the run report.
            missing = (sum(report.get("heat_maps_unpaired", {}).values())
                       + len(report.get("heat_maps_failed", {})))
            _log(sup, f"{missing} heat map(s) could not be drawn; the run report "
                      "says which region", level="warning", user=True)


def _measurement_lists(folder: str, regions) -> dict:
    """One measurement list per region, found by what its file is called.

    Upstream matches on the name -- a file whose name holds `CB` is the cranial
    base's list -- and so does this, because the lists are hand-made and named
    by whoever wrote them.
    """
    if not os.path.isdir(folder):
        raise ToolInputError(f"'{folder}' is not a folder of measurement lists.")

    spellings = {
        catalogs.REGION_CRANIAL_BASE: ("CB", "CRANIAL", "CRANIOFACIAL"),
        catalogs.REGION_MANDIBLE: ("MAND", "MANDIBLE", "MANDIBULAR"),
        catalogs.REGION_MAXILLA: ("MAX", "MAXILLA", "MAXILLARY"),
    }
    workbooks = sorted(
        os.path.join(folder, name) for name in os.listdir(folder)
        if name.lower().endswith((".xlsx", ".xls")) and not name.startswith("~$")
    )
    if not workbooks:
        raise ToolInputError(f"'{os.path.basename(folder)}' holds no Excel file.")

    found = {}
    for region in regions:
        # MANDible also holds "MAND" and would match the maxilla's "MAX" only
        # if a name held both; the first match wins, and the order below puts
        # the longer, less ambiguous spellings first.
        for path in workbooks:
            name = os.path.basename(path).upper()
            if any(spelling in name for spelling in spellings[region]):
                found[region] = aq3dc.read_measurement_list(path)
                break
    missing = [region for region in regions if region not in found]
    if missing:
        raise ToolInputError(
            f"No measurement list found for {', '.join(missing)}. A list is matched "
            "to its region by its file name, so one of them should carry "
            + "; ".join(f"{region}: {'/'.join(spellings[region])}" for region in missing)
        )
    return found


def _classify(stats, feature_template: str, classifier_model: str,
              output_dir: str, report: dict) -> None:
    """The feature table, and the classification read off it.

    Two steps rather than one, because they need different things. A template
    alone gives the feature table -- the measurements assembled one row per
    patient, which is a usable answer and what somebody training a model would
    ask for. The verdict on top of it needs a bundle from the SAME training
    run, so the columns the models name exist.
    """
    if not feature_template:
        report["classification"] = (
            "not run: it needs 'feature_template', the workbook naming the features "
            "the classifier was trained on"
        )
        return

    try:
        columns = features.read_template_columns(feature_template)
        records = features.build_feature_table(stats, columns, report=report)
    except ToolInputError as exc:
        # The measurements are already computed and written. Raising here would
        # take a registration, a landmark search and a table of measurements
        # down with the verdict on top of them -- and the server destroys the
        # job directory on failure, so the clinician would be left with the GPU
        # minutes and nothing else.
        report["classification"] = f"not run: {exc}"
        return

    features.write_table(
        records, ["ID"] + list(columns),
        os.path.join(output_dir, MEASUREMENTS_DIRNAME, FEATURE_TABLE_NAME),
    )
    report["features"] = len(records)

    if not classifier_model:
        report["classification"] = (
            "not run: the feature table was written, and reading it as a verdict "
            "needs 'classifier_model' from the same training run"
        )
        return

    try:
        classified = classify.classify(records, classifier_model, report=report)
    except ToolInputError as exc:
        # Same reasoning, one step further along: the feature table is written
        # too by now, and it is what somebody would look at to find out WHY the
        # models could not read it.
        report["classification"] = (
            f"not run: {exc} The measurements and the feature table were written "
            "and are in this archive."
        )
        return

    classify.write_table(classified, os.path.join(
        output_dir, CLASSIFICATION_DIRNAME, CLASSIFICATION_NAME
    ))
    report["classified"] = len(classified)
