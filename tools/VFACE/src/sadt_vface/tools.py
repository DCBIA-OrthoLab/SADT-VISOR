"""The seam between VFACE and the six tools it drives.

VFACE measures and classifies. It does not orient a scan, segment bone, apply a
transform, register a timepoint or place a landmark -- each of those is another
tool in this repository, and each is reached through the **supervisor**, the
object whatever runs VFACE hands it as the keyword-only `sup`.

| Asked for | Tool | When |
|---|---|---|
| an orientation, per frame | `ASO` | Full pipeline |
| masks around the regions measured | `AMASSS` | anything that registers |
| the patient's own scan, mirrored | `AutoMatrix` | an asymmetry assessment |
| the mirror (or the follow-up) registered onto the baseline | `AREG_CBCT` | anything that registers |
| the landmarks every measurement is computed from | `ALI_CBCT` | measurements |
| surfaces to draw a heat map on | `Batch_Dental_Seg` | heat maps |

**Six is the widest call graph in this repository**, and two of them are
themselves supervised -- `ASO` reaches `ALI_CBCT`, `AREG_CBCT` reaches `AMASSS`
and `ASO` -- so a full run is four tools deep and the runner's cap is five.
Nothing here arranges that; it is recursion in the runner.

Three things worth knowing before changing anything here:

* **Tools are named by string, never by attribute.** `sup.run("ASO", ...)`, not
  `sup.ASO(...)`. A typo in a string is greppable and this file is the whole
  call graph; a typo in an attribute is an `AttributeError` an hour into a job.
* **The arguments are the callee's published schema**, not VFACE's vocabulary.
  When a tool renames an argument this file is what breaks, which is the point:
  it breaks in one place, with the name in it.
* **This file must stay in VFACE's own `src/`.** `describe.py` derives the
  schema's `calls` field by READING this source -- the call sites sit in
  branches only a real run reaches -- and the server refuses to start when a
  declared call names a tool it does not serve. Moved into a shared package,
  the names never reach `calls` and the startup check silently has nothing to
  verify. See CONTRIBUTING.md.
"""

import logging
import os

from .errors import SupervisorRequired, ToolInputError

logger = logging.getLogger(__name__)

# What each tool is asked for, and what a caller who cannot reach it can do
# instead. The advice is the useful half: "deploy a tool" is not an answer a
# clinician can act on, and every one of these has a mode that works without it.
_ADVICE = {
    "ASO": (
        "Orient the scans yourself beforehand and use the 'File already Oriented' "
        "mode, which takes them as they are."
    ),
    "AMASSS": (
        "Send your own segmentation masks in 'masks', which is what the modes "
        "that do not segment expect."
    ),
    "AutoMatrix": (
        "An asymmetry assessment compares a patient against their own mirror, and "
        "the mirror is made by applying one transform. Send the mirrored scans in "
        "'t2' and run a longitudinal study over them instead."
    ),
    "AREG_CBCT": (
        "Register the timepoints yourself and use the 'File already Registered' "
        "mode."
    ),
    "ALI_CBCT": (
        "Send your own landmark files in 'landmarks'; every measurement is "
        "computed from those and nothing else."
    ),
    "Batch_Dental_Seg": (
        "Heat maps need surfaces to draw on. Ask for measurements alone, which "
        "needs no segmentation."
    ),
}


def require(sup, tool: str, mode: str) -> None:
    """Refuse a mode that needs `tool` when there is no way to run it.

    Checked up front, before a single volume is read: a request that cannot
    work has to come back in a second, not after an hour of registration.
    """
    if sup is not None:
        return
    raise SupervisorRequired(
        f"{mode} needs the '{tool}' tool, and nothing here can run it: no "
        f"supervisor was supplied. {_ADVICE.get(tool, '')}".strip()
    )


def split_span(span, weights) -> list:
    """`span` cut into consecutive slices in proportion to `weights`, in order.

    A stage that makes several calls -- one per frame, one per region -- gives
    each its own slice, because the server folds each call into the span it is
    handed: two calls sharing one span would show the second as the bar going
    back to where the first started. None in, a None per call out, so a caller
    with no bar still makes every call.
    """
    weights = [float(weight) for weight in weights]
    if span is None or not weights:
        return [None] * len(weights)
    if sum(weights) <= 0:
        weights = [1.0] * len(weights)
    start, end = span
    total = sum(weights)
    slices, position = [], start
    for weight in weights:
        width = (end - start) * weight / total
        slices.append((round(position, 6), round(position + width, 6)))
        position += width
    # Rounding must not leave a sliver between the last slice and the next
    # stage: the last one ends exactly where the stage does.
    slices[-1] = (slices[-1][0], end)
    return slices


def _span(span) -> dict:
    """`_progress` for `sup.run`, or nothing when the caller gave no span.

    The keyword is the supervisor's, not the callee's: it removes it before the
    callee sees it and folds the callee's own 0..1 into that slice of this
    run's bar. Left out, the call behaves exactly as it did before there was
    such a thing.
    """
    return {"_progress": tuple(span)} if span else {}


def _returned(produced) -> str:
    """A tool returns a Path, or a dict of named ones; VFACE wants a directory."""
    if isinstance(produced, dict):
        produced = next(iter(produced.values()))
    return str(produced)


# No `output_dir` is passed to any of the six. Where a supervised tool writes is
# the supervisor's business: it gives each nested call its own slot,
# `<job>/sup/<NN>_<tool>/output`, numbered per CALL -- so the two orientations
# and the three registrations are already separated, which is what the helper
# this replaced was for. That slot is also the only place `keep_intermediate`
# collects from, so naming a directory here pointed every callee away from it
# and every step came back to the reader as "a call that wrote nowhere".
# What the call RETURNS is where to read from.
#
# `prediction_ID` survives for AMASSS and Batch_Dental_Seg, which still take it.
# ALI_CBCT does not: it turned the marker into a constant, `PREDICTION_ID`,
# precisely so that whatever consumes its output can predict it. The value sent
# here was "Pred", which is that constant, so nothing moved.

# ---------------------------------------------------------------------------
# The calls
# ---------------------------------------------------------------------------

def orient_scans(sup, scans: str, reference: str, landmarks, suffix: str,
                 landmark_model: str = "", label: str = "", span=None) -> str:
    """Orient a cohort into one frame.

    Fully-Automated, which for CBCT means ASO predicts the landmarks through
    ALI_CBCT and then registers on them -- the three steps upstream spells out
    as PRE_ASO, ALI and SEMI_ASO, in the order ASO already runs them. Reaching
    for the packaged tool rather than restaging its three parts is what keeps
    one description of "how a CBCT is oriented" in the repository.

    `landmarks` is the frame's own list, and it is not a detail: the cranial
    base frame is fitted on Ba/S/N/Po/Or and the maxillary frame on the
    occlusal points, and each `reference` is built on its own set.
    """
    logger.info("VFACE: asking 'ASO' to orient into the %s frame", label or "requested")
    parameters = {
        "input": scans,
        "reference": reference,
        "modality": "CBCT",
        "automation": "Fully-Automated",
        "cbct_landmarks": list(landmarks),
        "output_suffix": suffix,
    }
    # Sent only when the caller named one. ASO reaches the landmark tool
    # itself and asks only for a supervisor, and ALI_CBCT resolves its own
    # weights from the data folder -- so an omitted bundle is the right
    # request, not a forgotten one.
    if landmark_model:
        parameters["landmark_model"] = landmark_model
    return _returned(sup.run("ASO", **parameters, **_span(span)))


def segment_masks(sup, scans: str, model: str, structures, label: str = "",
                  span=None) -> str:
    """Segment the bone the registration and the heat maps are keyed on.

    `merge=["SEPARATE"]` for the same reason AREG asks for it: one binary file
    per structure, so a region's mask can be looked up by name. A merged
    multi-label volume makes every region resolve to the same file.
    """
    logger.info("VFACE: asking 'AMASSS' for %s masks", ", ".join(structures))
    return _returned(sup.run(
        "AMASSS",
        scans=scans,
        model=model,
        structures=list(structures),
        merge=["SEPARATE"],
        prediction_ID="seg",
        generate_surface=False,
        **_span(span),
    ))


def mirror(sup, files: str, transform: str, content: str = "Automatic",
           label: str = "", span=None) -> str:
    """The patient's own scan, mirrored -- which is what an asymmetry assessment
    compares against.

    There is no second scan in an asymmetry assessment. One transform, the same
    for every patient, reflects the scan across the mid-sagittal plane, and the
    rest of the pipeline then treats that reflection exactly as a longitudinal
    study treats a follow-up. That is the whole difference between the two
    studies, and it is this one call.
    """
    logger.info("VFACE: asking 'AutoMatrix' to mirror the %s", label or "scans")
    return _returned(sup.run(
        "AutoMatrix",
        files=files,
        transforms=transform,
        # One transform for the whole cohort: it is a reflection of the frame,
        # not something fitted per patient.
        same_transform_for_every_patient=True,
        output_suffix="mir",
        content=content,
        **_span(span),
    ))


def apply_transforms(sup, files: str, transforms: str, label: str = "",
                     span=None) -> str:
    """Move each patient's files by that patient's OWN transform.

    The other AutoMatrix call. `mirror` applies one transform to everybody --
    a reflection of the frame is the same for every patient -- and this one
    pairs a folder of transforms to a folder of files by patient name, which is
    what the registration produces: one matrix per patient per region.

    Sending a folder of transforms with `same_transform_for_every_patient` set
    would apply whichever one sorted first to the whole cohort, and the constant
    suffix would make each patient overwrite the last.
    """
    logger.info("VFACE: asking 'AutoMatrix' to move the %s by the registration",
                label or "landmarks")
    return _returned(sup.run(
        "AutoMatrix",
        files=files,
        transforms=transforms,
        same_transform_for_every_patient=False,
        output_suffix="reg",
        content="Automatic",
        **_span(span),
    ))


def register(sup, t1: str, t2: str, region: str, masks: str, label: str = "",
             span=None) -> str:
    """Register `t2` onto `t1` on one region's bone.

    Semi-Automated: VFACE has already oriented the scans and already has the
    masks, so asking AREG to segment and orient again would redo two steps and
    -- worse -- orient the mirror independently of the original, which is the
    one thing that must not happen. A mirror is only a mirror while it shares
    the original's frame.
    """
    logger.info("VFACE: asking 'AREG_CBCT' to register on the %s", region)
    return _region_folder(_returned(sup.run(
        "AREG_CBCT",
        t1=t1,
        t2=t2,
        automation="Semi-Automated",
        regions=[region],
        t1_masks=masks,
        output_suffix="Reg",
        **_span(span),
    )))


def _region_folder(produced: str) -> str:
    """Where AREG_CBCT actually put the matrices, one level down.

    It groups its results by region -- `CB/`, `MAND/`, `MAX/` beside its report
    -- so the directory it returns holds no transform at all. AutoMatrix, handed
    that directory next, refused with "No transform found", after the whole
    segmentation and registration had been paid for.

    The subfolder is found rather than named: VFACE asks for ONE region per
    call, so there is exactly one, and looking it up in AREG's own vocabulary
    would put that table in two places. `REGION_TABLE`'s `areg` column exists
    precisely because the two vocabularies are not the same.
    """
    try:
        entries = list(os.scandir(produced))
    except OSError as exc:
        raise ToolInputError(
            f"The registration returned '{os.path.basename(str(produced))}', which "
            f"cannot be read back ({exc.strerror})."
        ) from exc

    folders = [entry for entry in entries if entry.is_dir()]
    here = _transforms_in(produced)
    if here:
        return produced
    if len(folders) == 1 and _transforms_in(folders[0].path):
        return folders[0].path

    # Said HERE rather than left to AutoMatrix, which is handed this directory
    # three steps later and can only report that it holds no transform -- with
    # no way to say which tool produced it or what it does hold. The whole
    # segmentation and registration have been paid for by then.
    found = ", ".join(sorted(entry.name for entry in entries)) or "nothing"
    raise ToolInputError(
        f"The registration produced no transform to move the landmarks by. "
        f"Looked in '{os.path.basename(str(produced))}' and in each folder "
        f"below it; it holds: {found}.{_why_nothing_registered(produced)}"
    )


def _why_nothing_registered(produced: str) -> str:
    """AREG's own account of the run, when it left one and registered nothing.

    It writes a report whether or not it registered anything, and that report
    is the only place that says WHY -- which patients it could not pair, and
    what it did with the ones it could. Reading it here is what turns "no
    transform" into an answer: the alternative is a reader who has the refusal
    and no way to reach the explanation, the job directory being torn down with
    the failure.
    """
    import json

    report = os.path.join(produced, "AREG_report.json")
    try:
        with open(report, encoding="utf-8") as handle:
            content = json.load(handle)
    except (OSError, ValueError):
        return ""

    parts = []
    # AREG_CBCT writes "unmatched", `{"t1_without_t2": [...], "t2_without_t1":
    # [...]}`, always present and usually empty; "unpaired" is the older key,
    # read for a report written before the rename. Counted, never listed: the
    # entries are patient names.
    unmatched = content.get("unmatched") or {}
    if isinstance(unmatched, dict):
        t1_only = len(unmatched.get("t1_without_t2") or [])
        t2_only = len(unmatched.get("t2_without_t1") or [])
        if t1_only or t2_only:
            parts.append(f"unmatched: {t1_only} T1-only, {t2_only} T2-only")
    unpaired = content.get("unpaired")
    if unpaired:
        count = len(unpaired) if isinstance(unpaired, (list, dict)) else unpaired
        parts.append(f"unpaired: {count}")
    patients = content.get("patients") or {}
    for index, entry in enumerate(list(patients.values())[:3], start=1):
        state = entry.get("status", "?")
        reason = entry.get("error") or entry.get("reason") or ""
        # AREG records the reason PER REGION, one level below the patient, so
        # reading only the patient gives "failed" and nothing else.
        for code, region in (entry.get("regions") or {}).items():
            if region.get("reason"):
                reason = f"{code}: {region['reason']}"
                break
        parts.append(f"patient {index} of {len(patients)}: {state}"
                     f"{f' ({reason})' if reason else ''}")
    if not parts:
        parts.append("its report names no patient at all")
    return " AREG says -- " + "; ".join(parts) + "."


# What AutoMatrix will accept as a transform, so this tool can check for one
# before handing a directory over rather than after.
_TRANSFORM_SUFFIXES = (".tfm", ".mat", ".h5", ".txt")


def _transforms_in(directory: str) -> list:
    try:
        return [entry.name for entry in os.scandir(directory)
                if entry.is_file() and entry.name.endswith(_TRANSFORM_SUFFIXES)]
    except OSError:
        return []


def predict_landmarks(sup, scans: str, landmarks, model: str = "",
                      label: str = "", span=None) -> str:
    """The landmarks every measurement is computed from.

    Asked for by NAME, not by region: a measurement names the points it needs,
    and ALI spawns one search agent per landmark. Asking for a region's whole
    catalogue would search dozens of points no measurement reads, at about a
    minute each.
    """
    logger.info("VFACE: asking 'ALI_CBCT' for %d landmark(s)", len(landmarks))
    parameters = {
        "input": scans,
        "landmarks": list(landmarks),
    }
    if model:
        parameters["model"] = model
    return _returned(sup.run("ALI_CBCT", **parameters, **_span(span)))


def segment_surfaces(sup, scans: str, model: str, label: str = "", span=None) -> str:
    """The surfaces a heat map is drawn on.

    Only the visualisation path asks for this. The measurements are computed
    from landmarks and need no surface at all, which is why a deployment
    without this tool can still answer the quantitative half.

    `model` is positional and required because it is required THERE:
    `Batch_Dental_Seg.run` takes it with no default, so an omitted one is a
    `TypeError` one tool down rather than a refusal here.
    """
    logger.info("VFACE: asking 'Batch_Dental_Seg' for surfaces to draw on")
    return _returned(sup.run(
        "Batch_Dental_Seg",
        scans=scans,
        model=model,
        prediction_ID="Seg",
        **_span(span),
    ))
