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

from .errors import SupervisorRequired

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


def _output(sup, tool: str, label: str = "") -> str:
    """A directory of the supervisor's scratch for one callee's results.

    `label` separates two calls to the SAME tool -- VFACE orients twice and
    registers three times -- which would otherwise write into one directory and
    leave the second call reading the first one's output as its own.
    """
    destination = os.path.join(str(sup.tmp), "tools", tool, label or "run")
    os.makedirs(destination, exist_ok=True)
    return destination


def _returned(produced) -> str:
    """A tool returns a Path, or a dict of named ones; VFACE wants a directory."""
    if isinstance(produced, dict):
        produced = next(iter(produced.values()))
    return str(produced)


# ---------------------------------------------------------------------------
# The calls
# ---------------------------------------------------------------------------

def orient_scans(sup, scans: str, reference: str, landmarks, suffix: str,
                 landmark_model: str = "", label: str = "") -> str:
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
        "output_dir": _output(sup, "ASO", label),
        "modality": "CBCT",
        "automation": "Fully-Automated",
        "cbct_landmarks": list(landmarks),
        "output_suffix": suffix,
    }
    # ASO needs the bundle NAMED: a tool no longer resolves paths, and
    # forgetting it is a failure two tools down.
    if landmark_model:
        parameters["landmark_model"] = landmark_model
    return _returned(sup.run("ASO", **parameters))


def segment_masks(sup, scans: str, model: str, structures, label: str = "") -> str:
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
        output_dir=_output(sup, "AMASSS", label),
        structures=list(structures),
        merge=["SEPARATE"],
        prediction_ID="seg",
        generate_surface=False,
    ))


def mirror(sup, files: str, transform: str, content: str = "Automatic",
           label: str = "") -> str:
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
        output_dir=_output(sup, "AutoMatrix", label),
        # One transform for the whole cohort: it is a reflection of the frame,
        # not something fitted per patient.
        same_transform_for_every_patient=True,
        output_suffix="mir",
        content=content,
    ))


def register(sup, t1: str, t2: str, region: str, masks: str, label: str = "") -> str:
    """Register `t2` onto `t1` on one region's bone.

    Semi-Automated: VFACE has already oriented the scans and already has the
    masks, so asking AREG to segment and orient again would redo two steps and
    -- worse -- orient the mirror independently of the original, which is the
    one thing that must not happen. A mirror is only a mirror while it shares
    the original's frame.
    """
    logger.info("VFACE: asking 'AREG_CBCT' to register on the %s", region)
    return _returned(sup.run(
        "AREG_CBCT",
        t1=t1,
        t2=t2,
        output_dir=_output(sup, "AREG_CBCT", label or region),
        automation="Semi-Automated",
        regions=[region],
        t1_masks=masks,
        output_suffix="Reg",
    ))


def predict_landmarks(sup, scans: str, landmarks, model: str = "",
                      label: str = "") -> str:
    """The landmarks every measurement is computed from.

    Asked for by NAME, not by region: a measurement names the points it needs,
    and ALI spawns one search agent per landmark. Asking for a region's whole
    catalogue would search dozens of points no measurement reads, at about a
    minute each.
    """
    logger.info("VFACE: asking 'ALI_CBCT' for %d landmark(s)", len(landmarks))
    parameters = {
        "input": scans,
        "output_dir": _output(sup, "ALI_CBCT", label),
        "landmarks": list(landmarks),
        "prediction_ID": "Pred",
    }
    if model:
        parameters["model"] = model
    return _returned(sup.run("ALI_CBCT", **parameters))


def segment_surfaces(sup, scans: str, model: str, label: str = "") -> str:
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
        output_dir=_output(sup, "Batch_Dental_Seg", label),
        prediction_ID="Seg",
    ))
