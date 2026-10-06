"""How this tool reaches the four it depends on, through the supervisor.

AREG_IOSCBCT predicts nothing itself. The landmarks it registers on come from
ALI_CBCT and ALI_IOS, the tooth labels from Crown_Seg and the orientation from
ASO -- each in its own virtualenv, started as a subprocess by the supervisor.
That is what lets this tool drive an engine pinned to torch 2.8 and another
pinned to 2.11 while depending on neither.

**Tools are named by string, never imported.** A tool cannot import another --
separate virtualenvs are the reason the split exists -- so the name is a free
string, and `describe.py` reads THIS FILE to publish the `calls` list the server
checks at startup. Which is why the calls live here rather than in
`../common/`: shared, they would be invisible to that check.

No `output_dir` is passed to any of them. Where a supervised tool writes is
the supervisor's business, the same way it is the server's over HTTP: it gives
each nested call its own slot, `<job>/sup/<NN>_<tool>/output`, and that slot is
the only place `keep_intermediate` collects from. Naming a directory here
pointed every callee away from it, so each step ran, wrote its files, and came
back to the reader as "a call that wrote nowhere". What the call RETURNS is
where to read from.

The mapping from upstream, made deliberately rather than discovered during a
run, because getting it wrong has cost three separate defects already:

    upstream module          this tool asks for
    CrownSegmentationcli     Crown_Seg
    ALI_CBCT                 ALI_CBCT
    ALI_IOS                  ALI_IOS
    PRE_ASO_CBCT   \\
    SEMI_ASO_CBCT   >        ASO, with modality and automation as arguments
    PRE_ASO_IOS    /

Upstream's three ASO variants are three CLI modules; ours is one tool that
takes the mode as data. And note that `ALI_CBCT` / `ALI_IOS` are upstream's own
names -- our merged `ALI` was the anomaly, which is why an earlier
`sup.run("ALI", ...)` was wrong on both sides of the split.
"""

import logging
import os

from sadt_areg_common.errors import SupervisorRequired

logger = logging.getLogger(__name__)

# What to tell a caller who cannot run each tool. The advice is the useful half:
# somebody who cannot reach ALI_CBCT can still send their own CBCT landmarks.
# The CBCT landmarks the cross-modality alignment matches on, verbatim from
# upstream's IOSCBCT parameter dict. Twelve occlusal crown points, three per
# quadrant -- the central incisor, the canine and the first molar -- chosen
# because they are the points visible in BOTH modalities: a crown tip is a crown
# tip whether it was imaged by a CBCT or scanned intra-orally. That is also why
# the intraoral side asks for the Occlusal network alone.
CBCT_LANDMARKS = (
    "UR1O", "UR3O", "UR6O",
    "UL1O", "UL3O", "UL6O",
    "LR1O", "LR3O", "LR6O",
    "LL1O", "LL3O", "LL6O",
)

_ADVICE = {
    "Crown_Seg": (
        "Send meshes that already carry a per-point tooth-label array, and use "
        "the Registration mode instead."
    ),
    "ALI_CBCT": (
        "Send your own CBCT landmarks in 'cbct_landmarks' and use the "
        "Registration mode instead."
    ),
    "ALI_IOS": (
        "Send your own intraoral landmarks in 'ios_landmarks' and use the "
        "Registration mode instead."
    ),
    "ASO": (
        "Orient the CBCT yourself beforehand, and use a mode that takes it "
        "already oriented."
    ),
}
def require(sup, tool: str, mode: str) -> None:
    """Refuse a mode that needs `tool` when there is no way to run it.

    Checked up front, before a single file is read: a request that cannot work
    has to come back in a second, not after an hour of registration.
    """
    if sup is not None:
        return
    raise SupervisorRequired(
        f"{mode} needs the '{tool}' tool, and nothing here can run it: no supervisor "
        f"was supplied. {_ADVICE.get(tool, '')}"
    )



def _returned(produced) -> str:
    """A tool returns a Path, or a dict of named ones; AREG wants a directory."""
    if isinstance(produced, dict):
        produced = next(iter(produced.values()))
    return str(produced)


def _span(span) -> dict:
    """`_progress` for `sup.run`, or nothing when the caller gave no span.

    The keyword is the supervisor's, not the callee's: it removes it before the
    callee sees it and folds the callee's own 0..1 into that slice of this
    run's bar. Left out, the call behaves exactly as it did before there was
    such a thing, which is what a caller with no bar of its own wants.
    """
    return {"_progress": tuple(span)} if span else {}


def _waypoint(sup, span, message: str) -> None:
    """Say which tool the run is about to be inside.

    At the start of the call's span when it has one -- which is what keeps the
    waypoints in the order the calls are made, whatever that order is. With
    none there is no honest fraction to give, so the line goes to the log
    instead: a number made up here could send the bar backwards.
    """
    if sup is None:
        return
    if span and hasattr(sup, "progress"):
        sup.progress(span[0], message)
    elif hasattr(sup, "log"):
        sup.log(message)


def label_crowns(sup, mesh_dir: str, model_path: str = "", span=None) -> str:
    """Label the crowns of every mesh under `mesh_dir`.

    `skip_segmented` is left at its default: a mesh already carrying a
    tooth-label array passes through untouched, so a batch mixing segmented and
    raw meshes costs network time only for the ones that need it. The original
    did that by hand -- `__BypassCrownseg__` copied files into two directories
    and merged them afterwards -- which is work that belongs inside the tool
    doing the segmenting.
    """
    logger.info("AREG: asking 'Crown_Seg' for tooth-labelled meshes")
    _waypoint(sup, span, "labelling the intraoral crowns with Crown_Seg")
    parameters = {
        "meshes": mesh_dir,
        "suffix": "Seg",
    }
    if model_path:
        parameters["model"] = model_path
    return _returned(sup.run("Crown_Seg", **parameters, **_span(span)))


def predict_cbct_landmarks(sup, scan_dir: str, model_path: str, span=None) -> str:
    """The CBCT landmarks the cross-modality alignment registers on.

    Asked for BY NAME, not by region: the registration uses a handful of points,
    and asking by region would run every agent of every region containing one of
    them. One agent is a full two-scale walk of the volume.
    """
    _waypoint(sup, span, "predicting CBCT landmarks with ALI_CBCT")
    parameters = {
        "input": scan_dir,
        "landmarks": list(CBCT_LANDMARKS),
    }
    if model_path:
        parameters["model"] = model_path
    return _returned(sup.run("ALI_CBCT", **parameters, **_span(span)))


def predict_ios_landmarks(sup, mesh_dir: str, model_path: str, span=None) -> str:
    """The intraoral landmarks, the other half of the correspondence.

    `networks` names the occlusal family alone: the cross-modality alignment
    matches crown points against their CBCT counterparts, and the cervical and
    mucogingival passes would cost a run over every mesh for points nothing
    here reads.
    """
    _waypoint(sup, span, "predicting intraoral landmarks with ALI_IOS")
    parameters = {
        "input": mesh_dir,
        "networks": ["Occlusal"],
    }
    if model_path:
        parameters["model"] = model_path
    return _returned(sup.run("ALI_IOS", **parameters, **_span(span)))


def orient_cbct(sup, scan_dir: str, reference_path: str, landmark_model: str = "",
                span=None, frame: str = "") -> str:
    """Put the CBCT in the reference frame before anything is matched onto it.

    ASO's fully-automated CBCT mode: it predicts its own landmarks and registers
    the scan onto the gold reference. Upstream reaches this through three
    separate CLI modules (PRE_ASO_CBCT, SEMI_ASO_CBCT, PRE_ASO_IOS); ours is one
    tool taking the mode as data.

    `frame` is named in ASO's own words and ASO resolves the reference bundle
    that defines it: the bundles are ASO's. `reference_path` is an override for
    a caller that named one, and wins when given.
    """
    _waypoint(sup, span, "orienting the CBCT with ASO")
    parameters = {
        "input": scan_dir,
        "modality": "CBCT",
        "automation": "Fully-Automated",
    }
    if reference_path:
        parameters["reference"] = reference_path
    elif frame:
        parameters["frame"] = frame
    if landmark_model:
        parameters["landmark_model"] = landmark_model
    return _returned(sup.run("ASO", **parameters, **_span(span)))
