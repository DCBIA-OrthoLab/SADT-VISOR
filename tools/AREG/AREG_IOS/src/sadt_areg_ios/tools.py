"""The seam between AREG and the four tools it drives.

AREG registers a T2 onto its T1. Getting there needs work it does not do
itself -- masks around the regions to register on, an orientation both
timepoints share, tooth labels, a mucogingival line -- and each of those is
another tool in this repository. AREG calls them; it does not contain them.

Every call goes through the **supervisor**, the object whatever runs AREG hands
it as the keyword-only `sup`. Nothing here imports another tool: they have
different interpreters and irreconcilable dependency sets, which is the whole
reason the split exists. `sup.run("AMASSS", ...)` starts that tool in its own
venv and blocks until it is done.

Four things worth knowing before changing anything here:

* **Tools are named by string, never by attribute.** `sup.run("ASO", ...)`, not
  `sup.ASO(...)`. A typo in a string is greppable and this file is the whole
  call graph; a typo in an attribute is an `AttributeError` an hour into a job.
* **The arguments are the callee's published schema**, not AREG's vocabulary.
  When a tool renames an argument this file is what breaks, which is the point:
  it breaks in one place, with the name in it.
* **No `output_dir` is passed to a callee.** Where a supervised tool writes is
  the supervisor's business, the same way it is the server's over HTTP. Naming
  one pointed the callee into this run's scratch and left the slot the
  supervisor reserves for it -- `<job>/sup/<NN>_<tool>/output` -- empty, which
  is the ONLY place `keep_intermediate` collects from: the step ran, wrote its
  files, and came back to the caller as "a call that wrote nowhere". The
  directory the call returns is what to read from.
* **A missing supervisor is not a bad request.** Nothing about the caller's
  arguments is wrong -- there is simply no way to reach the other tool. Each
  `require_*` below says which mode to use instead, because that is a real
  answer and "deploy a tool" usually is not.
"""

import logging
import os

from sadt_areg_common.errors import SupervisorRequired

logger = logging.getLogger(__name__)

# What each tool is asked for, and what to do when it cannot be reached. The
# advice is the useful half: a caller who cannot run AMASSS can still send their
# own masks, and saying so beats naming a deployment problem they cannot fix.
_ADVICE = {
    "ASO": (
        "Orient the T1 and T2 scans yourself beforehand, and use the mode that takes "
        "them already oriented."
    ),
    "Crown_Seg": (
        "Segment the crowns yourself -- the meshes need a per-point tooth-label array "
        "-- and use Semi-Automated mode instead."
    ),
    "ALI_IOS": (
        "Send the 13 mucogingival landmarks per lower scan in 'mgl_landmarks' instead."
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


def orient_scans(sup, scan_dir: str, reference_path: str, modality: str,
                 landmark_model: str = "", span=None, **extra) -> str:
    """Orient every case under `scan_dir` onto `reference_path`.

    Fully-Automated on both modalities: for CBCT that is ASO predicting the
    landmarks through ALI, for IOS it is the tooth-centroid alignment. Either
    way AREG hands over a folder and gets an oriented folder back.

    **This is the nested call.** ASO is itself supervised for CBCT, so the chain
    is AREG -> ASO -> ALI, three tools and three venvs deep. Whatever supplies
    `sup` supplies the callee's too; nothing here arranges that.
    """
    logger.info("AREG: asking 'ASO' for oriented %s scans", modality)
    parameters = {
        "input": scan_dir,
        "reference": reference_path,
        "modality": modality,
        "automation": "Fully-Automated",
        "output_suffix": "Or",
    }
    # CBCT orientation is itself landmark-driven, and ASO needs the bundle
    # NAMED: it used to be optional because the server picked one matching the
    # input, and a tool no longer resolves paths. Forgetting it is a failure
    # three tools down, so it is passed explicitly and required by _check_cbct.
    if modality == "CBCT" and landmark_model:
        parameters["landmark_model"] = landmark_model
    parameters.update(extra)
    return _returned(sup.run("ASO", **parameters, **_span(span)))


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
    parameters = {
        "meshes": mesh_dir,
        "suffix": "Seg",
    }
    if model_path:
        parameters["model"] = model_path
    return _returned(sup.run("Crown_Seg", **parameters, **_span(span)))


def predict_mucogingival(sup, mesh_dir: str, model_path: str = "", span=None) -> str:
    """Predict the 13 mucogingival landmarks on every lower arch under `mesh_dir`.

    Returns the directory holding ALI's markups files, which `ios.mgl` then
    reads exactly as it reads a folder the caller sent -- the two paths differ
    only in where the landmarks came from.

    `ios_networks` names Mucogingival ALONE. It is the one network ALI leaves
    off by default, and asking for it alone is what keeps this from also running
    the occlusal and cervical passes over every mesh for landmarks nobody wants
    here. ALI restricts it to the mandible itself, so an upper arch in the batch
    costs nothing.
    """
    logger.info("AREG: asking 'ALI_IOS' for mucogingival landmarks")
    parameters = {
        "input": mesh_dir,
        # No `modality` any more, and that is the split showing through: this
        # used to pass "IOS" explicitly because ALI was one tool serving both,
        # defaulted to CBCT, and would have been handed surfaces. ALI_IOS takes
        # no such argument -- the tool IS the modality -- so passing it now
        # fails on an unexpected keyword. The disambiguation was deleted rather
        # than moved.
        #
        # `networks`, not `ios_networks`, for the same reason: an argument that
        # only ever applied to one modality stopped needing the prefix that told
        # them apart.
        "networks": ["Mucogingival"],
        # No `prediction_ID` either, and for a third time the same reason: ALI
        # fixed its marker at `Pred` because a caller that moved it made the
        # output file name unpredictable -- nothing downstream could then find
        # the landmarks it had just asked for. Passing it fails on an
        # unexpected keyword. This call kept it three days longer than its
        # siblings because no test here reads the parameters it sends.
    }
    if model_path:
        parameters["model"] = model_path
    return _returned(sup.run("ALI_IOS", **parameters, **_span(span)))
