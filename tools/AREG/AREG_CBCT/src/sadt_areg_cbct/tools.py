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

import json
import logging
import os

from sadt_areg_common.errors import SupervisorRequired

logger = logging.getLogger(__name__)

# What each tool is asked for, and what to do when it cannot be reached. The
# advice is the useful half: a caller who cannot run AMASSS can still send their
# own masks, and saying so beats naming a deployment problem they cannot fix.
_ADVICE = {
    "AMASSS": (
        "Send your own T1 segmentation masks in 't1_masks' and use Semi-Automated "
        "mode instead."
    ),
    "ASO": (
        "Orient the T1 and T2 scans yourself beforehand, and use the mode that takes "
        "them already oriented."
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
    # The landmarks to orient on are the ones the REFERENCE defines. Left out,
    # ASO falls back to its own defaults -- the Frankfurt set -- which only
    # happen to match the Frankfurt reference: the occlusal one defines none of
    # them, and every run with it was refused before ALI was even asked.
    if modality == "CBCT" and "cbct_landmarks" not in extra:
        labels = reference_landmarks(reference_path)
        if labels:
            parameters["cbct_landmarks"] = labels
    parameters.update(extra)
    return _returned(sup.run("ASO", **parameters, **_span(span)))


def reference_landmarks(reference_path: str) -> list:
    """The landmark labels the reference's markups file defines, in its order,
    or [] when there is none to read.

    Read from the reference itself rather than listed per reference here, so a
    new reference orients on its own landmarks without this file changing.
    """
    candidates = []
    if os.path.isfile(reference_path):
        candidates = [reference_path]
    elif os.path.isdir(reference_path):
        for directory, _subdirs, names in sorted(os.walk(reference_path)):
            candidates += [os.path.join(directory, name) for name in sorted(names)
                           if name.endswith(".mrk.json") and not name.startswith(".")]
    for path in candidates:
        try:
            with open(path, encoding="utf-8") as handle:
                document = json.load(handle)
        except (OSError, ValueError):
            continue
        labels = []
        for markup in document.get("markups") or []:
            for point in markup.get("controlPoints") or []:
                label = point.get("label")
                if isinstance(label, str) and label and label not in labels:
                    labels.append(label)
        if labels:
            return labels
    return []


def segment_masks(sup, scan_dir: str, model_path: str, mask_structures, span=None) -> str:
    """Segment every scan under `scan_dir` into the requested mask structures.

    Returns the directory holding AMASSS's output, which `cbct.pipeline.find_masks`
    then reads exactly as it reads a mask folder the caller sent -- the automated
    and semi-automated paths differ only in where the masks came from.

    `mask_structures` are AMASSS structure codes (CBMASK/MANDMASK/MAXMASK), and
    the packaged tool takes codes directly. The in-process version had to
    translate them into display names through AMASSS's own table; the schema
    publishes the codes now, so the translation is gone rather than restated.
    """
    logger.info("AREG: asking 'AMASSS' for T1 masks (%s)", ", ".join(mask_structures))
    return _returned(sup.run(
        "AMASSS",
        scans=scan_dir,
        model=model_path,
        structures=list(mask_structures),
        # One binary file per structure: `find_masks` looks each region's mask
        # up by name, and a merged multi-label volume would make every region
        # resolve to the same file.
        #
        # Which is why the original module's "Merge Segmentations" box is not
        # offered here: honouring it would mean a SECOND pass over the same
        # scan, the registration needing the separate masks either way, and
        # the card is serialised. Decided 2026-09-25, not overlooked.
        merge=["SEPARATE"],
        prediction_ID="seg",
        generate_surface=False,
        **_span(span),
    ))
