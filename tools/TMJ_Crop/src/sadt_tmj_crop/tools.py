"""How this tool reaches the three it depends on, through the supervisor.

TMJ_Crop predicts nothing itself. The orientation comes from ASO (which runs
ALI_CBCT for the Frankfurt landmarks, rejects outliers and rotates the scan) and
the mandible mask from AMASSS, each in its own virtualenv, started as a
subprocess by the supervisor. Convert turns an oriented scan that ASO wrote in
another format (it keeps the input's) into the NIfTI this tool reads.

**Tools are named by string, never imported**, and `describe.py` reads this
file to publish the `calls` list the server checks at startup.

No `output_dir` and no model path is passed to either. Where a supervised tool
writes is the supervisor's business, and each tool resolves its OWN weights and
reference bundle from its own data folder. What the call RETURNS is where to
read from.
"""

import logging

from .errors import SupervisorRequired

logger = logging.getLogger(__name__)

# The frame the crop is defined in. Fixed, not an argument: the search window
# around each porion and the box placement are millimetres in THIS frame.
FRAME = "Frankfurt horizontal"

def require(sup) -> None:
    """Refuse a run that cannot work, before a single file is read."""
    if sup is not None:
        return
    raise SupervisorRequired(
        "TMJ_Crop runs the 'ASO', 'AMASSS' and 'Convert' tools itself, and nothing here can "
        "run them: no supervisor was supplied. Call it through a VISOR server."
    )


def _returned(produced) -> str:
    """A tool returns a Path, or a dict of named ones; we want a directory."""
    if isinstance(produced, dict):
        produced = next(iter(produced.values()))
    return str(produced)


def _span(span) -> dict:
    """`_progress` for `sup.run`, or nothing when the caller gave no span."""
    return {"_progress": tuple(span)} if span else {}


def _waypoint(sup, span, message: str) -> None:
    """Say which tool the run is about to be inside."""
    if sup is None:
        return
    if span and hasattr(sup, "progress"):
        sup.progress(span[0], message)
    elif hasattr(sup, "log"):
        sup.log(message)


def orient(sup, scans: str, span=None) -> str:
    """Orient every CBCT onto the Frankfurt horizontal frame, with ASO."""
    _waypoint(sup, span, "orienting the CBCT with ASO")
    parameters = {
        "input": scans,
        "modality": "CBCT",
        "automation": "Fully-Automated",
        "frame": FRAME,
    }
    return _returned(sup.run("ASO", **parameters, **_span(span)))


def segment_mandible(sup, scans: str, span=None) -> str:
    """The mandible mask of every oriented CBCT, one file each, with AMASSS."""
    _waypoint(sup, span, "segmenting the mandible with AMASSS")
    parameters = {
        "scans": scans,
        "structures": ["MAND"],
        "merge": ["SEPARATE"],
    }
    return _returned(sup.run("AMASSS", **parameters, **_span(span)))


def convert(sup, scans: str) -> str:
    """Write every volume under `scans` as NIfTI, with Convert."""
    _waypoint(sup, None, "converting the oriented scans to NIfTI")
    return _returned(sup.run("Convert", scans=scans, to="nii.gz"))
