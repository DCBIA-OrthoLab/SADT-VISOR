"""Pair an intraoral scan with a CBCT of the same patient, and register it.

NOT longitudinal, unlike its two siblings. AREG_CBCT and AREG_IOS take a
baseline and a follow-up of one modality; this takes ONE timepoint imaged two
ways, and puts the intraoral scan into the CBCT's frame. Upstream's own test
set says so plainly -- `P001_T2_U.vtk` beside `P_0001_T2.nii.gz`, both T2.

That is why the arguments are `ios` and `cbct` rather than `t1` and `t2`, and
why `pairing.pair()` is not what pairs them: there is no timepoint to strip,
only a patient to match across two naming conventions.
"""

import json
import logging
import os

import numpy as np

from sadt_areg_common import pairing
from sadt_areg_common.errors import ToolInputError

from . import geometry

logger = logging.getLogger(__name__)

SURFACE_EXTENSIONS = (".vtk", ".stl")
LANDMARK_EXTENSIONS = (".json", ".mrk.json")


def patient_key(filename: str) -> str:
    """`P001_T2_U.vtk` and `P_0001_T2.nii.gz` are the same patient.

    The two modalities are named by different conventions -- the intraoral files
    by the scanner, the CBCT by the acquisition -- so the digits are what they
    genuinely share. Everything that is not a digit is dropped and leading zeros
    go with it, which makes `P001` and `P_0001` both `1`.

    Deliberately cruder than `pairing.patient_stem`, and only used here: that
    function matches two files that came from the SAME source and can rely on a
    shared stem. Across modalities there is no shared stem to rely on.

    Public because `dispatch` keys landmark files by it too: a landmark file
    belongs to the patient its name names, and matching one to a mesh by the
    jaw token alone is how a two-patient batch registered one patient against
    another's points.
    """
    stem = pairing.split_scan_extension(os.path.basename(filename))[0]
    # The trailing timepoint digit is part of the name, not the patient: strip
    # the tokens that name one before reducing to digits.
    tokens = [t for t in pairing.tokens(stem) if t not in ("t0", "t1", "t2")]
    digits = "".join(c for token in tokens for c in token if c.isdigit())
    return digits.lstrip("0") or digits or stem


def discover(ios_dir: str, cbct_dir: str) -> tuple:
    """`({patient: {"ios": [paths], "cbct": path}}, {patient: why})`.

    A patient with only one modality is reported rather than silently dropped:
    a batch that registered half of what was sent and said nothing is the
    failure this repository keeps finding.

    Both walks are ordered -- the subdirectories sorted in place, the files
    sorted -- and a patient with several CBCTs keeps the FIRST, the same rule
    `pairing.discover` uses for the longitudinal engines. It used to keep
    whichever `os.walk` happened to reach last, so which volume a patient was
    registered onto depended on the order the filesystem returned directories
    in: the same request could give two different answers on two machines.
    """
    ios: dict = {}
    for root, directories, files in os.walk(ios_dir):
        directories.sort()
        for name in sorted(files):
            if name.lower().endswith(SURFACE_EXTENSIONS):
                ios.setdefault(patient_key(name), []).append(os.path.join(root, name))

    cbct: dict = {}
    for root, directories, files in os.walk(cbct_dir):
        directories.sort()
        for name in sorted(files):
            if pairing.is_scan_file(name):
                cbct.setdefault(patient_key(name), os.path.join(root, name))

    paired, unpaired = {}, {}
    for key in sorted(set(ios) | set(cbct)):
        if key in ios and key in cbct:
            paired[key] = {"ios": sorted(ios[key]), "cbct": cbct[key]}
        else:
            unpaired[key] = "no CBCT" if key in ios else "no intraoral scan"

    if not paired:
        raise ToolInputError(
            "No patient has both an intraoral scan and a CBCT. Found "
            f"{len(ios)} intraoral key(s) and {len(cbct)} CBCT key(s): {unpaired}."
        )
    if unpaired:
        # The counts, never the keys: a key is the caller's own file name, and
        # this line reaches the server's log. `unpaired` is returned as it is,
        # and the run report names every one of them -- that goes back to
        # whoever sent the data.
        without_cbct = sum(1 for reason in unpaired.values() if reason == "no CBCT")
        logger.warning(
            "Not registered, only one modality present: %d patient(s) with no "
            "CBCT, %d with no intraoral scan",
            without_cbct, len(unpaired) - without_cbct,
        )
    return paired, unpaired


def read_landmarks(path: str) -> dict:
    """`{label: [x, y, z]}` from a Slicer markups file or a plain JSON one.

    Both spellings are in upstream's own test set -- `.mrk.json` for the CBCT
    side, `.json` for the intraoral -- so both are read rather than one being
    declared canonical.
    """
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)

    points = {}
    for markup in payload.get("markups", []):
        for control_point in markup.get("controlPoints", []):
            label = control_point.get("label")
            position = control_point.get("position")
            if label and position:
                points[label] = [float(value) for value in position]
    if points:
        return points

    # The plainer shape: {label: [x, y, z]} at the top level.
    for label, position in payload.items():
        if isinstance(position, (list, tuple)) and len(position) == 3:
            points[label] = [float(value) for value in position]
    return points


def shared_landmarks(moving: dict, fixed: dict) -> tuple:
    """The points both sides name, in one order, plus what was dropped.

    Intersected rather than assumed equal: the two modalities are landmarked by
    different networks, and one missing point on one side used to be an
    IndexError three frames down instead of a line in a report.
    """
    common = sorted(set(moving) & set(fixed))
    dropped = sorted((set(moving) | set(fixed)) - set(common))
    if len(common) < 3:
        raise ToolInputError(
            f"The two modalities share only {len(common)} landmark(s), and an "
            f"alignment needs 3. Intraoral has {sorted(moving)}; CBCT has {sorted(fixed)}."
        )
    return (
        np.array([moving[label] for label in common], dtype=float),
        np.array([fixed[label] for label in common], dtype=float),
        common,
        dropped,
    )


def register_one(mesh, ios_landmarks: dict, cbct_landmarks: dict,
                 cbct_surface=None, on_enamel: dict = None,
                 max_dist: float = geometry.ICP_MAX_DIST_MM):
    """Place the arch on its landmarks, then let the CBCT surface correct it.

    Two stages, and the order is load-bearing. The landmark fit puts the two
    meshes in roughly the same place so the ICP starts inside its capture range;
    an ICP started on unaligned meshes converges to whatever local minimum it
    happens to reach. What the fit cannot do is notice that it is wrong: six
    points are perfectly consistent with themselves however badly one of them
    was predicted, so the arch follows the bad one and the residual looks like
    ordinary noise. Forty thousand crown points against the CBCT surface is what
    notices.

    Without a surface this degrades to the landmark fit alone, which is what the
    "Registration" mode does -- the arch is placed, nothing corrects it, and the
    report says the ICP did not run.
    """
    moving, fixed, used, dropped = shared_landmarks(ios_landmarks, cbct_landmarks)
    matrix, prealignment = geometry.prealign(moving, fixed, used, on_enamel)
    report = {
        "landmarks_used": used,
        "landmarks_dropped": dropped,
        "prealignment": prealignment,
        "icp": None,
    }
    if cbct_surface is None:
        return matrix, report

    target = geometry.Target.from_mesh(cbct_surface).around(fixed)
    moved = mesh.transform(matrix, inplace=False)
    refinement, stats = geometry.icp_point_to_plane(moved, target, max_dist=max_dist)
    report["icp"] = stats

    if stats["fitness"] < geometry.MIN_ICP_FITNESS:
        # Refused rather than written out: an ICP that matched nothing still
        # returns a matrix, and writing it put an untouched intraoral scan in the
        # results under the name of a registered one.
        raise RuntimeError(
            "Only %.1f%% of this arch matched the CBCT surface, under the %.0f%% "
            "floor. Nothing is written for it: check its landmarks -- the "
            "pre-alignment %s."
            % (100 * stats["fitness"], 100 * geometry.MIN_ICP_FITNESS,
               "was accepted" if prealignment["accepted"] else "was skipped")
        )

    # The one check on the ICP that the ICP does not grade itself. Measured over
    # the pairs the pre-alignment was fitted on, so the two numbers are the same
    # measurement taken twice: including a pair the pre-alignment rejected would
    # compare the ICP against a residual nothing tried to minimise, and report a
    # drift on an arch that had not moved.
    report["icp"].update(
        _landmark_drift(moving, fixed, used, matrix, refinement,
                        prealignment["landmarks_fitted"], stats))
    return refinement @ matrix, report


def _landmark_drift(moving, fixed, labels, matrix, refinement, fitted, stats) -> dict:
    """Whether the ICP moved the arch towards its landmarks or away from them.

    Only logged, never refused: a registration can legitimately trade a little
    landmark agreement for a much better surface fit, and on these six points it
    usually should. What it must not do is trade it silently.
    """
    keep = [index for index, label in enumerate(labels) if label in set(fitted)]
    if not keep:
        return {}

    placed = geometry.apply(moving[keep], matrix)
    before = float(np.sqrt(np.mean(
        np.sum((placed - fixed[keep]) ** 2, axis=1))))
    after = float(np.sqrt(np.mean(
        np.sum((geometry.apply(placed, refinement) - fixed[keep]) ** 2, axis=1))))

    if after - before > geometry.MAX_ICP_LANDMARK_DRIFT_MM:
        logger.warning(
            "The ICP left the arch %.1f mm from its landmarks where the fit had "
            "it at %.1f mm, a drift of %.1f mm. It matched %.1f%% of the arch at "
            "%.2f mm, so the surface agrees with it -- but on a smooth occlusal "
            "surface that pattern is also what a fit sliding along the arch looks "
            "like, and its own fitness cannot tell the two apart.",
            after, before, after - before, 100 * stats["fitness"],
            stats["inlier_rmse"])
    else:
        logger.info("Landmarks %.1f mm from their CBCT counterparts after the ICP, "
                    "against %.1f mm before it.", after, before)
    return {"landmark_rms_before_mm": before, "landmark_rms_after_mm": after}
