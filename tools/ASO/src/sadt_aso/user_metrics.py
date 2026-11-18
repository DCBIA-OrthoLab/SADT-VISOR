"""What an orientation measured, shown to the person who started it.

Both engines register predicted or supplied points onto a gold reference's
points, first by a three-point fit and then by an ICP. This module turns what
they measured into `metric` lines for the clinician's console: the landmarks
left out of the fit and why, the distance of each point to its reference
counterpart after each step, what the ICP itself ended on, and the rotation
and translation applied.

The numbers are computed by the engines and stored in the per-scan report
entry under `"metrics"` (so the JSON report carries them too); this module
only measures with numpy and formats. Never a file name, only positions in
the batch and landmark or tooth names.
"""

import numpy as np

from . import progress

# The last step is named for what the user gets, not for the algorithm.
STEP_ORIENTATION = "orientation"
STEP_THREE_POINT = "the initial three-point fit"
STEP_OCCLUSAL_PLANE = "the occlusal-plane alignment"

# What the points are called, singular and plural.
POINTS_LANDMARK = ("landmark", "landmarks")
POINTS_CENTROID = ("tooth centroid", "teeth")


def distances(moved: dict, target: dict, labels, centred: bool = False) -> list:
    """Euclidean distance, in mm, from each moved point to its reference.

    `centred` first moves the points so their mean sits on the reference's:
    for an orientation that applies a rotation only, the distance then says
    how well the rotation fits, without the offset between the two frames'
    origins, which nothing in the orientation was meant to remove.
    """
    moved = {name: np.asarray(moved[name], dtype=float) for name in labels}
    target = {name: np.asarray(target[name], dtype=float) for name in labels}
    if centred and labels:
        shift = (np.mean([target[n] for n in labels], axis=0)
                 - np.mean([moved[n] for n in labels], axis=0))
        moved = {name: point + shift for name, point in moved.items()}
    return [float(np.linalg.norm(moved[name] - target[name])) for name in labels]


def closest_point_rms(moved, target) -> float:
    """RMS distance from each moved point to the NEAREST target point.

    The quantity `vtkIterativeClosestPointTransform` minimises and reports as
    its mean distance in RMS mode. VTK only computes that mean when its
    `CheckMeanDistance` is on, which also lets it stop early; turning it on
    would change the orientation, so the same quantity is recomputed here on
    the ICP's final pose instead.
    """
    moved = np.asarray(list(moved), dtype=float)
    target = np.asarray(list(target), dtype=float)
    gaps = np.linalg.norm(moved[:, None, :] - target[None, :, :], axis=2).min(axis=1)
    return float(np.sqrt(np.mean(gaps ** 2)))


def rotation_and_translation(matrix) -> tuple:
    """`(rotation angle in degrees, translation length in mm)` of a 4x4.

    The angle of the rotation part about its own axis, from its trace; the
    translation is the length of the matrix's offset column, i.e. where the
    coordinate origin is sent.
    """
    matrix = np.asarray(matrix, dtype=float)
    cosine = np.clip((np.trace(matrix[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine))), float(np.linalg.norm(matrix[:3, 3]))


def _spread(values) -> str:
    """`mean ± std mm` for a list of distances (the population std)."""
    values = np.asarray(values, dtype=float)
    return "{:.2f} ± {:.2f} mm".format(float(np.mean(values)), float(np.std(values)))


def metric_lines(metrics: dict) -> list:
    """What the orientation measured, one quantity per line, for the user.

    - Every landmark the fit left out, and why.
    - For each step, in order, the distance of each point to its reference
      counterpart: mean ± population std, the furthest point, and the count.
      The last step is the orientation the user receives.
    - The ICP: how many iterations it ran and the RMS distance from each point
      to the nearest reference point at its final pose.
    - The rotation and translation the orientation applies.

    Numbers and landmark or tooth names only, never a file name.
    """
    if not metrics:
        return []
    lines = []
    for label, why in sorted((metrics.get("left_out") or {}).items()):
        lines.append("{} left out of the fit: {}".format(label, why))
    if metrics.get("moved_with"):
        lines.append("moved with the {} jaw's transform, not fitted on its own".format(
            str(metrics["moved_with"]).lower()))
    singular, plural = metrics.get("points") or POINTS_LANDMARK
    labels = metrics.get("labels") or []
    for stage in metrics.get("stages") or []:
        values = stage.get("distances_mm") or []
        if not values or len(values) != len(labels):
            continue
        worst = int(np.argmax(values))
        lines.append("{} distance to the reference after {}{}: {}, max {:.2f} mm ({}), "
                     "{} {}".format(singular, stage["after"],
                                    ", centres matched" if stage.get("centred") else "",
                                    _spread(values), values[worst], labels[worst],
                                    len(values), plural))
    icp = metrics.get("icp") or {}
    if icp.get("rms_closest_mm") is not None:
        # No iteration count: this ICP has no convergence test and always runs
        # its fixed maximum, which says nothing about the case.
        lines.append("ICP closest-point RMS distance at its final pose: {:.2f} mm".format(
            icp["rms_closest_mm"]))
    if metrics.get("rotation_deg") is not None:
        lines.append("orientation applied: rotation {:.1f}°, translation {:.1f} mm".format(
            metrics["rotation_deg"], metrics.get("translation_mm", 0.0)))
    return lines


def report_metrics(here: str, metrics: dict) -> None:
    """Send `metric_lines` to the user, under one line saying which scan."""
    lines = metric_lines(metrics)
    if lines:
        progress.log(here, level="metric", user=True)
    for line in lines:
        progress.log(line, level="metric", user=True)
