"""What a registration measured, for the clinician who started it.

Two quantities the registration computes anyway, kept in each region's report
entry and sent to the user's console as `metric` lines:

* the optimiser's cost, Mattes mutual information, at the iterations where it
  visibly fell. elastix and the GPU engine both minimise the NEGATIVE mutual
  information, so the cost is negative and lower is better;
* the rigid transform that was applied: its rotation angle and how far it
  moves the T1 volume's centre.

No landmark distance: this tool has no landmarks on both timepoints (ASO
orients only the T1), and a number it did not measure is not reported.

Numbers and region names only, never a file or patient name. Light on purpose
(numpy only): `dispatch` imports it on a server without SimpleITK.
"""

import math

import numpy as np

from . import progress

METRIC = "Mattes mutual information"


def cost_milestones(values, share=0.10, most=6, digits=4, floor=0.001) -> list:
    """The iterations at which the cost visibly fell, as `[[i, cost]]`.

    The first iteration, every one at least `share` of the whole descent (and
    at least `floor`) below the last milestone kept, and the last: a few
    numbers that show the optimiser converging, where every iteration would
    only show noise. A share of the descent rather than of the value, because
    the cost is negative and its size says nothing about how much it moved;
    the floor keeps a descent of a few ten-thousandths from reading as several
    improvements. Thinned to `most`, keeping
    the first and the last; two neighbours equal once rounded are one
    milestone, the later. A value that is not finite is skipped.
    """
    points = [(i, float(v)) for i, v in enumerate(values) if np.isfinite(v)]
    if not points:
        return []
    descent = points[0][1] - min(v for _i, v in points)
    kept = [points[0]]
    if descent > 0:
        for i, value in points[1:]:
            if value <= kept[-1][1] - max(share * descent, floor):
                kept.append((i, value))
    if kept[-1] != points[-1]:
        kept.append(points[-1])
    if len(kept) > most:
        step = (len(kept) - 1) / (most - 1)
        kept = [kept[round(k * step)] for k in range(most)]
    rounded = []
    for i, value in kept:
        value = round(value, digits)
        if rounded and rounded[-1][1] == value:
            rounded[-1] = [int(i), value]
        else:
            rounded.append([int(i), value])
    return rounded


def summarise_cost(levels, engine: str, computed_on: str = None) -> dict:
    """The report's record of the optimiser's cost, or None without one.

    `levels` is what an engine's `register(cost=...)` filled: one list of
    per-iteration values per resolution, coarse to fine.
    """
    levels = [list(level) for level in (levels or []) if len(level)]
    if not levels:
        return None
    record = {
        "metric": METRIC,
        "lower_is_better": True,
        "engine": engine,
        "resolutions": [{"iterations": len(level), "milestones": cost_milestones(level)}
                        for level in levels],
    }
    if computed_on:
        record["computed_on"] = computed_on
    return record


def rigid_motion(transform) -> dict:
    """`{"rotation_deg", "translation_mm"}` of a rigid SimpleITK transform.

    The rotation is the angle of its matrix about its axis,
    `acos((trace - 1) / 2)`. The translation is how far it moves its own
    centre, the T1 volume's geometric centre (both engines centre it there),
    which is the length of its translation vector. The same two numbers
    describe the inverse, T2 to T1.
    """
    matrix = np.asarray(transform.GetMatrix(), dtype=float).reshape(3, 3)
    cosine = np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0)
    centre = np.asarray(transform.GetCenter(), dtype=float)
    moved = np.asarray(transform.TransformPoint(centre.tolist()), dtype=float)
    return {
        "rotation_deg": round(math.degrees(math.acos(cosine)), 3),
        "translation_mm": round(float(np.linalg.norm(moved - centre)), 3),
    }


def header(position: int, total: int, entry: dict) -> str:
    """The line the metrics of one registration come under: which patient of
    the run, by position, and which region the cost was computed over."""
    region = str(entry.get("region") or "region").lower()
    line = "patient {}/{}, {}".format(position, total, region)
    computed_on = (entry.get("registration_cost") or {}).get("computed_on")
    if computed_on:
        line += ", cost computed on the T1 inside {}".format(computed_on)
    return line


def _sequence(milestones) -> str:
    """`a > b > c`, the sign telling which way each step went."""
    text = "{:.4f}".format(milestones[0][1])
    for (_i, before), (_j, after) in zip(milestones, milestones[1:]):
        text += " {} {:.4f}".format(">" if after < before else "<", after)
    return text


def metric_lines(entry: dict) -> list:
    """What one registration measured, one quantity per line.

    - The optimiser's cost at the finest resolution, the one whose answer is
      written, at its milestones, with that resolution's iteration count and
      the count over all of them. Mutual information is not comparable from
      one resolution to the next (each is computed on a differently blurred
      image), so the coarser ones are only counted.
    - The rigid transform applied: rotation in degrees, translation in mm.
    """
    lines = []
    cost = entry.get("registration_cost") or {}
    resolutions = cost.get("resolutions") or []
    if resolutions and resolutions[-1].get("milestones"):
        finest = resolutions[-1]
        where = ("" if len(resolutions) == 1
                 else ", finest of {} resolutions".format(len(resolutions)))
        total = sum(level.get("iterations", 0) for level in resolutions)
        lines.append("registration cost, {} (lower is better){}: {}, {} iterations{}".format(
            cost.get("metric", METRIC), where, _sequence(finest["milestones"]),
            finest["iterations"],
            "" if len(resolutions) == 1 else " ({} in all)".format(total)))
    rigid = entry.get("rigid_transform") or {}
    if rigid.get("rotation_deg") is not None and rigid.get("translation_mm") is not None:
        lines.append("rigid transform applied: rotation {:.1f}°, translation {:.1f} mm".format(
            rigid["rotation_deg"], rigid["translation_mm"]))
    return lines


def report_metrics(here: str, entry: dict) -> None:
    """Send `metric_lines` to the user, under the line saying which case."""
    lines = metric_lines(entry)
    if lines:
        progress.log(here, level="metric", user=True)
    for line in lines:
        progress.log(line, level="metric", user=True)
