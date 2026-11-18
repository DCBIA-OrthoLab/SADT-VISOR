"""The measurements themselves: distances, angles, and what their signs mean.

Ported from `VFACE_utils/Measure.py`, `Point.py` and `Line.py`, which are the
AQ3DC measurement classes. The arithmetic is kept as upstream wrote it; the
envelope is not. Upstream's `Measure` is a Qt-aware object with a `keep_sign`
checkbox, string-keyed `__getitem__` and a `position` dictionary threaded
through `__setitem__`; here a measurement is a description of what to compute
and `compute()` returns a row.

**Two halves, and only one of them is arithmetic.** The components are simple:
a difference of two points, or the rejection of a point onto a line, or an
angle read in each of three planes. What takes the space is the SIGN MEANING --
the label that says which way a positive number points, which is not the same
question for a right upper molar as for the cranial base. Those tables are the
measurement's clinical content and they are reproduced exactly; they are the
reason a component can be turned back into a signed feature later.

One deliberate change, forced and behaviour-preserving: upstream takes
`np.cross` of two 2-vectors, which NumPy 2.0 removed. The scalar it used to
return is `a[0]*b[1] - a[1]*b[0]`, which upstream computes on the very next
line anyway, so the cross product is replaced by that value and its magnitude.
"""

import logging
import math

import numpy as np

logger = logging.getLogger(__name__)

# The four things a measurement list can ask for. Upstream spells the time into
# the same string (`"Distance between 2 points T1"`); here the kind and the
# timepoint are separate fields, because one of them selects arithmetic and the
# other selects which landmark file to read.
DISTANCE_2_POINTS = "Distance between 2 points"
DISTANCE_POINT_LINE = "Distance point line"
ANGLE_2_LINES = "Angle between 2 lines"
ANGLE_LINE_T1_T2 = "Angle line T1 and line T2"

KINDS = (DISTANCE_2_POINTS, DISTANCE_POINT_LINE, ANGLE_2_LINES, ANGLE_LINE_T1_T2)

# Which tooth sits where. A sign means something different in each of the eight
# groups -- "towards the midline" is one direction on the right and the other on
# the left -- so the group is what selects the labels.
UPPER_RIGHT_BACK = ("UR8", "UR7", "UR6", "UR5", "UR4", "UR3")
UPPER_RIGHT_FRONT = ("UR1", "UR2")
UPPER_LEFT_BACK = ("UL8", "UL7", "UL6", "UL5", "UL4", "UL3")
UPPER_LEFT_FRONT = ("UL1", "UL2")
LOWER_RIGHT_BACK = ("LR8", "LR7", "LR6", "LR5", "LR4", "LR3")
LOWER_RIGHT_FRONT = ("LR1", "LR2")
LOWER_LEFT_BACK = ("LL8", "LL7", "LL6", "LL5", "LL4", "LL3")
LOWER_LEFT_FRONT = ("LL1", "LL2")

# Every tooth the sign tables know, for deciding dental against skeletal.
TEETH = tuple(
    f"{jaw}{side}{number}"
    for jaw in ("L", "U")
    for side in ("L", "R")
    for number in range(1, 8)
)

# What a NEGATIVE and a POSITIVE component mean, per group, for a distance
# between two teeth. Upstream writes this as eight if-blocks of nine lines; it
# is one table, and reading it as one is how you can see that the upper and
# lower halves differ only in the superior-inferior column.
#
#                          (lr-, lr+)   (ap-, ap+)   (si-, si+)
_DENTAL_DISTANCE = {
    UPPER_RIGHT_BACK:   (("L", "B"), ("D", "M"), ("E", "I")),
    UPPER_RIGHT_FRONT:  (("M", "D"), ("L", "B"), ("E", "I")),
    UPPER_LEFT_BACK:    (("B", "L"), ("D", "M"), ("E", "I")),
    UPPER_LEFT_FRONT:   (("D", "M"), ("L", "B"), ("E", "I")),
    LOWER_RIGHT_BACK:   (("L", "B"), ("D", "M"), ("I", "E")),
    LOWER_RIGHT_FRONT:  (("M", "D"), ("L", "B"), ("I", "E")),
    LOWER_LEFT_BACK:    (("B", "L"), ("D", "M"), ("I", "E")),
    LOWER_LEFT_FRONT:   (("D", "M"), ("L", "B"), ("I", "E")),
}

# The same, for an angle between two lines both drawn on teeth. The columns are
# yaw, pitch, roll rather than left-right, anterior-posterior, superior-inferior.
#
#                          (yaw-, yaw+)   (pitch-, pitch+)  (roll-, roll+)
_DENTAL_ANGLE = {
    UPPER_RIGHT_BACK:   (("DR", "MR"), ("D", "M"), ("L", "B")),
    UPPER_RIGHT_FRONT:  (("DR", "MR"), ("L", "B"), ("M", "D")),
    UPPER_LEFT_BACK:    (("MR", "DR"), ("D", "M"), ("B", "L")),
    UPPER_LEFT_FRONT:   (("MR", "DR"), ("L", "B"), ("D", "M")),
    LOWER_RIGHT_BACK:   (("DR", "MR"), ("M", "D"), ("B", "L")),
    LOWER_RIGHT_FRONT:  (("DR", "MR"), ("B", "L"), ("D", "M")),
    LOWER_LEFT_BACK:    (("MR", "DR"), ("M", "D"), ("L", "B")),
    LOWER_LEFT_FRONT:   (("MR", "DR"), ("B", "L"), ("M", "D")),
}


class MeasurementError(Exception):
    """One measurement could not be computed. Recorded; the batch goes on."""


# ---------------------------------------------------------------------------
# The arithmetic
# ---------------------------------------------------------------------------

def _components(delta) -> tuple:
    """Upstream's component convention, and its rounding.

    The first two axes are NEGATED and the third is not. That is not a
    coincidence of sign: a Slicer volume is in LPS, so +x runs to the patient's
    left and +y posteriorly, while the columns this feeds are read as
    right-positive and anterior-positive. The third axis is superior in both.
    """
    return (
        round(float(-delta[0]), 3),
        round(float(-delta[1]), 3),
        round(float(delta[2]), 3),
        round(float(np.linalg.norm(delta)), 3),
    )


def distance_between_points(point1, point2) -> tuple:
    """`(right-left, anterior-posterior, superior-inferior, 3D)` between two points."""
    return _components(np.asarray(point2, dtype=float) - np.asarray(point1, dtype=float))


def distance_point_to_line(point, line_start, line_end) -> tuple:
    """The same four numbers, from a point to the line through two others.

    The rejection of `point - line_end` onto `line_start - line_end`: what is
    left of the offset once the part running ALONG the line is taken out, which
    is the perpendicular distance and its components.

    Two identical line points describe no line, and upstream falls back to the
    plain point-to-point distance rather than dividing by zero. Kept: a
    measurement list naming the same landmark twice is a mistake in the list,
    and answering it with a distance is more use than a NaN.
    """
    point = np.asarray(point, dtype=float)
    line_start = np.asarray(line_start, dtype=float)
    line_end = np.asarray(line_end, dtype=float)

    if np.allclose(line_start, line_end, atol=1e-5):
        return _components(point - line_start)

    axis = line_start - line_end
    offset = point - line_end
    return _components(offset - axis * (np.dot(offset, axis) / np.dot(axis, axis)))


def _angle_in_plane(line1, line2, dropped_axis, start2, end1) -> float:
    """The angle between two lines seen in one plane, signed.

    `dropped_axis` is the axis looked ALONG: dropping it projects both lines
    into the plane, and the angle there is the rotation about it. 2 gives yaw
    (seen from above), 0 pitch (from the side), 1 roll (from the front).
    """
    mask = [True, True, True]
    mask[dropped_axis] = False
    first, second = np.asarray(line1)[mask], np.asarray(line2)[mask]

    if np.linalg.norm(first) == 0 or np.linalg.norm(second) == 0:
        raise MeasurementError(
            "one of the two lines has no length once projected -- it runs along "
            "the axis the angle is read about"
        )
    first = first / np.linalg.norm(first)
    second = second / np.linalg.norm(second)

    # `np.cross` of two 2-vectors was removed in NumPy 2.0. The scalar it
    # returned is exactly this, which upstream computes on the next line anyway.
    turn = float(first[0] * second[1] - first[1] * second[0])
    degrees = math.degrees(math.atan2(abs(turn), float(np.dot(first, second))))

    # Two lines that meet head to tail describe the angle's supplement.
    if np.all(np.asarray(end1) == np.asarray(start2)):
        degrees = 180.0 - degrees

    if (turn < 0) == (dropped_axis == 2):
        return -degrees
    return degrees


def angles_between_lines(point1, point2, point3, point4) -> tuple:
    """`(yaw, pitch, roll)` between the line p1->p2 and the line p3->p4.

    Read in the three anatomical planes: yaw about the superior axis (the axial
    view), pitch about the right-left axis (sagittal), roll about the
    anterior-posterior axis (coronal).
    """
    line1 = np.asarray(point2, dtype=float) - np.asarray(point1, dtype=float)
    line2 = np.asarray(point4, dtype=float) - np.asarray(point3, dtype=float)
    return tuple(
        round(_angle_in_plane(line1, line2, axis, point3, point2), 3)
        for axis in (2, 0, 1)
    )


# ---------------------------------------------------------------------------
# What the sign means
# ---------------------------------------------------------------------------

def is_dental(name: str) -> bool:
    """True when a landmark name names a tooth.

    A substring test, and deliberately: the names are `UR1O`, `UR6MB`,
    `Mid_UR6O_UL6O`, so what is looked for is the tooth inside the name rather
    than the name itself.
    """
    return any(tooth in name.upper() for tooth in TEETH)


def _all_in_group(names, group) -> bool:
    """Whether every name belongs to one quadrant-and-segment group.

    Upstream counts substring hits and requires the count to equal the number of
    names, plus one more for each midpoint -- a midpoint names two teeth and so
    has to match twice. Reproduced as written: it is what decides which sign
    table a measurement is read with.
    """
    matched = 0
    midpoints = 0
    for name in names:
        if "MID" in name.upper():
            midpoints += 1
        for tooth in group:
            if tooth in name:
                matched += name.count(tooth)
    return matched == len(names) + midpoints


def _group_of(names):
    for group in _DENTAL_DISTANCE:
        if _all_in_group(names, group):
            return group
    return None


def _side_of(name):
    """The side letter upstream reads off a landmark, for a distance's label.

    **The first character of the name, and nothing cleverer.** Upstream has a
    block above this that resolves a midpoint's side from the two landmarks it
    joins -- and then overwrites its own answer: the block replaces
    `direction` with a letter, which makes the guard below it (`if
    direction1 != "Mid" and direction2 != "Mid"`) always true, and that guard
    re-reads `name[0]`. So `Mid_ROr_LOr` resolves to `M`, not to `R` or `L`,
    and the midpoint logic never reaches a result.

    That is reproduced rather than repaired. The label decides the SIGN of the
    feature the classifier is trained on, so quietly making it read `R` where
    every model saw `M` would change classifications on the models already
    trained -- which is a clinical decision and not a port's to take. It is
    written down here, and in the README, as something to fix deliberately.

    `None` is what a point-to-line measurement's second operand gives: upstream
    indexes the Line object itself, and `Line.__getitem__(0)` returns None.
    """
    if name is None:
        return None
    return name[0] if name else "No_direction"


def skeletal_distance_meanings(name1, name2, lr, ap, si) -> tuple:
    """The labels for a distance between landmarks that are not both teeth.

    `name2` is None for a point-to-line distance, which is what upstream ends
    up with there -- see `_side_of`.

    The left-right label carries the content: where both operands agree on a
    side it reads `Medial`/`Lateral` -- towards or away from the midline -- and
    where they name opposite sides there is no such direction and the label is
    `x`. Only where neither carries a side does it fall back to plain `R`/`L`.
    """
    side1, side2 = _side_of(name1), _side_of(name2)

    if side1 == side2:
        side = side1
    elif side1 == "No_direction":
        side = side2
    elif side2 == "No_direction":
        side = side1
    elif {side1, side2} == {"R", "L"}:
        side = "No_direction"
    else:
        side = None

    if side == "R":
        lateral = "Lateral" if lr > 0 else "Medial"
    elif side == "L":
        lateral = "Medial" if lr > 0 else "Lateral"
    elif side == "No_direction":
        lateral = "x"
    else:
        lateral = "R" if lr > 0 else "L"

    return lateral, ("A" if ap > 0 else "P"), ("S" if si > 0 else "I")


def dental_distance_meanings(name1: str, name2: str, lr, ap, si) -> tuple:
    """The labels for a distance between two points on teeth, or None.

    None when the two names do not both belong to one quadrant-and-segment
    group: buccal and lingual are not the same direction on the two sides of
    the arch, so a measurement spanning them has no single label -- and upstream
    leaves the meanings empty rather than picking one.
    """
    group = _group_of([name1, name2])
    if group is None:
        return None
    return tuple(
        positive if value > 0 else negative
        for (negative, positive), value in zip(_DENTAL_DISTANCE[group], (lr, ap, si))
    )


def dental_angle_meanings(names, yaw, pitch, roll) -> tuple:
    """The labels for an angle between two lines drawn on teeth, or None."""
    group = _group_of(list(names))
    if group is None:
        return None
    return tuple(
        positive if value > 0 else negative
        for (negative, positive), value in zip(_DENTAL_ANGLE[group], (yaw, pitch, roll))
    )


def cross_timepoint_angle_meanings(yaw, pitch, roll) -> tuple:
    """The labels for an angle between one timepoint's line and the other's.

    There is no anatomy to name here -- both lines are the same anatomy, seen
    at two times -- so what a sign says is which way it turned.
    """
    return (
        "CounterC" if yaw > 0 else "ClockWise",
        "CounterC" if pitch > 0 else "ClockWise",
        "Clockwise" if roll > 0 else "CounterC",
    )
