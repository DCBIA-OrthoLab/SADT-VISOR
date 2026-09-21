"""This port against upstream's own code, transcribed verbatim.

A repackaging makes one claim -- that the numbers are the same -- and the only
way to check it is to run both. Upstream's `Measure` class cannot be imported
here: it pulls Qt, and it reads its input through a chain of `__setitem__`
calls on Point and Line objects. So the parts that DO the arithmetic are
transcribed below, line for line, and driven against the port on random input.

**The transcription is the reference, not the port.** Where the two disagree
the port is wrong, including where upstream's own logic is odd -- and two
pieces of it are: the midpoint side resolution that overwrites its own answer,
and the `Line` object indexed as though it were a string. Both are reproduced,
and both are covered here.

One change is forced rather than chosen: `np.cross` of two 2-vectors was
removed in NumPy 2.0. The scalar it returned is `a[0]*b[1] - a[1]*b[0]`, which
upstream computes on the very next line anyway, so the transcription uses that
and the two remain the same function.
"""

import itertools
import random
import re

import numpy as np

from sadt_vface import features, measure


# ---------------------------------------------------------------------------
# Upstream: Measure.Distance.__computeDistance / __reject / __computeLinePoint
# ---------------------------------------------------------------------------

def up_compute_distance(p1, p2):
    delta = p2 - p1
    norm = np.linalg.norm(delta)
    return (round(-delta[0], 3), round(-delta[1], 3), round(delta[2], 3), round(norm, 3))


def up_reject(vec, axis):
    vec = np.asarray(vec)
    axis = np.asarray(axis)
    return vec - axis * (np.dot(vec, axis) / np.dot(axis, axis))


def up_compute_line_point(line1, line2, point):
    if np.allclose(line1, line2, atol=1e-5):
        delta = point - line1
    else:
        delta = up_reject(point - line2, line1 - line2)
    norm = np.linalg.norm(delta)
    return (round(-delta[0], 3), round(-delta[1], 3), round(delta[2], 3), round(norm, 3))


# ---------------------------------------------------------------------------
# Upstream: Measure.Angle.__computeAngle / __computeAngles
# ---------------------------------------------------------------------------

def up_compute_angle(line1, line2, axis, point1, point2, point3, point4):
    mask = [True] * 3
    mask[axis] = False
    line1 = line1[mask]
    line2 = line2[mask]
    norm1 = np.linalg.norm(line1)
    norm2 = np.linalg.norm(line2)
    if norm1 == 0 or norm2 == 0:
        raise ZeroDivisionError(line1, line2)
    line1 = line1 / norm1
    line2 = line2 / norm2
    produit_scalaire = np.dot(line1, line2)
    crossz = line1[0] * line2[1] - line1[1] * line2[0]   # np.cross, NumPy 1.x
    radians = np.arctan2(np.linalg.norm(crossz), produit_scalaire)
    degree = np.degrees(radians)
    if np.all(point2 == point3):
        degree = 180 - degree
    z = line1[0] * line2[1] - line1[1] * line2[0]
    if z < 0:
        return -degree if axis == 2 else degree
    return degree if axis == 2 else -degree


def up_compute_angles(p1, p2, p3, p4):
    line1 = p2 - p1
    line2 = p4 - p3
    return tuple(round(up_compute_angle(line1, line2, axis, p1, p2, p3, p4), 3)
                 for axis in (2, 0, 1))


# ---------------------------------------------------------------------------
# Upstream: Measure.check and the eight sign blocks
# ---------------------------------------------------------------------------

URB = ["UR8", "UR7", "UR6", "UR5", "UR4", "UR3"]
URF = ["UR1", "UR2"]
ULB = ["UL8", "UL7", "UL6", "UL5", "UL4", "UL3"]
ULF = ["UL1", "UL2"]
LRB = ["LR8", "LR7", "LR6", "LR5", "LR4", "LR3"]
LRF = ["LR1", "LR2"]
LLB = ["LL8", "LL7", "LL6", "LL5", "LL4", "LL3"]
LLF = ["LL1", "LL2"]


def up_check(list_landmark, tocheck):
    nb_correct = 0
    nb_midpoint = 0
    for landmark in list_landmark:
        if "Mid".upper() in landmark.upper():
            nb_midpoint += 1
        for check in tocheck:
            if check in landmark:
                nb_correct += landmark.count(check)
    return nb_correct == len(list_landmark) + nb_midpoint


def up_dental_distance(names, lr, ap, si):
    for group, neg, pos in [
        (URB, ("L", "D", "E"), ("B", "M", "I")),
        (URF, ("M", "L", "E"), ("D", "B", "I")),
        (ULB, ("B", "D", "E"), ("L", "M", "I")),
        (ULF, ("D", "L", "E"), ("M", "B", "I")),
        (LRB, ("L", "D", "I"), ("B", "M", "E")),
        (LRF, ("M", "L", "I"), ("D", "B", "E")),
        (LLB, ("B", "D", "I"), ("L", "M", "E")),
        (LLF, ("D", "L", "I"), ("M", "B", "E")),
    ]:
        if up_check(names, group):
            return (pos[0] if lr > 0 else neg[0],
                    pos[1] if ap > 0 else neg[1],
                    pos[2] if si > 0 else neg[2])
    return None


def up_dental_angle(names, lr, ap, si):
    """Upstream assigns ap, si then lr; the port returns yaw, pitch, roll."""
    for group, neg, pos in [
        (URB, ("D", "L", "DR"), ("M", "B", "MR")),
        (URF, ("L", "M", "DR"), ("B", "D", "MR")),
        (ULB, ("D", "B", "MR"), ("M", "L", "DR")),
        (ULF, ("L", "D", "MR"), ("B", "M", "DR")),
        (LRB, ("M", "B", "DR"), ("D", "L", "MR")),
        (LRF, ("B", "D", "DR"), ("L", "M", "MR")),
        (LLB, ("M", "L", "MR"), ("D", "B", "DR")),
        (LLF, ("B", "M", "MR"), ("L", "D", "DR")),
    ]:
        if up_check(names, group):
            return (pos[2] if lr > 0 else neg[2],
                    pos[0] if ap > 0 else neg[0],
                    pos[1] if si > 0 else neg[1])
    return None


class UpstreamLine:
    """Upstream's `Line`, as far as `__SignMeaningDist` touches it.

    It indexes the object with a slice and an integer; `Line.__getitem__`
    answers neither, so both come back None. That is the whole reason a
    point-to-line measurement never reads a side off its line.
    """

    def __getitem__(self, key):
        if key == "point 1" or key == 1:
            return "p1"
        if key == "point 2" or key == 2:
            return "p2"
        return None


def up_sign_meaning_dist(name1, point2line, lr, ap, si):
    lst_measurement = [name1, point2line]
    try:
        direction1 = lst_measurement[0][0:3]
        direction2 = lst_measurement[1][0:3]
    except Exception:
        direction1 = "No_direction"
        direction2 = "No_direction"

    for which in (0, 1):
        current = direction1 if which == 0 else direction2
        if current == "Mid":
            parts = lst_measurement[which].split("_")
            first = parts[1] if len(parts) > 1 else None
            second = parts[2] if len(parts) > 2 else None
            if first[0] == second[0]:
                resolved = first[0]
            elif (first[0] == "R" and second[0] == "L") or (first[0] == "L" and second[0] == "R"):
                resolved = "No_direction"
            elif first[0] in ("R", "L"):
                resolved = first[0]
            elif second[0] in ("R", "L"):
                resolved = second[0]
            else:
                resolved = None
            if which == 0:
                direction1 = resolved
            else:
                direction2 = resolved

    # Always true once the block above has run: it replaced "Mid" with a letter.
    if direction1 != "Mid" and direction2 != "Mid":
        try:
            direction1 = lst_measurement[0][0]
            direction2 = lst_measurement[1][0]
        except Exception:
            direction1 = "No_direction"
            direction2 = "No_direction"

    direction = None
    if direction1 == direction2:
        direction = direction1
    elif direction1 == "No_direction" and direction2 != "No_direction":
        direction = direction2
    elif direction2 == "No_direction" and direction1 != "No_direction":
        direction = direction1
    elif (direction1 == "R" and direction2 == "L") or (direction1 == "L" and direction2 == "R"):
        direction = "No_direction"

    if direction == "R":
        lateral = "Medial"
    elif direction == "L":
        lateral = "Lateral"
    elif direction == "No_direction":
        lateral = "x"
    else:
        lateral = "L"

    if lr > 0:
        if direction == "R":
            lateral = "Lateral"
        elif direction == "L":
            lateral = "Medial"
        elif direction == "No_direction":
            lateral = "x"
        else:
            lateral = "R"

    return lateral, ("A" if ap > 0 else "P"), ("S" if si > 0 else "I")


# ---------------------------------------------------------------------------
# Upstream: createlistprocess.reorganizeStat
# ---------------------------------------------------------------------------

TOOTHS = [
    "UR8", "UR7", "UR6", "UR5", "UR4", "UR3", "UR1", "UR2",
    "UL8", "UL7", "UL6", "UL5", "UL4", "UL3", "UL1", "UL2",
    "LR8", "LR7", "LR6", "LR5", "LR4", "LR3", "LR1", "LR2",
    "LL8", "LL7", "LL6", "LL5", "LL4", "LL3", "LL1", "LL2",
]


def _signed(value, meaning, negative):
    if value != "x" and value != "":
        value = float(value)
        if meaning == negative:
            value = -value
    return value


def up_reorganize(pc):
    keys = ["ID", "Landmarks", "Transverse", "AP", "Vertical", "3D",
            "Yaw", "Pitch", "Roll", "BL", "MD", "Rotation", "Arch", "Segment"]
    stats = {key: [] for key in keys}
    for i in range(len(pc["Patient"])):
        patient = str(pc["Patient"][i])
        numbered = re.fullmatch(r"(?:patient|pat|p)[ _-]?(\d+)", patient, re.IGNORECASE)
        stats["ID"].append(numbered.group(1) if numbered else patient)
        stats["Landmarks"].append(pc["Landmarks"][i])

        tooth = None
        for candidate in TOOTHS:
            if candidate in pc["Landmarks"][i]:
                tooth = candidate

        if tooth is not None:
            stats["Arch"].append(0 if "U" in tooth else 1)
            stats["Segment"].append(1 if ("1" in tooth or "2" in tooth) else 0)
            if stats["Segment"][-1] == 1:
                stats["AP"].append(str(_signed(pc["A-P Component"][i], pc["A-P Meaning"][i], "L")))
                stats["Transverse"].append(
                    str(_signed(pc["R-L Component"][i], pc["R-L Meaning"][i], "D")))
                stats["BL"].append(
                    str(_signed(pc["Pitch Component"][i], pc["Pitch Meaning"][i], "L")))
                stats["MD"].append(
                    str(_signed(pc["Roll Component"][i], pc["Roll Meaning"][i], "D")))
            else:
                stats["AP"].append(str(_signed(pc["A-P Component"][i], pc["A-P Meaning"][i], "D")))
                stats["Transverse"].append(
                    str(_signed(pc["R-L Component"][i], pc["R-L Meaning"][i], "B")))
                stats["MD"].append(
                    str(_signed(pc["Pitch Component"][i], pc["Pitch Meaning"][i], "D")))
                stats["BL"].append(
                    str(_signed(pc["Roll Component"][i], pc["Roll Meaning"][i], "L")))
            stats["Vertical"].append(
                str(_signed(pc["S-I Component"][i], pc["S-I Meaning"][i], "I")))
            stats["Rotation"].append(
                str(_signed(pc["Yaw Component"][i], pc["Yaw Meaning"][i], "DR")))
            stats["3D"].append(str(pc["3D Distance"][i]))
            stats["Yaw"].append("x")
            stats["Pitch"].append("x")
            stats["Roll"].append("x")
        else:
            stats["Arch"].append("x")
            stats["Segment"].append("x")
            stats["BL"].append("x")
            stats["MD"].append("x")
            stats["Rotation"].append("x")
            rl = pc["R-L Component"][i]
            if rl != "x" and rl != "":
                rl = float(rl)
                if pc["R-L Meaning"][i] in ("Medial", "L"):
                    rl = -rl
            stats["Transverse"].append(str(rl))
            stats["AP"].append(str(_signed(pc["A-P Component"][i], pc["A-P Meaning"][i], "P")))
            stats["Vertical"].append(
                str(_signed(pc["S-I Component"][i], pc["S-I Meaning"][i], "S")))
            stats["Yaw"].append(
                str(_signed(pc["Yaw Component"][i], pc["Yaw Meaning"][i], "CounterC")))
            stats["Pitch"].append(
                str(_signed(pc["Pitch Component"][i], pc["Pitch Meaning"][i], "CounterC")))
            stats["Roll"].append(
                str(_signed(pc["Roll Component"][i], pc["Roll Meaning"][i], "CounterC")))
            stats["3D"].append(str(pc["3D Distance"][i]))

    for key in [key for key, values in stats.items()
                if not values or all(value == "x" for value in values)]:
        del stats[key]
    return stats


# ---------------------------------------------------------------------------
# The comparisons
# ---------------------------------------------------------------------------

def test_the_two_distances_are_the_same_function():
    rng = np.random.default_rng(0)
    for _ in range(3000):
        first, second = rng.normal(scale=40.0, size=3), rng.normal(scale=40.0, size=3)
        assert up_compute_distance(first, second) == \
            measure.distance_between_points(first, second)


def test_the_two_point_to_line_distances_are_the_same_function():
    rng = np.random.default_rng(1)
    for _ in range(3000):
        point, start, end = (rng.normal(scale=40.0, size=3) for _ in range(3))
        assert up_compute_line_point(start, end, point) == \
            measure.distance_point_to_line(point, start, end)


def test_a_line_of_no_length_falls_back_the_same_way_in_both():
    rng = np.random.default_rng(2)
    for _ in range(200):
        point, start = rng.normal(scale=40.0, size=3), rng.normal(scale=40.0, size=3)
        assert up_compute_line_point(start, start.copy(), point) == \
            measure.distance_point_to_line(point, start, start.copy())


def test_the_two_angle_computations_are_the_same_function():
    rng = np.random.default_rng(3)
    compared = 0
    for _ in range(3000):
        points = [rng.normal(scale=40.0, size=3) for _ in range(4)]
        try:
            expected = up_compute_angles(*points)
        except ZeroDivisionError:
            continue
        assert expected == measure.angles_between_lines(*points)
        compared += 1
    assert compared > 2500, "the random input degenerated too often to prove anything"


def test_the_head_to_tail_supplement_is_taken_in_both():
    rng = np.random.default_rng(4)
    for _ in range(300):
        first, shared, last = (rng.normal(scale=40.0, size=3) for _ in range(3))
        assert up_compute_angles(first, shared, shared, last) == \
            measure.angles_between_lines(first, shared, shared, last)


SUFFIXES = ("O", "MB", "DB", "R")
TOOTH_NAMES = [tooth + suffix
               for tooth in URB + URF + ULB + ULF + LRB + LRF + LLB + LLF
               for suffix in SUFFIXES]
SIGNS = ((-1.0, -1.0, -1.0), (1.0, -1.0, 1.0), (-1.0, 1.0, 1.0), (1.0, 1.0, 1.0))


def test_the_dental_distance_labels_are_the_same_table():
    for first, second in itertools.product(TOOTH_NAMES, repeat=2):
        for lr, ap, si in SIGNS:
            assert up_dental_distance([first, second], lr, ap, si) == \
                measure.dental_distance_meanings(first, second, lr, ap, si)


def test_the_dental_angle_labels_are_the_same_table():
    for group in (URB[:2], URF, ULB[:2], ULF, LRB[:2], LRF, LLB[:2], LLF):
        pool = [tooth + suffix for tooth in group for suffix in SUFFIXES]
        for combo in itertools.islice(itertools.product(pool, repeat=4), 300):
            for lr, ap, si in SIGNS:
                assert up_dental_angle(list(combo), lr, ap, si) == \
                    measure.dental_angle_meanings(combo, lr, ap, si)


def test_a_measurement_spanning_two_groups_is_unlabelled_in_both():
    assert up_dental_distance(["UR6O", "UL6O"], 1, 1, 1) is None
    assert measure.dental_distance_meanings("UR6O", "UL6O", 1, 1, 1) is None


SKELETAL_NAMES = ["Ba", "S", "N", "RPo", "LPo", "ROr", "LOr", "ANS", "PNS", "Me",
                  "Gn", "RCo", "LCo", "RGo", "LGo", "A", "B", "Pog",
                  "Mid_ROr_LOr", "Mid_RPo_LPo", "Mid_RCo_RGo", "Mid_LCo_LGo"]


def test_the_skeletal_distance_labels_are_the_same_logic():
    for first, second in itertools.product(SKELETAL_NAMES, repeat=2):
        for lr, ap, si in itertools.product((-1.0, 1.0), repeat=3):
            assert up_sign_meaning_dist(first, second, lr, ap, si) == \
                measure.skeletal_distance_meanings(first, second, lr, ap, si)


def test_a_point_to_line_distance_reads_the_line_the_same_way_in_both():
    """Upstream's second operand is the Line OBJECT, which answers None to both
    the slice and the index it is given. The port passes None for it."""
    line = UpstreamLine()
    for name in SKELETAL_NAMES:
        for lr in (-1.0, 1.0):
            assert up_sign_meaning_dist(name, line, lr, 1.0, 1.0) == \
                measure.skeletal_distance_meanings(name, None, lr, 1.0, 1.0)


LABELS = ["ROr - LOr", "UR6O - UL6O", "LL1O - LR1O", "Ba - S", "Mid_ROr_LOr - Me",
          "RCo-LCo / ROr-LOr", "UR6O-UR3O / UL6O-UL3O"]
MEANINGS = ["Medial", "Lateral", "R", "L", "x", "A", "P", "S", "I", "D", "B", "M",
            "E", "DR", "MR", "CounterC", "ClockWise", "Clockwise"]


def test_the_two_stat_reorganisations_are_the_same_function():
    rng = random.Random(0)
    for _ in range(300):
        rows = []
        for _ in range(rng.randint(1, 12)):
            row = {
                "Patient": rng.choice(["P1", "P_0001", "Dupont", "Patient 12"]),
                "Type of measurement": "Distance between 2 points T1 T2",
                "Landmarks": rng.choice(LABELS),
                "3D Distance": round(rng.uniform(0, 50), 3),
            }
            for axis in ("R-L", "A-P", "S-I"):
                row[f"{axis} Component"] = rng.choice(["x", round(rng.uniform(0, 30), 3)])
                row[f"{axis} Meaning"] = rng.choice(MEANINGS)
            for axis in ("Yaw", "Pitch", "Roll"):
                row[f"{axis} Component"] = rng.choice(["x", round(rng.uniform(0, 90), 3)])
                row[f"{axis} Meaning"] = rng.choice(MEANINGS)
            rows.append(row)

        columns = {key: [row[key] for row in rows] for key in rows[0]}
        assert up_reorganize(columns) == features.to_stats(rows)
