"""The measurement arithmetic, and what a sign means.

The arithmetic is checked against upstream's own on random input in
`test_measure_matches_upstream.py`; what is pinned here is the behaviour a
reader needs to know: which way the axes run, what happens at the degenerate
cases, and which table a measurement is read with.
"""

import math

import numpy as np
import pytest

from sadt_vface import measure


# ---------------------------------------------------------------------------
# Distances
# ---------------------------------------------------------------------------

def test_the_first_two_axes_are_negated_and_the_third_is_not():
    """A Slicer volume is in LPS, so +x runs to the patient's LEFT and +y
    posteriorly, while the columns this feeds are read right-positive and
    anterior-positive. The superior axis agrees in both."""
    lr, ap, si, _norm = measure.distance_between_points([0, 0, 0], [1, 2, 3])
    assert (lr, ap, si) == (-1.0, -2.0, 3.0)


def test_the_3d_distance_is_the_length_of_the_offset():
    _lr, _ap, _si, norm = measure.distance_between_points([0, 0, 0], [3, 4, 12])
    assert norm == pytest.approx(13.0)


def test_a_distance_is_reported_to_the_micron():
    """Three decimals, as upstream rounds. It is not precision, it is a stable
    key: the post-processing looks a row up by its landmarks and reads the
    number back, and an unrounded float printed twice is two strings."""
    _lr, _ap, _si, norm = measure.distance_between_points([0, 0, 0], [1, 1, 1])
    assert norm == round(math.sqrt(3), 3)


def test_the_distance_to_a_line_is_perpendicular_to_it():
    """What is left of the offset once the part running ALONG the line is taken
    out. A point directly above the middle of a line is its own height away."""
    lr, ap, si, norm = measure.distance_point_to_line(
        [0.0, 0.0, 5.0], [-10.0, 0.0, 0.0], [10.0, 0.0, 0.0]
    )
    assert norm == pytest.approx(5.0)
    assert si == pytest.approx(5.0)
    assert (lr, ap) == pytest.approx((0.0, 0.0))


def test_a_point_on_the_line_is_no_distance_from_it():
    _lr, _ap, _si, norm = measure.distance_point_to_line(
        [3.0, 0.0, 0.0], [-10.0, 0.0, 0.0], [10.0, 0.0, 0.0]
    )
    assert norm == pytest.approx(0.0)


def test_a_line_whose_two_points_coincide_falls_back_to_a_plain_distance():
    """It describes no line. Answering with the point-to-point distance is more
    use than a NaN, and a measurement list naming one landmark twice is a
    mistake in the list rather than in the data."""
    assert measure.distance_point_to_line([0, 0, 5], [0, 0, 0], [0, 0, 0]) == \
        measure.distance_between_points([0, 0, 0], [0, 0, 5])


# ---------------------------------------------------------------------------
# Angles
# ---------------------------------------------------------------------------

def test_two_parallel_lines_measure_nothing():
    yaw, pitch, roll = measure.angles_between_lines(
        [0, 0, 0], [1, 2, 3], [5, 5, 5], [7, 9, 11]
    )
    assert (abs(yaw), abs(pitch), abs(roll)) == pytest.approx((0.0, 0.0, 0.0))


def test_a_known_turn_in_one_plane_is_read_as_that_angle():
    """Both lines are tilted out of every axis -- see
    `test_a_line_running_along_an_axis_is_refused` for why -- and the second is
    the first turned 30 degrees about the superior axis. Yaw reads the turn;
    the other two planes see the projection of it, not zero."""
    turn = math.radians(30.0)
    first = np.array([1.0, 0.0, 0.4])
    rotated = np.array([
        first[0] * math.cos(turn) - first[1] * math.sin(turn),
        first[0] * math.sin(turn) + first[1] * math.cos(turn),
        first[2],
    ])
    yaw, _pitch, _roll = measure.angles_between_lines(
        [0.0, 0.0, 0.0], first, [0.0, 0.0, 0.0], rotated
    )
    assert abs(yaw) == pytest.approx(30.0, abs=1e-6)


def test_two_lines_meeting_head_to_tail_measure_the_supplement():
    """`A->B` and `B->C` continuing straight on is a 180 degree angle at B, not
    a zero one: the second line leaves where the first arrived."""
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([1.0, 2.0, 3.0])
    c = np.array([2.0, 4.0, 6.0])
    yaw, _pitch, _roll = measure.angles_between_lines(a, b, b, c)
    assert abs(yaw) == pytest.approx(180.0)


def test_reversing_one_line_gives_the_supplement_with_the_other_sign():
    """Not the negation, which is the easy thing to assume. Turning a line
    round changes which way it sweeps AND swaps the angle for its supplement,
    so a 62 degree turn read one way is -118 read the other.

    It matters because a measurement list names the two ends of each line, and
    naming them in the other order is not a cosmetic choice.
    """
    forward = measure.angles_between_lines(
        [0, 0, 0], [1.0, 0.2, 0.4], [0, 0, 0], [0.3, 1.0, 0.5]
    )
    backward = measure.angles_between_lines(
        [0, 0, 0], [1.0, 0.2, 0.4], [0.3, 1.0, 0.5], [0, 0, 0]
    )
    assert forward[0] > 0 and backward[0] < 0
    assert abs(forward[0]) + abs(backward[0]) == pytest.approx(180.0, abs=1e-3)


def test_a_line_running_along_an_axis_is_refused():
    """The angle about an axis is read in the plane perpendicular to it, and a
    line parallel to that axis has no length there -- so there is no direction
    to measure from. All three planes are computed, so ANY axis-parallel line
    refuses the whole measurement. Upstream divides by zero here; this names it.
    """
    with pytest.raises(measure.MeasurementError, match="runs along the axis"):
        measure.angles_between_lines([0, 0, 0], [0, 0, 1], [0, 0, 0], [0, 1, 0])


# ---------------------------------------------------------------------------
# Which table a measurement is read with
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["UR1O", "UR6MB", "LL3R", "Mid_UR6O_UL6O"])
def test_a_name_holding_a_tooth_is_dental(name):
    assert measure.is_dental(name) is True


@pytest.mark.parametrize("name", ["Ba", "S", "N", "RPo", "ANS", "Me", "Mid_ROr_LOr"])
def test_a_skeletal_name_is_not(name):
    assert measure.is_dental(name) is False


def test_the_dental_table_covers_every_tooth_once():
    """Eight groups, and a tooth in two of them would be read with whichever
    the loop reached first."""
    seen = [tooth for group in measure._DENTAL_DISTANCE for tooth in group]
    assert len(seen) == len(set(seen))
    assert set(seen) == {
        f"{jaw}{side}{n}" for jaw in "UL" for side in "LR" for n in range(1, 9)
    } - {f"{jaw}{side}{n}" for jaw in "UL" for side in "LR" for n in ()}


def test_the_two_dental_tables_name_the_same_groups():
    """Distances and angles are read with different labels but the same
    grouping; a group in one and not the other would be silently unlabelled."""
    assert set(measure._DENTAL_DISTANCE) == set(measure._DENTAL_ANGLE)


# ---------------------------------------------------------------------------
# Skeletal sign meanings
# ---------------------------------------------------------------------------

def test_two_landmarks_on_one_side_read_medial_and_lateral():
    """Which is the useful reading: on the right, a positive left-right
    component points away from the midline."""
    assert measure.skeletal_distance_meanings("RPo", "ROr", 1.0, 1.0, 1.0)[0] == "Lateral"
    assert measure.skeletal_distance_meanings("RPo", "ROr", -1.0, 1.0, 1.0)[0] == "Medial"
    # And on the left it is the other way round.
    assert measure.skeletal_distance_meanings("LPo", "LOr", 1.0, 1.0, 1.0)[0] == "Medial"


def test_two_landmarks_on_opposite_sides_have_no_lateral_direction():
    """A measurement spanning the midline is not medial to anything."""
    assert measure.skeletal_distance_meanings("RPo", "LPo", 1.0, 1.0, 1.0)[0] == "x"


def test_landmarks_with_no_side_fall_back_to_plain_right_and_left():
    assert measure.skeletal_distance_meanings("Ba", "S", 1.0, 1.0, 1.0)[0] == "R"
    assert measure.skeletal_distance_meanings("Ba", "S", -1.0, 1.0, 1.0)[0] == "L"


def test_the_other_two_axes_are_read_straight_off_the_sign():
    _lr, ap, si = measure.skeletal_distance_meanings("Ba", "S", 0.0, 1.0, 1.0)
    assert (ap, si) == ("A", "S")
    _lr, ap, si = measure.skeletal_distance_meanings("Ba", "S", 0.0, -1.0, -1.0)
    assert (ap, si) == ("P", "I")


def test_a_midpoint_reads_its_first_letter_and_not_the_side_it_spans():
    """Upstream resolves a midpoint's side from the two landmarks it joins and
    then OVERWRITES its own answer -- the guard below that block is always
    true, and it re-reads `name[0]`. So `Mid_ROr_LOr` reads `M`.

    Reproduced rather than repaired: the label decides the sign of a feature the
    models were trained on, so changing it would change classifications. See
    `measure._side_of`.
    """
    assert measure._side_of("Mid_ROr_LOr") == "M"
    assert measure.skeletal_distance_meanings("Mid_ROr_LOr", "Ba", 1.0, 1.0, 1.0)[0] == "R"


def test_a_point_to_line_distance_reads_no_side_off_the_line():
    """Upstream's second operand there is the Line object, and indexing it
    gives None -- so however the line is named, it contributes no side."""
    assert measure._side_of(None) is None
    assert measure.skeletal_distance_meanings("RPo", None, 1.0, 1.0, 1.0)[0] == "R"


# ---------------------------------------------------------------------------
# Dental sign meanings
# ---------------------------------------------------------------------------

def test_a_dental_distance_inside_one_group_is_labelled():
    labels = measure.dental_distance_meanings("UR6O", "UR3O", 1.0, 1.0, 1.0)
    assert labels == ("B", "M", "I")


def test_a_dental_distance_across_two_groups_is_not():
    """Buccal and lingual are not the same direction on the two sides of the
    arch, so a measurement spanning them has no single label."""
    assert measure.dental_distance_meanings("UR6O", "UL6O", 1.0, 1.0, 1.0) is None


def test_the_upper_and_lower_halves_differ_only_in_the_vertical_column():
    """Erupting and intruding swap between the arches; buccal, lingual, mesial
    and distal do not."""
    for upper, lower in (
        (measure.UPPER_RIGHT_BACK, measure.LOWER_RIGHT_BACK),
        (measure.UPPER_LEFT_FRONT, measure.LOWER_LEFT_FRONT),
    ):
        up, low = measure._DENTAL_DISTANCE[upper], measure._DENTAL_DISTANCE[lower]
        assert up[0] == low[0] and up[1] == low[1]
        assert up[2] == tuple(reversed(low[2]))


def test_an_angle_between_two_timepoints_is_labelled_by_which_way_it_turned():
    """Both lines are the same anatomy seen at two times, so there is no
    anatomical direction to name."""
    assert measure.cross_timepoint_angle_meanings(1.0, 1.0, 1.0) == (
        "CounterC", "CounterC", "Clockwise"
    )
    assert measure.cross_timepoint_angle_meanings(-1.0, -1.0, -1.0) == (
        "ClockWise", "ClockWise", "CounterC"
    )
