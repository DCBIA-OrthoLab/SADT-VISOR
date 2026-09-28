"""The registration arithmetic: a closed-form landmark fit, then an ICP.

Neither needs a card, a checkpoint or another tool, so both run for real here.
Two stages, and the order is load-bearing: the landmark transform puts the two
meshes in roughly the same place so the ICP starts inside its capture range.

There are two ICPs here, and the difference matters. `icp_point_to_plane` is
what the chain refines with: it measures along the target's normals, which is
what lets a surface slide along a smooth occlusal plane instead of stalling on
it. `icp_point_to_point` is the older estimator, kept because the point-to-plane
step falls back to it when its linearised system is degenerate -- and kept with
the name corrected, since upstream called it point-to-plane and computed
point-to-point.
"""

import numpy as np
import pytest

from sadt_areg_ioscbct import geometry


def rigid_matrix(angle=0.21, translation=(4.0, -3.0, 2.5)):
    rotation = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix


CROWNS = np.array([
    [0.0, 0.0, 0.0], [10.0, 0.0, 1.0], [20.0, 4.0, 0.0],
    [0.0, 10.0, 2.0], [10.0, 10.0, 0.0], [20.0, 14.0, 3.0],
])


# ---------------------------------------------------------------------------
# align_by_landmarks
# ---------------------------------------------------------------------------

def test_a_known_rigid_motion_is_recovered_to_the_float():
    """Closed form, not a search: same landmarks in, same matrix out, every
    time. The whole two-stage design rests on that."""
    truth = rigid_matrix()
    matrix = geometry.align_by_landmarks(CROWNS, geometry.apply(CROWNS, truth))
    assert matrix == pytest.approx(truth, abs=1e-6)


def test_the_fit_is_rigid_even_when_the_points_do_not_agree():
    """RigidBody mode, so no scaling and no reflection can sneak in through a
    noisy landmark -- a scaled intraoral scan is a wrong measurement that looks
    like a good registration."""
    rng = np.random.default_rng(3)
    noisy = geometry.apply(CROWNS, rigid_matrix()) + rng.normal(scale=0.8, size=CROWNS.shape)

    rotation = geometry.align_by_landmarks(CROWNS, noisy)[:3, :3]
    assert rotation @ rotation.T == pytest.approx(np.eye(3), abs=1e-9)
    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-9)


def test_the_same_landmarks_always_give_the_same_matrix():
    """Two runs of one request on one dataset must agree: this is patient data
    being resampled."""
    target = geometry.apply(CROWNS, rigid_matrix())
    first = geometry.align_by_landmarks(CROWNS, target)
    second = geometry.align_by_landmarks(CROWNS, target)
    assert np.array_equal(first, second)


def test_mismatched_point_counts_are_refused_with_both_counts():
    with pytest.raises(ValueError) as raised:
        geometry.align_by_landmarks(CROWNS, CROWNS[:4])
    assert "6 moving against 4 fixed" in str(raised.value)


def test_fewer_than_three_points_is_refused_with_the_count():
    """Two points fix a line, not a frame; the fit would return something and
    it would be wrong about the rotation around that line."""
    with pytest.raises(ValueError) as raised:
        geometry.align_by_landmarks(CROWNS[:2], CROWNS[:2])
    assert "at least 3 shared points, got 2" in str(raised.value)


def test_exactly_three_points_is_accepted():
    truth = rigid_matrix()
    matrix = geometry.align_by_landmarks(CROWNS[:3], geometry.apply(CROWNS[:3], truth))
    assert matrix == pytest.approx(truth, abs=1e-6)


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------

def test_apply_rotates_then_translates():
    matrix = rigid_matrix(angle=0.0, translation=(1.0, 2.0, 3.0))
    assert geometry.apply(np.array([[0.0, 0.0, 0.0]]), matrix)[0] == pytest.approx([1, 2, 3])


def test_apply_leaves_its_input_alone():
    """It returns a new array: the caller writes the ORIGINAL mesh's points
    back out under a different transform in the same loop."""
    points = CROWNS.copy()
    geometry.apply(points, rigid_matrix())
    assert np.array_equal(points, CROWNS)


def test_apply_accepts_a_list_of_tuples():
    """vtk hands back tuples, and `@` on a list of tuples is a TypeError."""
    assert geometry.apply([(1.0, 0.0, 0.0)], np.eye(4))[0] == pytest.approx([1, 0, 0])


# ---------------------------------------------------------------------------
# icp_point_to_point
# ---------------------------------------------------------------------------

def _cloud(count=400, seed=1):
    rng = np.random.default_rng(seed)
    return rng.normal(scale=6.0, size=(count, 3))


def test_the_icp_recovers_a_small_displacement():
    fixed = _cloud()
    displacement = rigid_matrix(angle=0.02, translation=(0.4, -0.3, 0.2))
    moving = geometry.apply(fixed, np.linalg.inv(displacement))

    matrix, stats = geometry.icp_point_to_point(moving, fixed, max_dist=5.0)
    assert geometry.apply(moving, matrix) == pytest.approx(fixed, abs=1e-3)
    assert stats["rmse"] < 1e-3
    assert stats["fitness"] == pytest.approx(1.0)
    assert stats["iterations"] >= 1


def test_the_icp_reports_what_it_did_rather_than_only_that_it_returned():
    """A caller has to be able to say whether the registration converged. The
    three numbers travel into the run report."""
    fixed = _cloud()
    _matrix, stats = geometry.icp_point_to_point(fixed, fixed, max_dist=1.5)
    assert set(stats) == {"rmse", "fitness", "iterations"}


def test_a_max_dist_too_small_to_match_anything_stops_rather_than_wandering():
    """Below the point spacing nothing is a correspondence, and continuing on
    fewer than three would be fitting a frame to a line."""
    fixed = _cloud()
    moving = geometry.apply(fixed, rigid_matrix(angle=0.5, translation=(50.0, 0.0, 0.0)))

    matrix, stats = geometry.icp_point_to_point(moving, fixed, max_dist=1e-6)
    assert np.array_equal(matrix, np.eye(4))
    assert stats["fitness"] == 0.0


def test_the_icp_never_returns_a_reflection():
    """Flipping the smallest singular vector is the standard repair. Without
    it a mesh comes back MIRRORED with a perfectly good RMSE -- a left canine
    where the right one should be.

    Built so the correspondence really is the mirror pairing: a near-flat
    sheet, mirrored through its own plane, so each moved point's nearest
    neighbour is its own reflection and the covariance is genuinely
    orientation-reversing. On a round cloud the nearest neighbours are a
    scramble and the SVD never sees the reflection at all.
    """
    rng = np.random.default_rng(5)
    grid = np.array([[float(x), float(y), 0.0] for x in range(18) for y in range(18)])
    grid[:, 2] = rng.uniform(-0.04, 0.04, size=len(grid))
    moving = grid * np.array([1.0, 1.0, -1.0])

    matrix, _stats = geometry.icp_point_to_point(moving, grid, max_dist=0.5)
    assert np.linalg.det(matrix[:3, :3]) == pytest.approx(1.0, abs=1e-6)


def test_the_icp_needs_three_points_on_each_side():
    cloud = _cloud(count=10)
    with pytest.raises(ValueError, match="at least 3 points"):
        geometry.icp_point_to_point(cloud[:2], cloud)
    with pytest.raises(ValueError, match="at least 3 points"):
        geometry.icp_point_to_point(cloud, cloud[:2])


def test_the_upstream_loop_bounds_are_kept():
    """The thresholds decide when the registration stops moving, so changing
    one changes results. Pinned rather than tuned."""
    assert geometry.DEFAULT_MAX_DIST == 1.5
    assert geometry._MAX_ITERATIONS == 2000
    assert geometry._RMSE_THRESHOLD == 1e-8
    assert geometry._FITNESS_THRESHOLD == 1e-8


def test_the_icp_stops_when_it_stops_improving_rather_than_running_out():
    """2000 iterations on a real arch is minutes. Two identical clouds
    converge on the second pass."""
    fixed = _cloud()
    _matrix, stats = geometry.icp_point_to_point(fixed, fixed, max_dist=1.5)
    assert stats["iterations"] < 5


# ---------------------------------------------------------------------------
# The landmark fit, and the pairs it throws out
# ---------------------------------------------------------------------------

LABELS = ["UR1O", "UR3O", "UR6O", "UL1O", "UL3O", "UL6O"]


def test_the_pair_that_does_not_belong_is_left_out_of_the_fit():
    """One landmark on the wrong tooth used to move the whole arch.

    Six pairs are perfectly consistent with themselves however badly one of
    them was predicted: the fit spreads that one error over all six and comes
    back with a residual that reads as ordinary noise. Refitting without each
    pair in turn is what tells them apart.
    """
    fixed = geometry.apply(CROWNS, rigid_matrix())
    fixed[4] += (0.0, 0.0, 9.0)

    _matrix, keep = geometry.fit_rigid_without_outliers(CROWNS, fixed, LABELS)
    assert 4 not in keep


def test_a_fit_that_is_merely_noisy_keeps_every_pair():
    """The other half: trimming has to cost something, or it eats the data."""
    rng = np.random.default_rng(7)
    fixed = geometry.apply(CROWNS, rigid_matrix()) + rng.normal(scale=0.3, size=CROWNS.shape)

    _matrix, keep = geometry.fit_rigid_without_outliers(CROWNS, fixed, LABELS)
    assert keep == list(range(len(CROWNS)))


def test_trimming_stops_before_the_residual_stops_meaning_anything():
    """A fit on three pairs passes through all three, so its residual is zero
    whatever they are. Trimming that far would keep dropping pairs on the
    strength of a number that no longer measures the fit."""
    fixed = geometry.apply(CROWNS, rigid_matrix())
    for index in (1, 2, 4, 5):
        fixed[index] += (0.0, 0.0, 9.0)

    _matrix, keep = geometry.fit_rigid_without_outliers(CROWNS, fixed, LABELS)
    assert len(keep) >= geometry.MIN_LANDMARK_PAIRS_AFTER_TRIM


def test_a_landmark_the_scan_says_is_off_the_tooth_goes_whatever_the_fit_says():
    """That verdict comes from the image, so it holds however the landmarks
    look to each other -- and it catches the case the fit cannot see, a point
    the network put on the opposing arch, which is self-consistent."""
    fixed = geometry.apply(CROWNS, rigid_matrix())

    _matrix, keep = geometry.fit_rigid_without_outliers(
        CROWNS, fixed, LABELS, on_enamel={"UL3O": False})
    assert LABELS.index("UL3O") not in keep


def test_landmarks_that_cannot_place_the_arch_leave_the_pose_alone():
    """Past the residual threshold the two sets do not describe the same
    points, and a fit to them would place the arch worse than not placing it.
    The identity leaves the ICP to start from the pose the orientation left."""
    rng = np.random.default_rng(3)
    fixed = rng.normal(scale=40.0, size=CROWNS.shape)

    matrix, report = geometry.prealign(CROWNS, fixed, LABELS)
    assert matrix == pytest.approx(np.eye(4))
    assert report["accepted"] is False


def test_too_few_pairs_leaves_the_pose_alone_and_says_which_it_needed():
    matrix, report = geometry.prealign(CROWNS[:2], CROWNS[:2], LABELS[:2])
    assert matrix == pytest.approx(np.eye(4))
    assert report["reason"] == "too few pairs"


def test_landmarks_in_a_line_are_kept_but_measured():
    """Being under-determined is not being wrong: the identity would start the
    ICP further off. What the run must not do is stay quiet about it."""
    collinear = np.array([[float(x), 0.0, 0.0] for x in range(6)])
    _matrix, report = geometry.prealign(
        collinear, geometry.apply(collinear, rigid_matrix()), LABELS)
    assert report["accepted"] is True
    assert report["spread"] < geometry.MIN_LANDMARK_SPREAD_RATIO


# ---------------------------------------------------------------------------
# The ICP target
# ---------------------------------------------------------------------------

def test_the_target_keeps_only_what_the_arch_could_match_to():
    """Cropping is what stops the cranial base and the vertebrae being
    candidates. Anchored on the CBCT landmarks, not on the moving scan: a scan
    whose pre-alignment was skipped is still in the frame the orientation left
    it in and would drag the box across the whole head."""
    rng = np.random.default_rng(11)
    near = rng.uniform(-5.0, 5.0, size=(50, 3))
    far = rng.uniform(200.0, 220.0, size=(50, 3))
    target = geometry.Target(np.vstack([near, far]), None)

    cropped = target.around(np.zeros((1, 3)), margin=25.0)
    assert len(cropped) == len(near)


def test_a_target_with_nothing_near_the_landmarks_is_kept_whole():
    """Better a wide target than none: an empty one would leave the ICP with no
    correspondences and the arch unregistered."""
    target = geometry.Target(np.full((10, 3), 500.0), None)
    assert len(target.around(np.zeros((1, 3)), margin=25.0)) == 10


# ---------------------------------------------------------------------------
# icp_point_to_plane
# ---------------------------------------------------------------------------

def plane_mesh(size=20.0, resolution=40, centre=(0.0, 0.0, 0.0)):
    import pyvista as pv

    return pv.Plane(center=centre, direction=(0.0, 0.0, 1.0), i_size=size,
                    j_size=size, i_resolution=resolution,
                    j_resolution=resolution).triangulate()


def test_the_point_to_plane_icp_closes_a_gap_along_the_normal():
    """The displacement it is built to remove: two parallel surfaces a little
    apart. Point-to-point pulls towards particular neighbours and stalls on a
    smooth plane; measuring along the normal does not."""
    fixed = plane_mesh()
    moving = plane_mesh(centre=(0.3, -0.2, 0.6))

    matrix, stats = geometry.icp_point_to_plane(moving, fixed, max_dist=2.0)
    assert stats["fitness"] > 0.9
    assert geometry.apply(moving.points, matrix)[:, 2] == pytest.approx(0.0, abs=1e-2)


def test_the_icp_reports_the_reading_of_the_normals_it_kept():
    """Which way a mesh winds its faces is a property of the file, and the two
    modalities do not have to agree. Both readings are tried, and the one that
    fits wins -- silently picking the wrong one locks the registration onto the
    opposing arch."""
    fixed = plane_mesh()
    moving = plane_mesh(centre=(0.0, 0.0, 0.5))

    _matrix, stats = geometry.icp_point_to_plane(moving, fixed, max_dist=2.0)
    assert stats["used_normals"] is True
    assert stats["sign"] in (1.0, -1.0)


def test_a_target_too_small_to_refine_on_is_refused_rather_than_ignored():
    fixed = geometry.Target(np.zeros((2, 3)), None)
    with pytest.raises(ValueError, match="3 target points"):
        geometry.icp_point_to_plane(plane_mesh(), fixed)


# ---------------------------------------------------------------------------
# crown_points
# ---------------------------------------------------------------------------

def test_the_gingiva_is_left_out_of_the_fit_when_the_labels_say_where_it_is():
    """An intraoral scan is about 60% gingiva, and gingiva has no counterpart
    in a CBCT surface. Those points are not matched so much as dragged, and
    they outnumber the crowns."""
    mesh = plane_mesh()
    labels = np.full(mesh.n_points, 11)
    labels[: mesh.n_points // 2] = geometry.GINGIVA_LABELS[0]
    mesh.point_data["Universal_ID"] = labels

    crowns = geometry.crown_points(mesh)
    assert crowns is not None and crowns.sum() == mesh.n_points - mesh.n_points // 2


def test_labels_too_sparse_to_trust_register_the_whole_arch():
    mesh = plane_mesh()
    labels = np.full(mesh.n_points, geometry.GINGIVA_LABELS[0])
    labels[:2] = 11
    mesh.point_data["Universal_ID"] = labels

    assert geometry.crown_points(mesh) is None


def test_a_mesh_with_no_labels_registers_whole_rather_than_failing():
    assert geometry.crown_points(plane_mesh()) is None
