"""The registration itself: landmarks first, then a point-to-plane ICP.

Ported from upstream's `AREG_IOSCBCT/AREG_IOSCBCT.py`, which is a Slicer CLI
module. Nothing here predicts anything -- the landmarks, the tooth labels and
the orientation all arrive as inputs, produced by other tools. That is what
lets this tool depend on neither torch nor pytorch3d while driving engines
pinned to both.

Two stages, and the order is load-bearing: the landmark transform puts the two
meshes in roughly the same place so the ICP starts inside its capture range. An
ICP started on unaligned meshes converges to whatever local minimum it happens
to reach.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# How far a point may be from its nearest neighbour and still count as a
# correspondence, in millimetres. This is the point-to-POINT estimator's own
# default and belongs to it alone; the chain refines with `icp_point_to_plane`
# and its `ICP_MAX_DIST_MM`, which is what upstream's call sites pass.
DEFAULT_MAX_DIST = 1.5

# Upstream's loop bounds, kept as they are: the thresholds decide when the
# registration stops moving, and changing them changes results.
_MAX_ITERATIONS = 2000
_RMSE_THRESHOLD = 1e-8
_FITNESS_THRESHOLD = 1e-8


def align_by_landmarks(moving_lms, fixed_lms) -> np.ndarray:
    """The 4x4 that best maps `moving_lms` onto `fixed_lms`, rigid.

    `vtkLandmarkTransform` in RigidBody mode, which is a closed-form fit rather
    than a search: same landmarks in, same matrix out, every time.

    It takes the landmarks and nothing else. It used to take the mesh's points
    as its first argument and never read them, which reads as though the fit
    were influenced by the surface -- it is not, and that is the property the
    two-stage design depends on.
    """
    import vtk

    if len(moving_lms) != len(fixed_lms):
        raise ValueError(
            "Landmark alignment needs the same points on both sides: "
            f"{len(moving_lms)} moving against {len(fixed_lms)} fixed."
        )
    if len(moving_lms) < 3:
        raise ValueError(
            f"Landmark alignment needs at least 3 shared points, got {len(moving_lms)}."
        )

    source, target = vtk.vtkPoints(), vtk.vtkPoints()
    for point in moving_lms:
        source.InsertNextPoint(*point)
    for point in fixed_lms:
        target.InsertNextPoint(*point)

    transform = vtk.vtkLandmarkTransform()
    transform.SetSourceLandmarks(source)
    transform.SetTargetLandmarks(target)
    transform.SetModeToRigidBody()
    transform.Update()

    matrix = np.eye(4)
    vtk_matrix = transform.GetMatrix()
    for row in range(4):
        for column in range(4):
            matrix[row, column] = vtk_matrix.GetElement(row, column)
    return matrix


def icp_point_to_point(moving_points: np.ndarray, fixed_points: np.ndarray,
                       max_dist: float = DEFAULT_MAX_DIST) -> tuple:
    """Refine an alignment; return `(4x4 matrix, {rmse, fitness, iterations})`.

    **Renamed from upstream's `run_icp_point_to_plane`, which is not what it
    computes.** The update below is the point-to-point SVD -- centre both point
    sets, take the SVD of their covariance, repair a reflection. A true
    point-to-plane step minimises distance along the fixed surface's NORMALS and
    solves a linearised 6x6 system; upstream computes the fixed mesh's normals
    and then never uses them.

    The arithmetic is kept exactly as upstream wrote it: this is a repackaging,
    and swapping the estimator would change every result the tool has produced.
    Only the name is corrected, which changes nothing and stops the next reader
    trusting a label that contradicts the code under it.

    Returns the metrics as well as the matrix, so a caller can report whether
    the registration actually converged rather than only that it returned.
    """
    from scipy.spatial import cKDTree

    if len(moving_points) < 3 or len(fixed_points) < 3:
        raise ValueError("ICP needs at least 3 points on each mesh.")

    transformation = np.eye(4)
    current = np.asarray(moving_points, dtype=float).copy()
    tree = cKDTree(np.asarray(fixed_points, dtype=float))

    previous_rmse, previous_fitness = np.inf, 0.0
    rmse, fitness, iteration = np.inf, 0.0, 0

    for iteration in range(_MAX_ITERATIONS):
        distances, indices = tree.query(current, k=1)
        valid = distances < max_dist
        if valid.sum() < 3:
            logger.warning("ICP stopped at iteration %d: fewer than 3 correspondences", iteration)
            break

        rmse = float(np.sqrt(np.mean(distances[valid] ** 2)))
        fitness = float(valid.sum() / len(current))
        if (abs(previous_rmse - rmse) < _RMSE_THRESHOLD
                and abs(previous_fitness - fitness) < _FITNESS_THRESHOLD):
            break
        previous_rmse, previous_fitness = rmse, fitness

        source = current[valid]
        target = np.asarray(fixed_points, dtype=float)[indices[valid]]
        source_centre, target_centre = source.mean(axis=0), target.mean(axis=0)
        covariance = (source - source_centre).T @ (target - target_centre)
        u, _s, vt = np.linalg.svd(covariance)
        rotation = vt.T @ u.T
        if np.linalg.det(rotation) < 0:
            # A reflection is not a rigid motion: flipping the smallest singular
            # vector is the standard repair, and without it a mesh can come back
            # mirrored with a perfectly good RMSE.
            vt[-1, :] *= -1
            rotation = vt.T @ u.T
        translation = target_centre - rotation @ source_centre

        step = np.eye(4)
        step[:3, :3] = rotation
        step[:3, 3] = translation
        transformation = step @ transformation
        current = (rotation @ current.T).T + translation

    return transformation, {
        "rmse": rmse,
        "fitness": fitness,
        "iterations": iteration + 1,
    }


def apply(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """`points` through a 4x4, as a new array."""
    points = np.asarray(points, dtype=float)
    return (matrix[:3, :3] @ points.T).T + matrix[:3, 3]


# ---------------------------------------------------------------------------
# The landmark fit, with the pairs it cannot account for taken out
# ---------------------------------------------------------------------------
#
# A rigid fit spreads one bad pair over every other one. With six landmarks per
# arch, a single point the network put on the opposing tooth moves the whole
# arch and leaves a residual that looks like ordinary noise. The fit is
# therefore refitted without each pair in turn, and a pair is dropped when
# leaving it out improves the fit enough to prove it was never the same point on
# both scans.

# Below this a rigid fit is underdetermined: one point is a plain translation,
# two leave a free rotation about the axis through them.
MIN_LANDMARK_PAIRS = 3

# Trimming stops here rather than at MIN_LANDMARK_PAIRS. A fit on exactly three
# pairs passes through them all, so its residual is zero whatever they are, and
# the test below would keep dropping pairs on the strength of a number that no
# longer measures anything.
MIN_LANDMARK_PAIRS_AFTER_TRIM = 4

# A pair has to be this far out before it is a candidate at all, and leaving it
# out has to bring the fit down to this share of what it was. Two conditions
# rather than one: the first stops ordinary noise being trimmed, the second
# stops a point being blamed for a fit that was poor for another reason.
LANDMARK_OUTLIER_FLOOR_MM = 3.0
LANDMARK_OUTLIER_IMPROVEMENT = 0.75

# Past this the two landmark sets do not describe the same points, and the fit
# would place the arch worse than not placing it. The ICP then starts from the
# pose the orientation left, which is further off but not actively wrong.
MAX_LANDMARK_RESIDUAL_MM = 10.0

# Landmarks nearly in a line leave the rotation about that line resting on their
# noise alone, and the residual cannot show it. A full arch measures about 0.32.
MIN_LANDMARK_SPREAD_RATIO = 0.10


def alignment_residual(moving_lms, fixed_lms, matrix) -> float:
    """RMS distance between the pairs once `matrix` has moved the moving side."""
    moved = apply(moving_lms, matrix)
    return float(np.sqrt(np.mean(np.sum((moved - fixed_lms) ** 2, axis=1))))


def landmark_spread_ratio(landmarks) -> float:
    """How far from a straight line the landmarks are, as a ratio in [0, 1].

    The second singular value of the centred points over the first: 0 is a
    perfect line, and a full arch measures about 0.32.
    """
    landmarks = np.asarray(landmarks, dtype=float)
    if len(landmarks) < 3:
        return 0.0
    singular = np.linalg.svd(landmarks - landmarks.mean(axis=0), compute_uv=False)
    return float(singular[1] / singular[0]) if singular[0] > 0 else 0.0


def fit_rigid_without_outliers(moving_lms, fixed_lms, labels=None, on_enamel=None):
    """Fit, drop any pair the fit cannot account for, fit again.

    Two tests, in the order their evidence is worth. A landmark the scan says is
    not on a tooth goes first and unconditionally: that verdict comes from the
    image, so it holds however the other landmarks look. What is left is then
    trimmed on the fit, ONE pair per round -- each drop changes every other
    residual, so deciding them all from one fit would throw away points that
    were never the problem.

    Returns the matrix and the indices it was fitted on.
    """
    moving_lms = np.asarray(moving_lms, dtype=float)
    fixed_lms = np.asarray(fixed_lms, dtype=float)
    keep = list(range(len(moving_lms)))
    labels = list(labels) if labels is not None else None

    if on_enamel and labels:
        off = [i for i in keep if on_enamel.get(labels[i]) is False]
        if off and len(keep) - len(off) >= MIN_LANDMARK_PAIRS:
            logger.warning(
                "%s %s not on a tooth in the CBCT, so the fit is made on the "
                "remaining %d.", ", ".join(labels[i] for i in off),
                "is" if len(off) == 1 else "are", len(keep) - len(off))
            keep = [i for i in keep if i not in off]
        elif off:
            logger.warning(
                "%d of %d CBCT landmarks are not on a tooth, which leaves too "
                "few to fit with. They are all kept, but this arch's landmarks "
                "should be corrected before its registration is used.",
                len(off), len(keep))

    while True:
        matrix = align_by_landmarks(moving_lms[keep], fixed_lms[keep])
        rms = alignment_residual(moving_lms[keep], fixed_lms[keep], matrix)
        residuals = np.linalg.norm(
            apply(moving_lms[keep], matrix) - fixed_lms[keep], axis=1)

        if len(keep) <= MIN_LANDMARK_PAIRS_AFTER_TRIM:
            break

        # What the fit would be without each pair in turn.
        without = []
        for position in range(len(keep)):
            subset = [k for j, k in enumerate(keep) if j != position]
            trial = align_by_landmarks(moving_lms[subset], fixed_lms[subset])
            without.append(
                (alignment_residual(moving_lms[subset], fixed_lms[subset], trial),
                 position))

        best, position = min(without)
        if not (residuals[position] > LANDMARK_OUTLIER_FLOOR_MM
                and best <= LANDMARK_OUTLIER_IMPROVEMENT * rms):
            break

        logger.warning(
            "%s sits %.1f mm from where the rest of the arch puts it, and the fit "
            "over the other %d landmarks is %.1f mm against %.1f mm with it. It is "
            "not the same point on both scans, so it is left out of the fit.",
            labels[keep[position]] if labels else "#%d" % keep[position],
            residuals[position], len(keep) - 1, best, rms)
        keep.pop(position)

    return matrix, keep


def prealign(moving_lms, fixed_lms, labels=None, on_enamel=None):
    """The pose the ICP starts from, and what it is worth.

    Returns `(matrix, report)`. The matrix is the identity when the landmarks
    cannot place the arch at all -- being under-determined is not being wrong,
    and the identity leaves the ICP to start from the pose the orientation left
    rather than from somewhere actively wrong.
    """
    moving_lms = np.asarray(moving_lms, dtype=float)
    fixed_lms = np.asarray(fixed_lms, dtype=float)

    if len(moving_lms) < MIN_LANDMARK_PAIRS:
        logger.warning(
            "%d landmark pair(s), %d needed for a rigid fit. Skipping the "
            "pre-alignment.", len(moving_lms), MIN_LANDMARK_PAIRS)
        return np.eye(4), {"accepted": False, "reason": "too few pairs",
                           "landmarks_fitted": [], "residual_mm": None}

    matrix, keep = fit_rigid_without_outliers(
        moving_lms, fixed_lms, labels, on_enamel)
    fitted = [labels[i] for i in keep] if labels else list(keep)
    rms = alignment_residual(moving_lms[keep], fixed_lms[keep], matrix)

    if rms > MAX_LANDMARK_RESIDUAL_MM:
        logger.warning(
            "%.1f mm residual over %d pairs (threshold %.1f). The intraoral and "
            "CBCT landmarks do not describe the same points. Skipping the "
            "pre-alignment.", rms, len(keep), MAX_LANDMARK_RESIDUAL_MM)
        return np.eye(4), {"accepted": False, "reason": "residual too large",
                           "landmarks_fitted": fitted, "residual_mm": rms}

    spread = landmark_spread_ratio(moving_lms[keep])
    if spread < MIN_LANDMARK_SPREAD_RATIO:
        logger.warning(
            "The %d landmarks are nearly in a straight line (spread %.2f, under "
            "the %.2f floor; a full arch measures about 0.32). The rotation about "
            "that line rests on their noise alone, and the %.1f mm residual "
            "cannot show it.", len(keep), spread, MIN_LANDMARK_SPREAD_RATIO, rms)

    logger.info("Pre-alignment accepted, %.1f mm residual over %d pair(s) (%s).",
                rms, len(keep), ", ".join(str(f) for f in fitted))
    return matrix, {"accepted": True, "reason": None,
                    "landmarks_fitted": fitted, "residual_mm": rms,
                    "spread": spread}


# ---------------------------------------------------------------------------
# The refinement: a point-to-plane ICP on the crowns
# ---------------------------------------------------------------------------
#
# Point-to-point pulls a surface towards particular neighbours, which on the
# smooth, near-flat occlusal surfaces here means it slides along them and
# stalls. Measuring along the target normal lets the surface slide freely and
# only resists what actually separates the two surfaces. `icp_point_to_point`
# above is kept as the fallback for a degenerate system, which is the one place
# upstream still uses it.

# How close a point must be to its nearest neighbour to count as a
# correspondence. Upstream's function signature says 1.5 and every call site
# passes 1.0; the call sites are what produced the results, so 1.0 it is.
ICP_MAX_DIST_MM = 1.0

# How much two normals must agree to be the same surface. Below this the nearest
# point is on something facing the other way -- in a closed bite, the opposing
# arch, which is the failure this test exists for.
MIN_NORMAL_AGREEMENT = 0.5

# When the ICP has stopped moving: this little movement, this many times over.
ICP_SETTLED_SHIFT_MM = 1e-3
ICP_SETTLED_ITERATIONS = 3
ICP_MAX_ITERATIONS = 300

# How far around an arch's own landmarks to keep the CBCT surface. Wide enough
# that nothing a crown could match to is cut away, narrow enough to drop the
# cranial base, the vertebrae and the far side of the jaw.
CBCT_CROP_MARGIN_MM = 25.0

# An ICP that matched almost nothing still returns a matrix. Writing it out put
# an untouched intraoral scan in the results under the name of a registered one.
MIN_ICP_FITNESS = 0.05

# How far the ICP may pull the landmarks away from where the fit put them before
# it is worth saying so. The ICP is right to move them a little -- it has forty
# thousand points and they have six -- but not far.
MAX_ICP_LANDMARK_DRIFT_MM = 1.0

# Which `Universal_ID` values are not a tooth. An intraoral scan is about 60%
# gingiva, and gingiva has no counterpart in a CBCT surface: the nearest thing
# under it is alveolar bone, a millimetre or two further in. Those points are
# not matched so much as dragged, and they outnumber the crowns.
GINGIVA_LABELS = (0, 33)

# Below this share of crowns the labels are not worth trusting, and the whole
# arch is registered instead.
MIN_CROWN_SHARE = 0.10


def point_normals(mesh):
    """Per-point outward normals, in the order of `mesh.points`, or None.

    Point normals, not cell normals: cell normals are one per triangle, so
    indexing them with a point index reads the normal of an unrelated part of
    the surface.
    """
    if "Normals" in mesh.point_data:
        return np.asarray(mesh.point_data["Normals"], dtype=float)
    try:
        with_normals = mesh.compute_normals(
            point_normals=True, cell_normals=False,
            auto_orient_normals=False, inplace=False)
        return np.asarray(with_normals.point_data["Normals"], dtype=float)
    except Exception as exc:  # noqa: BLE001 - a surface without normals still registers
        logger.warning("No surface normals could be computed (%s). The opposing "
                       "arch cannot be told apart by orientation.", exc)
        return None


class Target:
    """The CBCT surface as the ICP consumes it: points and normals, nothing else.

    Cropping is then index selection on two arrays. Cutting the mesh itself
    instead costs more than the ICP saves: rebuilding a 2.8 million point
    surface is dearer than querying it.

    Normals are taken from the whole surface before any crop, so a point keeps
    the normal its neighbourhood gives it rather than one bent by the cut.
    """

    def __init__(self, points, normals):
        self.points = points
        self.normals = normals

    @classmethod
    def from_mesh(cls, mesh):
        return cls(np.asarray(mesh.points, dtype=float), point_normals(mesh))

    def __len__(self):
        return len(self.points)

    def around(self, anchors, margin=CBCT_CROP_MARGIN_MM):
        """The part of the surface this arch could plausibly be registered to.

        Anchored on the arch's own CBCT landmarks, not on the pre-aligned
        intraoral scan: the landmarks are in CBCT coordinates whatever the
        pre-alignment did, while a scan whose pre-alignment was skipped is still
        in the frame the orientation left it in and would drag the box across the
        whole head.

        The opposing arch stays in the box -- at this margin it cannot be
        excluded by position, and it does not need to be, the normals tell it
        apart.
        """
        if anchors is None or len(anchors) == 0:
            logger.warning("No CBCT landmark to crop around; the whole surface "
                           "is kept as the target.")
            return self

        anchors = np.asarray(anchors, dtype=float)
        inside = np.all((self.points >= anchors.min(axis=0) - margin)
                        & (self.points <= anchors.max(axis=0) + margin), axis=1)

        kept = int(np.sum(inside))
        if kept < 3:
            logger.warning("Nothing of the CBCT surface lies within %.0f mm of "
                           "this arch's landmarks; the whole surface is kept as "
                           "the target.", margin)
            return self

        logger.info("CBCT target cropped to %d of %d points (%.1f%%) within "
                    "%.0f mm of the arch's landmarks.",
                    kept, len(self), 100.0 * kept / max(len(self), 1), margin)
        return Target(self.points[inside],
                      self.normals[inside] if self.normals is not None else None)


def crown_points(mesh):
    """A mask over `mesh.points` selecting the crowns, or None for all of them.

    Registering on the crowns alone does not move the answer much -- it is the
    same anatomy either way -- but it roughly doubles the share of the moving
    surface that finds a real match, which is what the run is judged on.
    """
    if "Universal_ID" not in mesh.point_data:
        logger.info("No Universal_ID on the intraoral scan; the whole arch is "
                    "registered, gingiva included.")
        return None

    crowns = ~np.isin(np.asarray(mesh.point_data["Universal_ID"]), GINGIVA_LABELS)
    share = float(np.mean(crowns)) if crowns.size else 0.0
    if share < MIN_CROWN_SHARE:
        logger.warning("Only %.0f%% of the intraoral scan is labelled as crowns, "
                       "which is too little to trust. The whole arch is "
                       "registered instead.", 100 * share)
        return None

    logger.info("Registering on the %d crown point(s) of %d (%.0f%%); the gingiva "
                "has no counterpart in the CBCT.",
                int(np.sum(crowns)), len(crowns), 100 * share)
    return crowns


def _point_to_plane_step(source, target, normals):
    """The rigid step minimising distance to the target's tangent planes.

    Linearised in the rotation: for a correspondence (p, q, n) the residual is
    (p - q).n + w.(p x n) + t.n, which is linear in the six unknowns [w, t].

    Returns None when the system cannot be trusted, and the caller falls back to
    the point-to-point step.
    """
    from scipy.spatial.transform import Rotation

    a = np.hstack([np.cross(source, normals), normals])
    b = np.einsum("ij,ij->i", target - source, normals)

    solution, _residuals, rank, _singular = np.linalg.lstsq(a, b, rcond=None)
    if rank < 6 or not np.all(np.isfinite(solution)):
        return None

    omega, translation = solution[:3], solution[3:]
    # A linearised step is only meaningful while it stays small; a large one
    # means the system is being driven by outliers rather than by the surface.
    if np.linalg.norm(omega) > 0.5:
        return None

    step = np.eye(4)
    step[:3, :3] = Rotation.from_rotvec(omega).as_matrix()
    step[:3, 3] = translation
    return step


def _point_to_point_step(source, target):
    """Procrustes fit, the fallback when the point-to-plane system is degenerate."""
    source_centre, target_centre = source.mean(axis=0), target.mean(axis=0)
    u, _s, vt = np.linalg.svd((source - source_centre).T @ (target - target_centre))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T

    step = np.eye(4)
    step[:3, :3] = rotation
    step[:3, 3] = target_centre - rotation @ source_centre
    return step


def _icp_run(moving_points, moving_normals, target, tree, sign, max_dist):
    """One ICP, under one reading of which way the two meshes wind their faces.

    `sign` is +1 when a normal means the same thing on both meshes and -1 when
    one of them is wound the other way; it multiplies the CBCT normals before
    they are compared and before they are used as tangent planes.
    """
    use_normals = moving_normals is not None and target.normals is not None

    transformation = np.eye(4)
    current = moving_points.copy()
    current_normals = moving_normals.copy() if use_normals else None

    inlier_rmse, fitness, pairs, rejected_by_normal, settled, iteration = (
        float("inf"), 0.0, 0, 0, 0, 0)

    for iteration in range(ICP_MAX_ITERATIONS):
        # workers=-1 spreads the query over every core: it is the dominant cost
        # of the loop, one lookup per moving point per iteration.
        distances, indices = tree.query(current, k=1, workers=-1)

        near = distances < max_dist
        valid = near
        if use_normals:
            agreement = np.einsum("ij,ij->i", current_normals,
                                  sign * target.normals[indices])
            valid = near & (agreement > MIN_NORMAL_AGREEMENT)
            # The peak, not the last iteration: once the arch has settled on its
            # own side nothing nearby faces the wrong way any more, so the final
            # count says nothing about how ambiguous the start was.
            rejected_by_normal = max(rejected_by_normal,
                                     int(np.sum(near & ~valid)))

        inlier_rmse = (float(np.sqrt(np.mean(distances[valid] ** 2)))
                       if np.any(valid) else float("inf"))
        fitness = float(np.sum(valid)) / len(moving_points)
        pairs = int(np.sum(valid))

        # Checked here rather than after the update, so the numbers reported are
        # the ones that describe the transform actually returned.
        if settled >= ICP_SETTLED_ITERATIONS:
            break
        if pairs < 3:
            logger.debug("Iteration %d has %d usable correspondence(s); the ICP "
                         "stops here.", iteration, pairs)
            break

        matched = target.points[indices[valid]]
        step = None
        if use_normals:
            step = _point_to_plane_step(current[valid], matched,
                                        sign * target.normals[indices[valid]])
        if step is None:
            step = _point_to_point_step(current[valid], matched)

        transformation = step @ transformation
        previous = current
        current = apply(moving_points, transformation)
        if use_normals:
            current_normals = moving_normals @ transformation[:3, :3].T

        shift = float(np.max(np.linalg.norm(current - previous, axis=1)))
        settled = settled + 1 if shift < ICP_SETTLED_SHIFT_MM else 0

    return transformation, {
        "iterations": iteration,
        "settled": settled >= ICP_SETTLED_ITERATIONS,
        "fitness": fitness,
        "inlier_rmse": inlier_rmse,
        "pairs": pairs,
        "rejected_by_normal": rejected_by_normal,
        "used_normals": use_normals,
        "sign": sign,
    }


def icp_point_to_plane(moving_mesh, target, max_dist=ICP_MAX_DIST_MM):
    """Refine an intraoral scan onto a CBCT surface. Returns `(matrix, stats)`.

    Whether a normal points out of a tooth or into it is a property of how each
    file was written, and the two modalities do not have to agree. It cannot be
    read off the starting pose either: that is precisely where a scan sitting
    between the two arches has most of its nearest neighbours on the wrong one,
    and averaging over them reads the bite as an inversion and then locks the
    registration onto the opposing arch. So both readings are registered and the
    one that fits the CBCT better is kept -- the crowns are the same anatomy as
    their own arch in the CBCT and nothing else, so the right reading wins on
    the merits.
    """
    from scipy.spatial import cKDTree

    if not isinstance(target, Target):
        target = Target.from_mesh(target)
    if len(target) < 3:
        raise ValueError("A point-to-plane ICP needs at least 3 target points.")

    moving_points = np.asarray(moving_mesh.points, dtype=float)
    moving_normals = point_normals(moving_mesh)

    # Only the crowns drive the fit; the whole arch still moves by the matrix
    # they settle on, gingiva included, so nothing is lost from the output.
    crowns = crown_points(moving_mesh)
    if crowns is not None:
        moving_points = moving_points[crowns]
        if moving_normals is not None:
            moving_normals = moving_normals[crowns]
    if len(moving_points) < 3:
        raise ValueError("A point-to-plane ICP needs at least 3 moving points.")

    use_normals = moving_normals is not None and target.normals is not None
    # Only the moving points change from one iteration to the next, so the tree
    # over the fixed points is built once rather than per iteration.
    tree = cKDTree(target.points)

    attempts = []
    for sign in ((1.0, -1.0) if use_normals else (1.0,)):
        attempts.append(_icp_run(moving_points, moving_normals, target, tree,
                                 sign, max_dist))
        if use_normals:
            logger.debug("Normals read as %s gives %.1f%% matched at %.3f mm.",
                         "aligned" if sign > 0 else "opposed",
                         100 * attempts[-1][1]["fitness"],
                         attempts[-1][1]["inlier_rmse"])

    # More of the scan matched is the first thing that matters; a tie on that is
    # broken by how closely it matched.
    transformation, stats = max(
        attempts, key=lambda a: (round(a[1]["fitness"], 3), -a[1]["inlier_rmse"]))

    # Only of the attempt that was kept: the reading of the normals that loses is
    # expected to wander, and saying so about it reads as a doubt over the answer
    # actually returned.
    if not stats["settled"]:
        logger.warning("The ICP used all %d iterations without settling to within "
                       "%g mm. Its answer is wherever it had got to.",
                       ICP_MAX_ITERATIONS, ICP_SETTLED_SHIFT_MM)
    if use_normals and stats["sign"] < 0:
        logger.info("The intraoral scan and the CBCT surface wind their faces the "
                    "opposite way; the normals were flipped to compare them.")

    logger.info("ICP done in %d iterations, %.1f%% of the scan matched (%d points) "
                "at %.2f mm RMSE%s.",
                stats["iterations"], 100 * stats["fitness"], stats["pairs"],
                stats["inlier_rmse"],
                ", up to %d nearby points dropped as the opposing surface"
                % stats["rejected_by_normal"] if stats["rejected_by_normal"] else "")
    return transformation, stats
