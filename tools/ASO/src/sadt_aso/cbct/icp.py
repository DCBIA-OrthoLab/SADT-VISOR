"""CBCT landmark registration: outlier rejection, coarse alignment, ICP.

Ported from `ASO_CBCT/ASO_CBCT_utils/utils.py`. The algorithm is unchanged --
same outlier thresholds, same three-landmark initialisation, same
`vtkIterativeClosestPointTransform` settings, same composite transform. What
changed is that it works on landmark DICTS rather than on file paths, so the
fully-automated mode can hand over predictions that were never written to disk,
and so a caller can be told exactly which landmarks were dropped and why.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import SimpleITK as sitk
import vtk

from .. import geometry, user_metrics

logger = logging.getLogger(__name__)

# Fewest landmarks the coarse alignment can work with: it picks three distinct
# ones. The original checked `> 3` in `ICP()` and `< 3` four lines later, and
# the Slicer UI enforced `>= 3` -- so a selection of exactly three passed the UI
# and was then rejected by a function that said only "ICP registration returned
# None".
MIN_LANDMARKS = 3

# Outlier thresholds, unchanged from the original: a landmark whose distance to
# the others differs from the reference by more than 15 mm, or whose direction
# differs by more than 0.4 rad, is counted against.
_MAX_DISTANCE_DIFFERENCE_MM = 15.0
_MAX_DIRECTION_DIFFERENCE_RAD = 0.4
_MIN_DIRECTION_DIFFERENCE_RAD = 0.1


class RegistrationError(Exception):
    """Raised when one patient cannot be registered.

    Carries a message meant for the run report: it explains what was missing,
    never a stack trace, and never patient data.
    """


@dataclass
class Registration:
    """What one patient's registration produced."""

    # Maps an OUTPUT point back to an INPUT one, which is the direction
    # sitk.ResampleImageFilter wants. Its inverse is the forward move the
    # voxels undergo -- and the one the landmarks are moved by.
    resample_transform: sitk.Transform
    # The transform back to the ORIGINAL scan space, recentring included: it
    # maps ORIENTED -> ORIGINAL, so a point measured on the result comes back
    # onto the acquisition. This is what gets written as the .tfm next to the
    # oriented volume. The direction is worth stating because getting it
    # backwards is silent -- the file still loads, and still transforms.
    full_transform: sitk.Transform
    landmarks: dict
    used: list = field(default_factory=list)
    dropped: dict = field(default_factory=dict)
    # What the registration measured, for the user and the report: see
    # `user_metrics.metric_lines` for what each entry is.
    metrics: dict = field(default_factory=dict)


def landmarks_to_remove(source: dict, target: dict) -> list:
    """Landmarks whose position disagrees with the reference badly enough that
    including them would drag the registration off.

    Both sides are restricted to the labels they SHARE first. The original
    computed the pairwise tables independently and then indexed the reference's
    with the input's keys (`gold[lm][lm2]`), so one landmark present in the
    input and absent from the reference raised a KeyError that was caught far
    enough up to lose the whole patient.
    """
    shared = [name for name in source if name in target]
    if len(shared) < 2:
        return []
    source = {name: source[name] for name in shared}
    target = {name: target[name] for name in shared}

    distance_count = _count_over(
        _pairwise_difference(target, source, _distances),
        max_difference=_MAX_DISTANCE_DIFFERENCE_MM,
    )
    direction_count = _count_over(
        _angular_difference(target, source),
        max_difference=_MAX_DIRECTION_DIFFERENCE_RAD,
        min_difference=_MIN_DIRECTION_DIFFERENCE_RAD,
    )

    total = {name: distance_count[name] + direction_count[name] for name in shared}
    removed = [name for name in shared if total[name] > len(shared)]
    removed.extend(
        name
        for name in shared
        if direction_count[name] > len(shared) // 2 and name not in removed
    )
    return removed


def register(
    source_landmarks: dict,
    target_landmarks: dict,
    requested: list,
    pre_transform: sitk.Transform = None,
    max_triplets: int = 2500,
    seed: int = 0,
) -> Registration:
    """Register one patient's landmarks onto the reference's.

    `requested` is what the caller asked to register on; landmarks it names that
    either side does not have, and landmarks rejected as outliers, are reported
    in `Registration.dropped` rather than silently ignored.

    `pre_transform` is the recentring translation applied to the volume before
    this (see pipeline.recenter). It is folded into `full_transform` only --
    the resampling transform deliberately carries rotations alone, because both
    volumes are already centred on the physical origin.
    """
    dropped = {}
    shared = []
    for name in requested:
        if name not in source_landmarks:
            dropped[name] = "not in the input landmark file"
        elif name not in target_landmarks:
            dropped[name] = "not in the reference landmark file"
        else:
            shared.append(name)

    for name in landmarks_to_remove(source_landmarks, target_landmarks):
        if name in shared:
            shared.remove(name)
            dropped[name] = "rejected as an outlier"

    if len(shared) < MIN_LANDMARKS:
        raise RegistrationError(
            f"only {len(shared)} usable landmark(s), {MIN_LANDMARKS} are needed "
            f"(dropped: {', '.join(sorted(dropped)) or 'none'})"
        )

    # Sorted so the search is over a stable order regardless of the order the
    # markups file happened to list its control points in.
    source = {name: source_landmarks[name] for name in sorted(shared)}
    target = {name: target_landmarks[name] for name in sorted(shared)}

    picks = geometry.best_triplet(
        source, target, include_translation=False, max_triplets=max_triplets, seed=seed
    )
    coarse, coarse_matrix, steps = geometry.init_icp(
        source, target, picks, include_translation=False
    )

    icp_matrix, icp_iterations = _icp_matrix(coarse, target)
    # Measured on the pose the ICP ended in, translation included: that is the
    # quantity it minimised, before the translation is dropped below.
    icp_rms = user_metrics.closest_point_rms(
        geometry.transform_landmarks(coarse, icp_matrix).values(), target.values()
    )
    # The ICP translation is dropped for the same reason the coarse one is: the
    # scans share the physical origin already.
    icp_matrix[:3, 3] = 0.0

    transform_steps = _sitk_steps(steps)
    icp_step = sitk.Euler3DTransform()
    icp_step.SetMatrix(icp_matrix[:3, :3].flatten().tolist())
    transform_steps.append(icp_step)

    # Skips index 0 -- the coarse translation, which is not part of the
    # resampling for the reason above.
    rotations = sitk.CompositeTransform(3)
    for step in reversed(transform_steps[1:]):
        rotations.AddTransform(step)

    full = sitk.CompositeTransform(rotations)
    if pre_transform is not None:
        full.AddTransform(pre_transform)

    landmarks = geometry.transform_landmarks(source_landmarks, icp_matrix @ coarse_matrix)
    labels = sorted(shared)
    rotation_deg, translation_mm = user_metrics.rotation_and_translation(
        icp_matrix @ coarse_matrix @ _matrix_of(pre_transform)
    )
    metrics = {
        "points": list(user_metrics.POINTS_LANDMARK),
        "labels": labels,
        "left_out": dict(dropped),
        # Both steps rotate only, as applied: the scan and the reference are
        # each centred on the origin, and what is left between the two centred
        # frames is an offset the orientation never meant to remove. Measured
        # with the centres matched, so the distance grades the rotation; a raw
        # distance read 17 mm on a rotation recovered exactly.
        "stages": [
            {"after": user_metrics.STEP_THREE_POINT, "centred": True,
             "distances_mm": user_metrics.distances(
                 geometry.transform_landmarks(source, coarse_matrix), target, labels,
                 centred=True)},
            {"after": user_metrics.STEP_ORIENTATION, "centred": True,
             "distances_mm": user_metrics.distances(landmarks, target, labels, centred=True)},
        ],
        "icp": {"iterations": icp_iterations, "rms_closest_mm": icp_rms,
                "translation_applied": False},
        # Of the whole move from the ORIGINAL scan to the oriented one,
        # recentring included: the translation is the recentring's length.
        "rotation_deg": rotation_deg,
        "translation_mm": translation_mm,
    }

    return Registration(
        resample_transform=rotations.GetInverse(),
        full_transform=full.GetInverse(),
        landmarks=landmarks,
        used=labels,
        dropped=dropped,
        metrics=metrics,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _sitk_steps(steps: list) -> list:
    transforms = []
    for kind, payload in steps:
        if kind == geometry.STEP_TRANSLATION:
            step = sitk.TranslationTransform(3)
            step.SetOffset([float(value) for value in payload])
        else:
            step = sitk.VersorRigid3DTransform()
            step.SetMatrix(np.asarray(payload).flatten().tolist())
        transforms.append(step)
    return transforms


def _matrix_of(transform) -> np.ndarray:
    """The 4x4 of a linear SimpleITK transform (identity for None), read off
    where it sends the origin and the three unit vectors."""
    matrix = np.eye(4)
    if transform is None:
        return matrix
    origin = np.array(transform.TransformPoint((0.0, 0.0, 0.0)))
    for column, unit in enumerate(np.eye(3)):
        matrix[:3, column] = np.array(transform.TransformPoint(tuple(unit))) - origin
    matrix[:3, 3] = origin
    return matrix


def _icp_matrix(source: dict, target: dict) -> tuple:
    """`(4x4 ICP matrix, iterations it ran)`."""
    icp = vtk.vtkIterativeClosestPointTransform()
    icp.SetSource(_to_polydata(source))
    icp.SetTarget(_to_polydata(target))
    icp.GetLandmarkTransform().SetModeToRigidBody()
    icp.SetMaximumNumberOfIterations(1000)
    icp.StartByMatchingCentroidsOn()
    icp.Modified()
    icp.Update()

    matrix = icp.GetMatrix()
    return np.array(
        [[matrix.GetElement(row, col) for col in range(4)] for row in range(4)]
    ), int(icp.GetNumberOfIterations())


def _to_polydata(landmarks: dict) -> vtk.vtkPolyData:
    points = vtk.vtkPoints()
    vertices = vtk.vtkCellArray()
    labels = vtk.vtkStringArray()
    labels.SetNumberOfValues(len(landmarks))
    labels.SetName("labels")

    for index, (name, position) in enumerate(landmarks.items()):
        point_id = points.InsertNextPoint([float(value) for value in position])
        vertices.InsertNextCell(1)
        vertices.InsertCellPoint(point_id)
        labels.SetValue(index, name)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetVerts(vertices)
    polydata.GetPointData().AddArray(labels)
    return polydata


def _distances(landmarks: dict) -> dict:
    return {
        first: {
            second: float(np.linalg.norm(landmarks[first] - landmarks[second]))
            for second in landmarks
            if second != first
        }
        for first in landmarks
    }


def _directions(landmarks: dict) -> dict:
    return {
        first: {
            second: landmarks[first] - landmarks[second]
            for second in landmarks
            if second != first
        }
        for first in landmarks
    }


def _pairwise_difference(reference: dict, test: dict, measure) -> dict:
    reference_table, test_table = measure(reference), measure(test)
    return {
        first: {
            second: abs(reference_table[first][second] - value)
            for second, value in row.items()
        }
        for first, row in test_table.items()
    }


def _angular_difference(reference: dict, test: dict) -> dict:
    reference_table, test_table = _directions(reference), _directions(test)
    differences = {}
    for first, row in test_table.items():
        differences[first] = {}
        for second, vector in row.items():
            other = reference_table[first][second]
            norms = np.linalg.norm(other) * np.linalg.norm(vector)
            if norms == 0:
                differences[first][second] = 0.0
                continue
            # Clamped: two coincident landmarks give a dot product a hair over
            # 1, and np.arccos of that is NaN -- which compares False against
            # every threshold, so an outlier scored as a good landmark.
            differences[first][second] = float(
                np.arccos(np.clip(np.dot(other, vector) / norms, -1.0, 1.0))
            )
    return differences


def _count_over(differences: dict, max_difference: float, min_difference: float = None) -> dict:
    counts = {}
    for name, row in differences.items():
        count = 0
        for value in row.values():
            if value > max_difference:
                count += 1
            if min_difference is not None and value < min_difference:
                count -= 1
        counts[name] = count
    return counts
