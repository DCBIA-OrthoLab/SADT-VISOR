"""Rigidly aligning two palatal patches, and writing the result as a transform.

Ported from `AREG_IOS/AREG_IOS_utils/ICP.py` (ICP / vtkICP) and the
`saveMatrixAsTfm` / `read_matrix` half of `transformation.py`.

Two things the originals got wrong, both invisible:

* `vtkICP.__call__` returned its source UNMOVED: it built a
  `vtkTransformPolyDataFilter`, ran it, and returned the input polydata,
  throwing the filter's output away. `ICP.run` fed that into the next entry of
  `list_icp` and multiplied the matrices together, so every method after the
  first ran on unaligned points while its matrix was composed as if it had not.
  Only one method was ever configured, which is why it never showed. There is
  one alignment here and no list to compose.
* `read_matrix` read `GetTranslation()`, which is the whole translation only
  when the centre of rotation is the origin. A transform about any other centre
  maps `y = R(x - c) + c + t`, an offset of `t + c - Rc` -- `GetOffset()` in
  SimpleITK. The two happen to be equal here, ASO writing its transforms
  centred on the origin, so this is the same correction the CBCT engine needed
  for elastix, applied before it can bite rather than after.
"""

import numpy as np
import SimpleITK as sitk
import vtk

from . import surfaces

MAX_ITERATIONS = 1000


class RegistrationError(Exception):
    """One patient could not be registered. Reported, and the batch goes on."""


def align(source: vtk.vtkPolyData, target: vtk.vtkPolyData, stats: dict = None) -> np.ndarray:
    """The 4x4 matrix moving `source`'s points onto `target`'s.

    Both are the vertex-only patch clouds `butterfly.patch_cloud` builds.
    `stats`, when given, receives what the alignment itself measured and
    `measure` cannot recompute: the number of iterations it ran.
    """
    if source.GetNumberOfPoints() == 0 or target.GetNumberOfPoints() == 0:
        raise RegistrationError("one of the two patches holds no point")

    icp = vtk.vtkIterativeClosestPointTransform()
    icp.SetSource(source)
    icp.SetTarget(target)
    icp.GetLandmarkTransform().SetModeToRigidBody()
    icp.SetMaximumNumberOfIterations(MAX_ITERATIONS)
    icp.StartByMatchingCentroidsOn()
    icp.Modified()
    icp.Update()

    matrix = surfaces.matrix_from_vtk(icp.GetMatrix())
    if not np.all(np.isfinite(matrix)):
        raise RegistrationError("the alignment did not converge to a usable transform")
    if stats is not None:
        # GetMeanDistance is NOT reported: without CheckMeanDistanceOn, which
        # this alignment never set, VTK leaves it at 0, and even with it on it
        # is how far the points moved in the last iteration, not how far they
        # are from the target. The iteration count is real, and with no
        # stopping test it is always MAX_ITERATIONS.
        stats["iterations"] = int(icp.GetNumberOfIterations())
    return matrix


def _nearest_distances(points: np.ndarray, target: vtk.vtkPolyData) -> np.ndarray:
    """Distance from each of `points` to the closest point of `target`, in mm."""
    locator = vtk.vtkStaticPointLocator()
    locator.SetDataSet(target)
    locator.BuildLocator()
    target_points = surfaces.points_of(target)
    nearest = np.fromiter((locator.FindClosestPoint(point) for point in points),
                          dtype=np.int64, count=len(points))
    return np.linalg.norm(points - target_points[nearest], axis=1)


def measure(source: vtk.vtkPolyData, target: vtk.vtkPolyData, matrix: np.ndarray,
            iterations: int = None) -> dict:
    """How far `source`'s patch sits from `target`'s, before and after `matrix`.

    The cost `vtkIterativeClosestPointTransform` minimises is the mean squared
    distance from every source point to its closest target point (rigid, all
    points kept, none rejected). It exposes no history of it, so it is
    evaluated here where it starts -- the source translated onto the target's
    centroid, as `StartByMatchingCentroidsOn` does before the first iteration --
    and where it ends, the source moved by `matrix`. The final distances
    themselves, unsquared, are the surface distance after registration.
    """
    points = surfaces.points_of(source)
    target_points = surfaces.points_of(target)
    start = points + (target_points.mean(axis=0) - points.mean(axis=0))
    moved = points @ matrix[:3, :3].T + matrix[:3, 3]

    before = _nearest_distances(start, target)
    after = _nearest_distances(moved, target)
    return {
        "cost": "point-to-point",
        # [(iteration, mean squared distance in mm^2)], first and last.
        "cost_milestones": [(0, round(float(np.mean(before ** 2)), 4)),
                            (int(iterations or 0), round(float(np.mean(after ** 2)), 4))],
        "iterations": iterations,
        "surface_mean_mm": round(float(np.mean(after)), 4),
        "surface_std_mm": round(float(np.std(after)), 4),
        "surface_max_mm": round(float(np.max(after)), 4),
        "points": int(len(points)),
    }


# ---------------------------------------------------------------------------
# Transform files
# ---------------------------------------------------------------------------

def matrix_of(transform: sitk.Transform) -> np.ndarray:
    """A 4x4 matrix from a SimpleITK linear transform, centre included.

    A centred transform maps `y = R(x - c) + c + t`, so the 4x4 translation
    column is `t + c - Rc`, not `t`. SimpleITK has no accessor for it -- there
    is `GetTranslation()` and `GetCenter()` and nothing that returns the
    composition -- which is exactly why `read_matrix` reading `GetTranslation()`
    into `matrix[:3, 3]` looks correct and is not.
    """
    try:
        linear = np.asarray(transform.GetMatrix(), dtype=np.float64).reshape(3, 3)
        translation = np.asarray(transform.GetTranslation(), dtype=np.float64)
        center = np.asarray(transform.GetCenter(), dtype=np.float64)
    except AttributeError as exc:
        raise RegistrationError(
            f"'{transform.GetName()}' is not a linear transform, so it cannot be "
            f"composed with the registration."
        ) from exc

    matrix = np.eye(4)
    matrix[:3, :3] = linear
    matrix[:3, 3] = translation + center - linear @ center
    return matrix


def write_transform(matrix: np.ndarray, path: str, prior_path: str = None) -> str:
    """Write the resampling transform for a point-moving matrix.

    `matrix` moves the moving mesh's POINTS onto the fixed one. A `.tfm` is
    consumed the other way round -- it maps a point of the result back to where
    to sample it -- so the inverse is what is written, exactly as the CBCT
    engine writes elastix's output.

    `prior_path` is a transform already applied to the mesh before AREG saw it,
    which is the case in the automated modes: ASO oriented both timepoints
    first, so a transform relating AREG's inputs would refer to meshes the
    caller did not send. Composing the two gives one file mapping the caller's
    ORIGINAL mesh to the registered result.
    """
    combined = matrix
    if prior_path:
        # original --prior--> oriented --matrix--> registered
        combined = matrix @ np.linalg.inv(matrix_of(sitk.ReadTransform(prior_path)))

    inverted = np.linalg.inv(combined)
    transform = sitk.AffineTransform(3)
    transform.SetMatrix(inverted[:3, :3].flatten().tolist())
    transform.SetTranslation(inverted[:3, 3].tolist())
    sitk.WriteTransform(transform, path)
    return path
