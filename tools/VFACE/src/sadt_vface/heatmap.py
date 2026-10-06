"""Where the two surfaces differ, drawn on the baseline.

A heat map is the other half of what VFACE answers. The measurements say how
far a landmark moved; this says the same thing everywhere at once, as a signed
distance from each point of the baseline surface to the nearest point of the
one it is compared against -- the patient's own mirror, or their follow-up.

Signed, and that matters: an unsigned distance cannot tell a side that grew
from one that did not, which is the whole question.

**Upstream runs each pair in a subprocess of its own**, because
`vtkDistancePolyDataFilter` allocates cell locators and BSP trees on the C++
side that a garbage collection inside Slicer's long-lived process does not
reclaim. That is a Slicer-era workaround and it does not carry: a tool here is
already its own process, started by the runner and exited at the end of the
run. What does carry is the subsampling, which is the actual memory control --
the filter's cost goes as the product of the two point counts, and a pair of
full-head surfaces is millions each.
"""

import logging
import os

from .errors import describe_failure, most_common_failure
from . import catalogs, landmarks as landmark_files, tools

logger = logging.getLogger(__name__)

SURFACE_EXTENSIONS = (".vtk", ".vtp")
DISTANCE_ARRAY = "Distance"

# Combined point counts past which the pair is decimated before the distance is
# computed, and by how much. Upstream's thresholds.
SUBSAMPLE_ABOVE = 1_000_000
HEAVY_SUBSAMPLE_ABOVE = 2_000_000
MODERATE_REDUCTION = 0.5
HEAVY_REDUCTION = 0.7

# Below this a decimation costs more than it saves, and on a small surface it
# would take out detail the map is meant to show.
MIN_POINTS_TO_DECIMATE = 1000


def read_surface(path: str):
    import vtk

    reader = (vtk.vtkXMLPolyDataReader() if path.lower().endswith(".vtp")
              else vtk.vtkPolyDataReader())
    reader.SetFileName(str(path))
    reader.Update()
    surface = vtk.vtkPolyData()
    surface.DeepCopy(reader.GetOutput())
    return surface


def write_surface(surface, path: str) -> str:
    import vtk

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if path.lower().endswith(".vtp"):
        writer = vtk.vtkXMLPolyDataWriter()
    else:
        writer = vtk.vtkPolyDataWriter()
        # Version 42, the legacy format every Slicer in use reads. The writer's
        # default is 5.1, which older readers refuse outright.
        writer.SetFileVersion(42)
    writer.SetFileName(str(path))
    writer.SetInputData(surface)
    writer.Write()
    return path


def _clean(surface):
    """Merged duplicate points and nothing but triangles.

    The distance filter builds a locator over the cells, and a polygon that is
    not a triangle -- or two coincident points -- gives it a degenerate one.
    """
    import vtk

    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(surface)
    cleaner.SetAbsoluteTolerance(1e-6)
    cleaner.Update()

    triangles = vtk.vtkTriangleFilter()
    triangles.SetInputData(cleaner.GetOutput())
    triangles.Update()

    result = vtk.vtkPolyData()
    result.DeepCopy(triangles.GetOutput())
    return result


def _decimated(surface, reduction: float):
    import vtk

    if surface.GetNumberOfPoints() < MIN_POINTS_TO_DECIMATE:
        return surface
    decimation = vtk.vtkQuadricDecimation()
    decimation.SetInputData(surface)
    decimation.SetTargetReduction(reduction)
    decimation.Update()

    result = vtk.vtkPolyData()
    result.DeepCopy(decimation.GetOutput())
    return result


def _distance(moving, fixed, signed: bool = True):
    import vtk

    distance = vtk.vtkDistancePolyDataFilter()
    distance.SetInputData(0, moving)
    distance.SetInputData(1, fixed)
    if signed:
        distance.SignedDistanceOn()
    else:
        distance.SignedDistanceOff()
    # Only the first surface is being drawn on; computing the second direction
    # doubles the work for an answer nothing reads.
    distance.ComputeSecondDistanceOff()
    distance.Update()

    result = vtk.vtkPolyData()
    result.DeepCopy(distance.GetOutput())
    if result.GetPointData().GetArray(DISTANCE_ARRAY) is None:
        for index in range(result.GetPointData().GetNumberOfArrays()):
            array = result.GetPointData().GetArray(index)
            if array is not None and array.GetName() and "istance" in array.GetName():
                array.SetName(DISTANCE_ARRAY)
                break
    return result


def _interpolate_back(original, decimated):
    """The decimated surface's distances, carried onto every original point.

    Nearest point through a locator, which is O(n log n); the obvious double
    loop is O(n*m) over millions of points on each side.
    """
    import numpy as np
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

    values = decimated.GetPointData().GetArray(DISTANCE_ARRAY)
    if values is None:
        logger.warning("VFACE: the decimated surface carries no distance to carry back")
        return original

    values = vtk_to_numpy(values)
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(decimated)
    locator.BuildLocator()

    carried = np.array([
        values[locator.FindClosestPoint(original.GetPoint(index))]
        for index in range(original.GetNumberOfPoints())
    ])
    array = numpy_to_vtk(carried, deep=True)
    array.SetName(DISTANCE_ARRAY)
    original.GetPointData().AddArray(array)
    original.GetPointData().SetActiveScalars(DISTANCE_ARRAY)
    return original


def distance_map(baseline_path: str, compared_path: str, destination: str,
                 signed: bool = True) -> str:
    """One heat map: the baseline surface, carrying its distance to the other."""
    baseline = _clean(read_surface(baseline_path))
    compared = _clean(read_surface(compared_path))

    total = baseline.GetNumberOfPoints() + compared.GetNumberOfPoints()
    if total > HEAVY_SUBSAMPLE_ABOVE:
        reduction = HEAVY_REDUCTION
    elif total > SUBSAMPLE_ABOVE:
        reduction = MODERATE_REDUCTION
    else:
        reduction = 0.0

    if reduction:
        logger.info("VFACE: %d points across the pair, decimating by %.0f%% before "
                    "the distance", total, 100 * reduction)
        drawn = _interpolate_back(
            baseline,
            _distance(_decimated(baseline, reduction), _decimated(compared, reduction), signed),
        )
    else:
        drawn = _distance(baseline, compared, signed)

    return write_surface(drawn, destination)


def _surfaces_by_patient(root: str) -> dict:
    """`{patient: path}` for every surface under `root`, first one kept."""
    found = {}
    for directory, subdirectories, names in os.walk(root or ""):
        subdirectories.sort()
        for name in sorted(names):
            if name.startswith(".") or not name.lower().endswith(SURFACE_EXTENSIONS):
                continue
            found.setdefault(landmark_files.patient_of(name),
                             os.path.join(directory, name))
    return found


def draw_cohort(sup, oriented: dict, registered: dict, regions, surface_model: str,
                output_dir: str, work_dir: str, report: dict, span=None) -> str:
    """A heat map per patient per region, written under `Heat maps/<region>/`.

    Both sides are segmented rather than one: the baseline's surface comes from
    the oriented scan and the compared one from the registered scan, and they
    have to be the same kind of surface or the distance between them is the
    difference between two segmenters rather than between two anatomies.
    """
    destination = os.path.join(output_dir, "Heat maps")
    drawn = 0
    failures = []
    attempted = 0
    unpaired_total = 0
    # Two segmentations per region, each in its own slice of `span` and in the
    # order they are made. The distance maps between them are seconds against
    # the minutes of a segmentation, so they take no slice of their own.
    slices = iter(tools.split_span(span, [1] * (2 * len(regions))))

    for region in regions:
        frame = catalogs.REGION_TABLE[region]["frame"]
        baseline = _surfaces_by_patient(tools.segment_surfaces(
            sup, oriented[frame], surface_model, label=f"t1-{frame}", span=next(slices),
        ))
        compared = _surfaces_by_patient(tools.segment_surfaces(
            sup, registered[region], surface_model, label=f"t2-{region}",
            span=next(slices),
        ))

        paired = sorted(set(baseline) & set(compared))
        for index, patient in enumerate(paired, start=1):
            attempted += 1
            try:
                distance_map(
                    baseline[patient], compared[patient],
                    os.path.join(destination, region, f"{patient}_{region}_heatmap.vtk"),
                )
                drawn += 1
            except Exception as exc:  # noqa: BLE001 - one pair must not cost the rest
                logger.warning("VFACE: %s, patient %d of %d: heat map failed (%s)",
                               region, index, len(paired), describe_failure(exc))
                failures.append(exc)
                report.setdefault("heat_maps_failed", {})[f"{patient}/{region}"] = (
                    f"{type(exc).__name__}: {exc}"
                )

        unpaired = sorted(set(baseline) ^ set(compared))
        if unpaired:
            unpaired_total += len(unpaired)
            report.setdefault("heat_maps_unpaired", {})[region] = len(unpaired)

    if not drawn:
        # Counted on what was DRAWN. A guard counting the patients walked past
        # would pass on a cohort where not one map could be made.
        # Said in the error itself: the run report that holds the per-pair
        # detail is deleted with the job when the run fails. Not the caller's
        # 422 either -- every surface here came out of this run's own
        # segmentations.
        if not attempted:
            raise RuntimeError(
                f"0 heat map(s) drawn: no patient has a surface on both sides of "
                f"any region ({unpaired_total} surface(s) without a counterpart)"
            )
        raise RuntimeError(
            f"0 of {attempted} heat map(s) drawn; most common failure: "
            f"{most_common_failure(failures)}"
        )
    report["heat_maps"] = drawn
    return destination
