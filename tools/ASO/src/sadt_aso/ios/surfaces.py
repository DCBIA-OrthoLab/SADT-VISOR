"""Reading, writing and transforming intra-oral surface meshes.

Ported from `ASO_IOS/ASO_IOS_utils/utils.py` (ReadSurf/WriteSurf),
`transformation.py` (TransformSurf) and `OFFReader.py`.

Two changes worth knowing about:

* meshes are written BINARY. `vtkPolyDataWriter` defaults to ASCII, and the
  difference is not cosmetic: binary is smaller, ~100x faster to parse, and the
  MORE accurate of the two, round-tripping float32 exactly where ASCII prints
  six significant digits and moves points on read-back.
* the .off reader works. The original referenced an undefined `line` variable
  on the single-vertex branch, so any .off file containing one raised
  NameError.
"""

import os

import numpy as np

from ..errors import ToolInputError
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from ..markups import LABEL_ARRAY_NAMES

SURFACE_EXTENSIONS = (".vtk", ".vtp", ".stl", ".obj", ".off")


class SurfaceError(Exception):
    """A mesh could not be read, or does not carry what the mode needs."""


def is_surface_file(filename: str) -> bool:
    return filename.lower().endswith(SURFACE_EXTENSIONS)


def read_surface(path: str) -> vtk.vtkPolyData:
    extension = os.path.splitext(path)[1].lower()
    if extension == ".vtk":
        reader = vtk.vtkPolyDataReader()
        reader.ReadAllScalarsOn()
        reader.ReadAllVectorsOn()
        reader.ReadAllFieldsOn()
    elif extension == ".vtp":
        reader = vtk.vtkXMLPolyDataReader()
    elif extension == ".stl":
        reader = vtk.vtkSTLReader()
    elif extension == ".obj":
        reader = vtk.vtkOBJReader()
    elif extension == ".off":
        return _read_off(path)
    else:
        raise SurfaceError(
            f"'{os.path.basename(path)}': unsupported surface format. "
            f"Expected one of {', '.join(SURFACE_EXTENSIONS)}."
        )

    reader.SetFileName(path)
    reader.Update()
    surface = reader.GetOutput()
    if surface is None or surface.GetNumberOfPoints() == 0:
        raise SurfaceError(f"'{os.path.basename(path)}' holds no geometry.")
    return surface


def write_surface(surface: vtk.vtkPolyData, path: str) -> str:
    """Write a mesh, keeping its per-point arrays. Returns the path written."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if path.lower().endswith(".vtp"):
        writer = vtk.vtkXMLPolyDataWriter()
    else:
        writer = vtk.vtkPolyDataWriter()
        # Slicer reads a .vtk without a SPACE= header as LPS and warns about it
        # on every load; saying LPS keeps exactly that reading, without the warning.
        writer.SetHeader("SPACE=LPS")
        writer.SetFileTypeToBinary()
    writer.SetFileName(path)
    writer.SetInputData(surface)
    writer.Update()
    return path


def output_extension(input_path: str) -> str:
    """The format a result mesh is written in.

    `.vtp` in gives `.vtp` out; everything else becomes `.vtk`. The label array
    an intra-oral scan carries is the whole point of the fully-automated mode
    and `.stl`/`.off` cannot hold it, so those are never written back.
    """
    return ".vtp" if input_path.lower().endswith(".vtp") else ".vtk"


def label_array_name(surface: vtk.vtkPolyData) -> str:
    """The per-point tooth-label array this mesh carries, or None.

    Slicer's segmentation tools have used three names over time and all three
    are in the wild, so all three are accepted -- in preference order, since a
    mesh can carry more than one.
    """
    point_data = surface.GetPointData()
    present = {
        point_data.GetArrayName(index)
        for index in range(point_data.GetNumberOfArrays())
    }
    for name in LABEL_ARRAY_NAMES:
        if name in present:
            return name
    return None


def transform_surface(surface: vtk.vtkPolyData, matrix: np.ndarray) -> vtk.vtkPolyData:
    """Apply a 4x4 matrix to a mesh, returning a new one."""
    copy = vtk.vtkPolyData()
    copy.DeepCopy(surface)

    transform = vtk.vtkTransform()
    transform.SetMatrix(np.asarray(matrix, dtype=np.float64).reshape(16).tolist())

    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetTransform(transform)
    transform_filter.SetInputData(copy)
    transform_filter.Update()
    return transform_filter.GetOutput()


def points_of(surface: vtk.vtkPolyData) -> np.ndarray:
    return vtk_to_numpy(surface.GetPoints().GetData())


def labels_of(surface: vtk.vtkPolyData, array_name: str) -> np.ndarray:
    """The per-point tooth labels, whatever the producer named the array.

    `array_name` is tried first, so a caller that names one still decides, then
    markups.LABEL_ARRAY_NAMES -- the table this module already imported and
    this function was alone in not using. It matters here and nowhere else:
    the scan carries what the crown-segmentation tool wrote (`Universal_ID`)
    and the shipped gold reference carries `PredictedID`, and orienting one
    onto the other reads both in the same call. Looking only for the caller's
    name made the published reference unusable, with an AttributeError three
    frames deep inside vtk naming neither the file nor the array.
    """
    point_data = surface.GetPointData()
    for name in (array_name,) + LABEL_ARRAY_NAMES:
        if not name:
            continue
        array = point_data.GetScalars(name)
        if array is not None:
            return vtk_to_numpy(array)
    present = [point_data.GetArrayName(i) for i in range(point_data.GetNumberOfArrays())]
    raise ToolInputError(
        f"This surface carries no tooth-label array: looked for {array_name!r} and "
        f"{list(LABEL_ARRAY_NAMES)}, and it has {present}. A mesh has to be crown-"
        f"segmented before it can be oriented from its teeth."
    )


def _read_off(path: str) -> vtk.vtkPolyData:
    with open(path) as handle:
        if handle.readline().strip() != "OFF":
            raise SurfaceError(f"'{os.path.basename(path)}' is not a valid OFF file.")
        counts = handle.readline().split()
        vertex_count, face_count = int(counts[0]), int(counts[1])

        points = vtk.vtkPoints()
        for _ in range(vertex_count):
            values = [float(value) for value in handle.readline().split()]
            points.InsertNextPoint(values[0], values[1], values[2])

        cells = vtk.vtkCellArray()
        for _ in range(face_count):
            values = [int(value) for value in handle.readline().split()]
            if values[0] == 1:
                cell = vtk.vtkVertex()
                cell.GetPointIds().SetId(0, values[1])
            elif values[0] == 2:
                cell = vtk.vtkLine()
                cell.GetPointIds().SetId(0, values[1])
                cell.GetPointIds().SetId(1, values[2])
            elif values[0] == 3:
                cell = vtk.vtkTriangle()
                for index in range(3):
                    cell.GetPointIds().SetId(index, values[index + 1])
            else:
                continue
            cells.InsertNextCell(cell)

    surface = vtk.vtkPolyData()
    surface.SetPoints(points)
    surface.SetPolys(cells)
    return surface
