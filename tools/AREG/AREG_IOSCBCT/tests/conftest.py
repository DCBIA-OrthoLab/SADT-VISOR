"""Fixtures for AREG_IOSCBCT's suite: two modalities, their landmarks, and a
supervisor that is not there.

This tool predicts nothing, so nothing here needs stubbing except the four
tools it drives -- and those are stood in for by `FakeSup`, which is all a tool
can ever see of a supervisor: `run`, `progress`, `log`, `tmp`, `out`. Nothing
is imported across virtualenvs, here or in the server.

The CBCT volumes are named, never read: `discover` pairs on the file name and
`register` only reports the base name, so a byte of content would be a byte of
fiction. The MESHES are real vtk polydata, because they are read, transformed
and written back.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import vtk


# ---------------------------------------------------------------------------
# The two modalities
# ---------------------------------------------------------------------------

def write_mesh(path, points=None):
    """A small triangulated mesh at known coordinates."""
    if points is None:
        points = [
            (0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (0.0, 10.0, 0.0),
            (0.0, 0.0, 10.0), (10.0, 10.0, 0.0), (5.0, 5.0, 5.0),
        ]
    vtk_points = vtk.vtkPoints()
    for point in points:
        vtk_points.InsertNextPoint(*point)

    polys = vtk.vtkCellArray()
    for index in range(len(points) - 2):
        polys.InsertNextCell(3)
        for point_id in (index, index + 1, index + 2):
            polys.InsertCellPoint(point_id)

    mesh = vtk.vtkPolyData()
    mesh.SetPoints(vtk_points)
    mesh.SetPolys(polys)

    os.makedirs(os.path.dirname(str(path)) or ".", exist_ok=True)
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(mesh)
    writer.Write()
    return str(path)


def write_volume(path):
    """A CBCT by name only, whose content no reader can open.

    Most tests here are about the bookkeeping around the registration -- which
    landmark file belongs to which mesh, what the report says, how a failure is
    contained -- and for those the volume is a name. It is NOT inert any more,
    though: `register` contours it to refine on, so a cohort built with this one
    exercises the path where the CBCT cannot be read and the run falls back to
    the landmarks alone. `write_contourable_volume` is the other half.
    """
    os.makedirs(os.path.dirname(str(path)) or ".", exist_ok=True)
    with open(str(path), "wb") as handle:
        handle.write(b"\x1f\x8b not a volume")
    return str(path)


def write_plane(path, centre=(5.0, 5.0, 0.0), size=10.0, resolution=6):
    """A flat, consistently wound patch of surface with usable normals.

    `write_mesh` fans triangles around consecutive points, which on a flat set
    winds neighbouring triangles against each other: the shared points then
    average to a ZERO normal, and anything measuring along normals rejects them.
    A test about the point-to-plane step needs a surface whose normals mean
    something, or it pins that step's behaviour on degenerate input.
    """
    import pyvista as pv

    patch = pv.Plane(center=centre, direction=(0.0, 0.0, 1.0),
                     i_size=size, j_size=size,
                     i_resolution=resolution, j_resolution=resolution)
    os.makedirs(os.path.dirname(str(path)) or ".", exist_ok=True)
    patch.triangulate().save(str(path), binary=True)
    return str(path)


def write_contourable_volume(path, box=((3.0, -2.0, 1.0), (13.0, 8.0, 11.0)),
                             bright=1000.0, dark=-1000.0, margin=10.0):
    """A CBCT holding one bright box, so contouring it gives a real surface.

    The default box is where `write_mesh`'s arch lands once the landmark fit has
    applied `moved`'s translation: five of that mesh's six points then sit on the
    box's faces, which is what gives the ICP something to match and lets a test
    assert it ran rather than only that it was called.

    Two intensities and nothing between them, so the level `volume.read` works
    out lands halfway: the landmarks read `bright` at their brightest, the band
    it calls soft tissue is empty, and the surround falls back to `dark`.
    """
    import numpy as np
    import SimpleITK as sitk

    low = np.asarray(box[0], dtype=float) - margin
    high = np.asarray(box[1], dtype=float) + margin
    size = np.ceil(high - low).astype(int) + 1

    # (k, j, i): SimpleITK indexes an array by z, y, x, the reverse of the
    # coordinate order `low` and `size` are written in.
    array = np.full(size[::-1], dark, dtype=np.float32)
    start = np.floor(np.asarray(box[0], dtype=float) - low).astype(int)
    stop = np.ceil(np.asarray(box[1], dtype=float) - low).astype(int) + 1
    array[start[2]:stop[2], start[1]:stop[1], start[0]:stop[0]] = bright

    image = sitk.GetImageFromArray(array)
    image.SetSpacing((1.0, 1.0, 1.0))
    image.SetOrigin(tuple(float(value) for value in low))
    os.makedirs(os.path.dirname(str(path)) or ".", exist_ok=True)
    sitk.WriteImage(image, str(path))
    return str(path)


def read_mesh_points(path):
    from vtk.util.numpy_support import vtk_to_numpy

    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    return vtk_to_numpy(reader.GetOutput().GetPoints().GetData())


# ---------------------------------------------------------------------------
# Landmarks
# ---------------------------------------------------------------------------

# Twelve occlusal crown points, three per quadrant. The set the cross-modality
# alignment really uses -- a crown tip is a crown tip in either modality.
UPPER_LABELS = {
    "UR1O": [0.0, 0.0, 0.0],
    "UR3O": [10.0, 0.0, 0.0],
    "UR6O": [20.0, 4.0, 0.0],
    "UL1O": [0.0, 10.0, 0.0],
    "UL3O": [10.0, 10.0, 2.0],
    "UL6O": [20.0, 14.0, 1.0],
}
LOWER_LABELS = {
    "LR1O": [0.0, 0.0, -30.0],
    "LR3O": [10.0, 0.0, -30.0],
    "LR6O": [20.0, 4.0, -30.0],
    "LL1O": [0.0, 10.0, -30.0],
}


def moved(landmarks, shift=(3.0, -2.0, 1.0)):
    """The same labels, rigidly displaced -- the CBCT's view of the same crowns."""
    return {
        label: [value + delta for value, delta in zip(position, shift)]
        for label, position in landmarks.items()
    }


def write_markups(path, landmarks):
    """A Slicer markups file, the shape ALI writes."""
    content = {
        "@schema": "https://example.invalid/markups-schema-v1.0.0.json#",
        "markups": [
            {
                "type": "Fiducial",
                "coordinateSystem": "LPS",
                "controlPoints": [
                    {"id": str(index), "label": label, "position": list(position)}
                    for index, (label, position) in enumerate(landmarks.items(), start=1)
                ],
            }
        ],
    }
    os.makedirs(os.path.dirname(str(path)) or ".", exist_ok=True)
    with open(str(path), "w", encoding="utf-8") as handle:
        json.dump(content, handle)
    return str(path)


def write_flat_landmarks(path, landmarks):
    """`{label: [x, y, z]}` at the top level -- upstream's intraoral spelling."""
    os.makedirs(os.path.dirname(str(path)) or ".", exist_ok=True)
    with open(str(path), "w", encoding="utf-8") as handle:
        json.dump({label: list(position) for label, position in landmarks.items()}, handle)
    return str(path)


# ---------------------------------------------------------------------------
# One patient, complete
# ---------------------------------------------------------------------------

def cohort(tmp_path, patients=("P001",), jaws=("U",), landmarks=None):
    """A Registration-mode request: meshes, volumes, and both landmark sets.

    Named the way upstream's own test set is -- `P001_T2_U.vtk` beside
    `P_0001_T2.nii.gz`, both T2 -- because that difference in convention is
    what `patient_key` exists to bridge.
    """
    landmarks = landmarks or UPPER_LABELS
    for patient in patients:
        digits = patient.lstrip("P")
        write_volume(tmp_path / "cbct" / f"P_0{digits}_T2.nii.gz")
        write_markups(
            tmp_path / "cbct_lm" / f"P_0{digits}_T2_lm_Pred.mrk.json", moved(landmarks)
        )
        for jaw in jaws:
            write_mesh(tmp_path / "ios" / f"{patient}_T2_{jaw}.vtk")
            write_markups(
                tmp_path / "ios_lm" / f"{patient}_T2_{jaw}_lm_Pred.mrk.json", landmarks
            )
    return tmp_path


def run_registration(tmp_path, **overrides):
    """`dispatch.main` in Registration mode over the cohort above."""
    from sadt_areg_ioscbct import dispatch

    arguments = {
        "ios": str(tmp_path / "ios"),
        "cbct": str(tmp_path / "cbct"),
        "output_dir": str(tmp_path / "out"),
        "automation": "Registration",
        "ios_landmarks": str(tmp_path / "ios_lm"),
        "cbct_landmarks": str(tmp_path / "cbct_lm"),
    }
    arguments.update(overrides)
    return dispatch.main(**arguments)


def read_report(output_dir):
    from sadt_areg_ioscbct.dispatch import REPORT_NAME

    with open(os.path.join(str(output_dir), REPORT_NAME), encoding="utf-8") as handle:
        return json.load(handle)


def tree_of(root):
    return sorted(
        os.path.relpath(os.path.join(directory, name), str(root))
        for directory, _subdirs, names in os.walk(str(root))
        for name in names
    )


# ---------------------------------------------------------------------------
# The supervisor
# ---------------------------------------------------------------------------

class FakeSup:
    """A supervisor, as a tool sees one. Records what it was asked for.

    `outputs` maps a tool name to a callable taking the parameters it was sent
    and returning the directory it "produced", so a test can plant results
    without any of the real tools existing.
    """

    def __init__(self, tmp_path, outputs=None):
        self.out = Path(tmp_path) / "out"
        self.tmp = Path(tmp_path) / "tmp"
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.outputs = outputs or {}
        self.calls = []
        self.spans = []
        self.messages = []
        self.logs = []

    def run(self, tool, **params):
        # The caller's span of its own bar. The server's supervisor removes it
        # before the callee sees it, so it is recorded apart, never passed on.
        self.spans.append((tool, params.pop("_progress", None)))
        self.calls.append((tool, params))
        maker = self.outputs.get(tool)
        if maker is None:
            raise AssertionError(f"nothing planted for {tool!r} in this test")
        produced = maker(params)
        if isinstance(produced, Exception):
            raise produced
        return Path(produced)

    def progress(self, fraction, message):
        self.messages.append((fraction, message))

    def log(self, message, level="info", user=False):
        self.messages.append((None, message))
        self.logs.append((level, user, message))

    def asked(self, tool):
        """The parameters of the one call to `tool`."""
        matching = [params for name, params in self.calls if name == tool]
        assert len(matching) == 1, f"{tool} was called {len(matching)} time(s)"
        return matching[0]


@pytest.fixture
def rigid():
    """A known rigid motion, as a 4x4 and as a callable."""
    angle = 0.21
    rotation = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    translation = np.array([4.0, -3.0, 2.5])
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix
