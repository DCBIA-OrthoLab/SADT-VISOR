"""Every legacy .vtk a tool writes says which coordinate system it is in.

3D Slicer reads a .vtk without a `SPACE=` header line as LPS and logs a
warning about it on every load -- one red line per mesh, in the console of the
person reviewing the result. Every tool's meshes are already read as LPS, so
the header changes nothing about where they land; it only stops the warning.
This test makes a new writer say so too.
"""

import pathlib
import re

TOOLS = pathlib.Path(__file__).resolve().parents[2] / "tools"
WRITER = re.compile(r"vtkPolyDataWriter\(\)")
HEADER = 'SetHeader("SPACE=LPS")'
# How far below the writer's creation its header may be set.
WINDOW = 6


def _sources():
    for path in sorted(TOOLS.glob("**/src/**/*.py")):
        if ".venv" in path.parts:
            continue
        yield path


def test_every_polydata_writer_sets_the_lps_header():
    missing = []
    for path in _sources():
        lines = path.read_text(encoding="utf-8").splitlines()
        for number, line in enumerate(lines):
            if WRITER.search(line) and not any(
                    HEADER in later for later in lines[number:number + WINDOW]):
                missing.append(f"{path.relative_to(TOOLS)}:{number + 1}")
    assert not missing, (
        "These vtkPolyDataWriter calls write no SPACE=LPS header, so Slicer "
        "warns on every load: " + ", ".join(missing))


def test_no_mesh_is_saved_through_pyvista():
    """`mesh.save` writes a .vtk with no way to set its header, which is how
    AREG IOS-CBCT's registered meshes still warned once the writers were fixed."""
    offenders = []
    for path in _sources():
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"\b(mesh|surface|polydata|pv_mesh)\.save\(", line):
                offenders.append(f"{path.relative_to(TOOLS)}:{number}")
    assert not offenders, "Write these through vtkPolyDataWriter: " + ", ".join(offenders)


def test_the_check_sees_the_writers_it_is_meant_to():
    """A glob that matched nothing would pass the test above vacuously."""
    found = sum(len(WRITER.findall(path.read_text(encoding="utf-8"))) for path in _sources())
    assert found >= 8
