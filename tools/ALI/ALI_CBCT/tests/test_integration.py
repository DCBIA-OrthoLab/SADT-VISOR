"""ALI as the server actually invokes it: out of process, in its own venv.

Two things are only testable from out here:

* that the tool writes nothing outside `output_dir`. In-process the test shares
  a working directory with the tool, so a stray write lands somewhere the test
  was going to create anyway; `run_tool` runs it from a neutral directory.
* that the published schema and the callable agree. The server reads the schema
  and calls `run(**params)` from it, so an argument renamed in one place and not
  the other breaks the chain there, not here -- unless something checks.

The `Crown_Seg -> ALI` chain below is the real handoff, not a fixture standing
in for one: ALI's IOS engine needs meshes carrying tooth labels, and Crown_Seg
is what puts them there. It used to happen inside ALI, through an in-process
import; the server sequences the two now, and this is what says the seam still
fits. It skips unless both tools are built with their heavy extras -- see
tests/data/README.md.
"""

import sys

import numpy as np
import pytest
import SimpleITK as sitk
from sadt_testkit import ToolFailed, is_built, run_tool, tool_schema

# The engine, not the facade: `tools/ALI` is a folder holding two of them and
# has no venv of its own. Gated on "ALI", every test here skipped from the day
# the tool was split -- including the one below that checks the published schema
# still matches run()'s signature, which is what the server dispatches against.
TOOL = "ALI_CBCT"

needs_venv = pytest.mark.skipif(
    not is_built(TOOL),
    reason=f"run `cd tools/{TOOL} && uv sync` first",
)


@pytest.fixture
def cohort(tmp_path):
    """One synthetic CBCT in a folder, which is how a batch arrives."""
    folder = tmp_path / "cohort"
    folder.mkdir()
    array = np.zeros((16, 16, 16), dtype=np.int16)
    array[4:12, 4:12, 4:12] = 800
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((0.5, 0.5, 0.5))
    sitk.WriteImage(image, str(folder / "patient01.nii.gz"))
    return folder


@needs_venv
def test_the_published_schema_matches_the_callable(cohort, tmp_path):
    """The server drives `run()` from this schema, so it has to be true."""
    schema = tool_schema(TOOL)

    # Everything below was written before ALI was split into two engines, and
    # this test could not run from that day until 2026-09-14 -- `is_built("ALI")`
    # answered False whatever was built, so it skipped instead of failing. What
    # it asserted had drifted three ways in the meantime, and none of the three
    # was caught: the tool's NAME, the required set, and the argument names.
    assert schema["name"] == "ALI_CBCT"
    assert schema["returns"] == "path"

    # `model` is optional now, and deliberately: a neighbour asking through the
    # supervisor omits it, and this engine resolves its own bundle. Required, a
    # supervised call died on "run() missing 1 required positional argument".
    required = [name for name, spec in schema["arguments"].items() if spec["required"]]
    assert required == ["input", "output_dir"]

    # `regions`, not `cbct_regions`: this engine IS the CBCT one, so the prefix
    # that told the two apart inside one tool says nothing here. `ios_networks`
    # belongs to the other engine and is not published by this one at all.
    for name in ("model", "regions", "landmarks"):
        assert not schema["arguments"][name]["required"]
    assert "ios_networks" not in schema["arguments"]

    # And the options the schema publishes are options the tool really takes.
    assert "Cranial base" in schema["arguments"]["regions"]["choices"]
    assert "Ba" in schema["arguments"]["landmarks"]["choices"]
    assert schema["arguments"]["device"]["choices"] == (
        ["cuda", "cpu", "mps"] if sys.platform == "darwin" else ["cuda", "cpu"]
    )


@needs_venv
def test_a_failure_comes_back_named(cohort, tmp_path):
    """A chain that breaks has to say which link broke, and why.

    An empty bundle is the cheap way to reach the error path out of process:
    no weights, no GPU, and the message is the one a client would be shown.
    """
    empty_bundle = tmp_path / "bundle"
    empty_bundle.mkdir()

    with pytest.raises(ToolFailed, match="No CBCT landmark weights"):
        run_tool(TOOL, input=cohort, model=empty_bundle, output_dir=tmp_path / "out")


@needs_venv
def test_a_mixed_input_is_refused_out_of_process_too(tmp_path):
    """The mode-detection refusal, through the real invocation path."""
    folder = tmp_path / "mixed"
    folder.mkdir()
    array = np.zeros((8, 8, 8), dtype=np.int16)
    sitk.WriteImage(sitk.GetImageFromArray(array), str(folder / "scan.nii.gz"))
    (folder / "arch.stl").write_bytes(b"solid empty\nendsolid empty\n")

    with pytest.raises(ToolFailed, match="mixes"):
        run_tool(TOOL, input=folder, model=tmp_path, output_dir=tmp_path / "out")


# ---------------------------------------------------------------------------
# The Crown_Seg -> ALI chain
# ---------------------------------------------------------------------------

CROWNSEG_MODEL = "tests/data/crownseg_checkpoint.pth"

needs_chain = pytest.mark.skipif(
    not (is_built(TOOL) and is_built("Crown_Seg")),
    reason="run `uv sync` in tools/ALI and tools/Crown_Seg",
)


@needs_chain
@pytest.mark.models
@pytest.mark.gpu
def test_crownseg_output_feeds_straight_into_ali(tmp_path):
    """The handoff `ALILogic.ensure_segmented()` used to make in-process.

    Crown_Seg writes `segmented_meshes` into its run report and ALI consumes
    the same directory. Nothing is imported across the two: each runs in its
    own venv, with its own torch, exactly as the server invokes them.
    """
    from pathlib import Path

    raw = Path("tests/data/raw_arch.vtk").resolve()
    checkpoint = Path(CROWNSEG_MODEL).resolve()
    if not (raw.is_file() and checkpoint.is_file()):
        pytest.skip("see tests/data/README.md for the fixtures this needs")

    segmented = run_tool(
        "Crown_Seg", meshes=raw, model=checkpoint, output_dir=tmp_path / "segmented"
    )

    landmarks = run_tool(
        TOOL,
        input=segmented,
        model=Path("tests/data/ALI_IOS_Models").resolve(),
        output_dir=tmp_path / "landmarks",
        ios_networks=["Occlusal"],
    )

    assert sorted(landmarks.rglob("*.mrk.json"))


@needs_chain
@pytest.mark.ios
def test_ali_refuses_an_unsegmented_mesh_and_names_crown_seg(tmp_path):
    """The other half of the chain: what happens when it is skipped.

    No weights and no GPU -- but it DOES need pytorch3d, and that is not an
    oversight in the test. `predict_landmarks` checks the engine's imports
    before it looks at a single mesh, deliberately: a missing dependency belongs
    to the venv, not to a patient's data, and reporting it per mesh would bury
    it behind "produced no landmarks for any mesh". So on a venv synced without
    the `ios` extra the honest answer really is "no pytorch3d", not "run
    Crown_Seg first" -- fixing the labels would not help.

    The message is still the whole point, because once the engine CAN run it is
    what tells whoever sent the request which tool to run first.
    """
    pytest.importorskip("pytorch3d", reason="needs `uv sync --extra ios`")
    vtk = pytest.importorskip("vtk")

    folder = tmp_path / "raw"
    folder.mkdir()
    points = vtk.vtkPoints()
    for coordinates in ((0, 0, 0), (1, 0, 0), (0, 1, 0)):
        points.InsertNextPoint(*coordinates)
    polys = vtk.vtkCellArray()
    polys.InsertNextCell(3)
    for point_id in range(3):
        polys.InsertCellPoint(point_id)
    surface = vtk.vtkPolyData()
    surface.SetPoints(points)
    surface.SetPolys(polys)
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(str(folder / "arch.vtk"))
    writer.SetInputData(surface)
    writer.Write()

    with pytest.raises(ToolFailed, match="Crown_Seg"):
        run_tool(TOOL, input=folder, model=tmp_path, output_dir=tmp_path / "out")
