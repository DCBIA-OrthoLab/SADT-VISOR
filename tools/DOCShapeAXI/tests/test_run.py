"""End-to-end tests for DOCShapeAXI.

The network and the GradCAM pass are stubbed, so everything around them runs
for real: discovery, the checkpoint catalog, the report, the suffixing, and the
regression-versus-classification split. The network itself cannot run in CI --
it needs a CUDA device and 72 MB of weights per checkpoint -- which is what the
`gpu`/`models` markers are for.
"""

import json
import os

import pytest

from sadt_docshapeaxi import _rename_with_suffix, run
from sadt_docshapeaxi import catalog, engine, pipeline
from sadt_docshapeaxi.errors import ToolInputError, ToolUnavailableError


def write_surface(path):
    """A one-triangle `.vtk`, which is all discovery and naming need."""
    vtk = pytest.importorskip("vtk")

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

    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(surface)
    writer.Write()
    return str(path)


@pytest.fixture
def stubbed(monkeypatch):
    """Replace the three things that need a GPU and a checkpoint."""
    monkeypatch.setattr(pipeline, "resolve_device", lambda requested: "cpu")
    monkeypatch.setattr(pipeline, "load_network", lambda *a, **k: object())

    def predict(model, analysis, surfaces, mount_point, device, span=(0.0, 1.0)):
        return [float(index % max(analysis.classes, 1)) for index in range(len(surfaces))]

    def explain(model, analysis, surfaces, mount_point, device, output_dir,
                span=(0.0, 1.0)):
        written = []
        for path in surfaces:
            destination = os.path.join(output_dir, os.path.basename(path))
            write_surface(destination)
            written.append(destination)
        return written

    monkeypatch.setattr(engine, "predict", predict)
    monkeypatch.setattr(engine, "explain", explain)


def test_the_checkpoint_name_carries_the_whole_analysis():
    analysis = catalog.analysis_for("airways_4_regress.ckpt")
    assert analysis.is_regression
    assert analysis.classes == 1
    assert catalog.analysis_for("airways_2_class.ckpt").classes == 2


def test_an_unknown_checkpoint_names_the_ones_that_exist():
    with pytest.raises(ValueError) as raised:
        catalog.analysis_for("clefts_4_class.ckpt")
    # `clefts_4_class` is upstream's own spelling, and there is no such file.
    assert "cleft_4_class.ckpt" in str(raised.value)


def test_discovery_is_recursive_and_ignores_everything_but_vtk(tmp_path):
    write_surface(tmp_path / "a.vtk")
    write_surface(tmp_path / "nested" / "b.vtk")
    (tmp_path / "notes.txt").write_text("not a surface")
    (tmp_path / "scan.nii.gz").write_bytes(b"\x1f\x8b")

    found = pipeline.discover_surfaces(str(tmp_path))
    assert [os.path.basename(path) for path in found] == ["a.vtk", "b.vtk"]


def test_a_folder_with_no_surface_is_refused_before_anything_is_loaded(tmp_path):
    (tmp_path / "scan.nii.gz").write_bytes(b"\x1f\x8b")
    checkpoint = tmp_path / "airways_2_class.ckpt"
    checkpoint.write_bytes(b"")
    with pytest.raises(ValueError, match="No .vtk surface"):
        run(surfaces=str(tmp_path), model=str(checkpoint),
            output_dir=str(tmp_path / "out"))


def test_a_missing_checkpoint_is_reported_by_name(tmp_path):
    write_surface(tmp_path / "a.vtk")
    with pytest.raises(FileNotFoundError, match="airways_2_class.ckpt"):
        run(surfaces=str(tmp_path), model=str(tmp_path / "airways_2_class.ckpt"),
            output_dir=str(tmp_path / "out"))


def test_a_classification_run_reports_a_class_per_surface(tmp_path, stubbed):
    write_surface(tmp_path / "in" / "a.vtk")
    write_surface(tmp_path / "in" / "b.vtk")
    checkpoint = tmp_path / "condyles_4_class.ckpt"
    checkpoint.write_bytes(b"")

    result = run(surfaces=str(tmp_path / "in"), model=str(checkpoint),
                 output_dir=str(tmp_path / "out"))

    assert [entry["class"] for entry in result["predictions"]] == [0, 1]
    assert all("score" not in entry for entry in result["predictions"])


def test_a_regression_run_reports_a_value_not_class_zero(tmp_path, monkeypatch, stubbed):
    """Upstream took an argmax for every checkpoint. A regression network has
    one output column, so its argmax is 0 for every subject -- every patient
    graded identically, with no error anywhere."""
    write_surface(tmp_path / "in" / "a.vtk")
    checkpoint = tmp_path / "airways_4_regress.ckpt"
    checkpoint.write_bytes(b"")
    monkeypatch.setattr(engine, "predict", lambda *a, **k: [2.75])

    result = run(surfaces=str(tmp_path / "in"), model=str(checkpoint),
                 output_dir=str(tmp_path / "out"))

    assert result["predictions"] == [{"surface": "a.vtk", "score": 2.75}]


def test_every_surface_is_written_once_under_its_own_name(tmp_path, stubbed):
    """Upstream wrote inside the per-class loop, to a path that did not depend
    on the class, so a four-class run wrote the same file four times."""
    write_surface(tmp_path / "in" / "a.vtk")
    write_surface(tmp_path / "in" / "b.vtk")
    checkpoint = tmp_path / "cleft_4_class.ckpt"
    checkpoint.write_bytes(b"")

    run(surfaces=str(tmp_path / "in"), model=str(checkpoint),
        output_dir=str(tmp_path / "out"), output_suffix="_pred")

    written = sorted(os.listdir(tmp_path / "out"))
    assert written == ["DOCShapeAXI_report.json", "a_pred.vtk", "b_pred.vtk"]


def test_the_report_says_which_checkpoint_answered(tmp_path, stubbed):
    write_surface(tmp_path / "in" / "a.vtk")
    checkpoint = tmp_path / "airways_2_class.ckpt"
    checkpoint.write_bytes(b"")

    result = run(surfaces=str(tmp_path / "in"), model=str(checkpoint),
                 output_dir=str(tmp_path / "out"), explain=False)

    report = json.loads(open(result["report"], encoding="utf-8").read())
    assert report["model"] == "airways_2_class.ckpt"
    assert report["task"] == "binary"
    assert report["classes"] == 2
    assert report["explained"] is False
    assert report["surfaces"] == 1


def test_without_explain_no_surface_is_written(tmp_path, stubbed):
    write_surface(tmp_path / "in" / "a.vtk")
    checkpoint = tmp_path / "airways_2_class.ckpt"
    checkpoint.write_bytes(b"")

    run(surfaces=str(tmp_path / "in"), model=str(checkpoint),
        output_dir=str(tmp_path / "out"), explain=False)

    assert os.listdir(tmp_path / "out") == ["DOCShapeAXI_report.json"]


def test_a_single_file_is_accepted_as_well_as_a_folder(tmp_path, stubbed):
    surface = write_surface(tmp_path / "in" / "a.vtk")
    checkpoint = tmp_path / "airways_2_class.ckpt"
    checkpoint.write_bytes(b"")

    result = run(surfaces=surface, model=str(checkpoint),
                 output_dir=str(tmp_path / "out"))

    assert len(result["predictions"]) == 1


def test_an_empty_suffix_leaves_the_written_names_alone(tmp_path):
    written = [write_surface(tmp_path / "a.vtk")]
    assert _rename_with_suffix(written, "") == written


def test_a_flat_attribution_map_normalises_to_zeros_not_nan():
    """Upstream divided by `max - min` unconditionally. A network that
    attributes nothing produced a surface array of NaN, which VTK writes and
    Slicer colours as if it meant something."""
    numpy = pytest.importorskip("numpy")

    scaled = pipeline.scale_attribution(numpy.zeros((1, 8, 8), dtype="float32"))
    assert numpy.isfinite(scaled).all()
    assert not scaled.any()


def test_an_attribution_map_is_rescaled_into_minus_one_to_one():
    numpy = pytest.importorskip("numpy")

    ramp = numpy.linspace(-40, 90, 64, dtype="float32").reshape(1, 8, 8)
    scaled = pipeline.scale_attribution(ramp)
    assert scaled.shape == (1, 224, 224)
    assert -1.0001 <= scaled.min() and scaled.max() <= 1.0001


@pytest.mark.gpu
@pytest.mark.models
def test_real_model_classifies_a_real_surface(tmp_path):
    """Run by hand against the staged checkpoints; the result goes in the PR."""
    models = os.environ.get("DOCSHAPEAXI_MODELS")
    surfaces = os.environ.get("DOCSHAPEAXI_SURFACES")
    if not models or not surfaces:
        pytest.skip("set DOCSHAPEAXI_MODELS and DOCSHAPEAXI_SURFACES")

    result = run(surfaces=surfaces,
                 model=os.path.join(models, "condyles_4_class.ckpt"),
                 output_dir=str(tmp_path))
    assert result["predictions"]


# ---------------------------------------------------------------------------
# The catalog: one argument names the checkpoint, and everything else follows.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,classes,task,regression", [
    ("airways_2_class.ckpt", 2, "binary", False),
    ("airways_4_class.ckpt", 4, "severity", False),
    ("airways_4_regress.ckpt", 1, "regression", True),
    ("condyles_4_class.ckpt", 4, "severity", False),
    ("cleft_4_class.ckpt", 4, "severity", False),
])
def test_every_checkpoint_implies_its_own_task_and_class_count(name, classes, task, regression):
    analysis = catalog.analysis_for(name)
    assert (analysis.classes, analysis.task, analysis.is_regression) == (classes, task, regression)


def test_the_catalog_covers_exactly_the_five_published_checkpoints():
    assert set(catalog.ANALYSES) == {
        "airways_2_class.ckpt", "airways_4_class.ckpt", "airways_4_regress.ckpt",
        "condyles_4_class.ckpt", "cleft_4_class.ckpt",
    }


def test_each_row_is_keyed_by_its_own_checkpoint_name():
    for name, analysis in catalog.ANALYSES.items():
        assert analysis.checkpoint == name


def test_only_the_regression_row_uses_the_regression_network():
    regressors = [n for n, a in catalog.ANALYSES.items() if a.network == catalog.REGRESSION]
    assert regressors == ["airways_4_regress.ckpt"]


def test_the_regression_row_has_exactly_one_output():
    assert catalog.ANALYSES["airways_4_regress.ckpt"].classes == 1


def test_every_classification_row_has_at_least_two_classes():
    for name, analysis in catalog.ANALYSES.items():
        if not analysis.is_regression:
            assert analysis.classes >= 2, name


def test_the_two_airway_classifiers_grade_the_same_anatomy_differently():
    two = catalog.analysis_for("airways_2_class.ckpt")
    four = catalog.analysis_for("airways_4_class.ckpt")
    assert two.anatomy == four.anatomy
    assert two.classes != four.classes


def test_a_checkpoint_given_as_a_full_path_resolves_by_its_basename(tmp_path, stubbed):
    """A client sends a name; the server hands the tool an absolute path."""
    write_surface(tmp_path / "in" / "a.vtk")
    checkpoint = tmp_path / "models" / "condyles_4_class.ckpt"
    os.makedirs(checkpoint.parent, exist_ok=True)
    checkpoint.write_bytes(b"")

    result = run(surfaces=str(tmp_path / "in"), model=str(checkpoint),
                 output_dir=str(tmp_path / "out"), explain=False)
    assert json.loads(open(result["report"], encoding="utf-8").read())["classes"] == 4


def test_a_checkpoint_with_the_wrong_extension_is_refused():
    with pytest.raises(ValueError):
        catalog.analysis_for("condyles_4_class.pt")


def test_an_empty_checkpoint_name_is_refused():
    with pytest.raises(ValueError):
        catalog.analysis_for("")


def test_the_catalog_is_case_sensitive_because_the_file_system_is():
    with pytest.raises(ValueError):
        catalog.analysis_for("Condyles_4_Class.ckpt")


def test_an_unknown_checkpoint_lists_all_five_alternatives():
    with pytest.raises(ValueError) as raised:
        catalog.analysis_for("nose_9_class.ckpt")
    for known in catalog.ANALYSES:
        assert known in str(raised.value)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def test_discovery_returns_files_not_directories(tmp_path):
    write_surface(tmp_path / "a.vtk")
    for path in pipeline.discover_surfaces(str(tmp_path)):
        assert os.path.isfile(path)


def test_discovery_of_an_empty_folder_returns_nothing(tmp_path):
    os.makedirs(tmp_path / "empty")
    assert pipeline.discover_surfaces(str(tmp_path / "empty")) == []


def test_discovery_skips_dotfiles(tmp_path):
    write_surface(tmp_path / "a.vtk")
    write_surface(tmp_path / ".hidden.vtk")
    assert [os.path.basename(p) for p in pipeline.discover_surfaces(str(tmp_path))] == ["a.vtk"]


def test_discovery_accepts_an_uppercase_extension(tmp_path):
    write_surface(tmp_path / "a.vtk")
    os.replace(tmp_path / "a.vtk", tmp_path / "A.VTK")
    assert len(pipeline.discover_surfaces(str(tmp_path))) == 1


def test_discovery_is_sorted_so_a_run_is_reproducible(tmp_path):
    for name in ("c.vtk", "a.vtk", "b.vtk"):
        write_surface(tmp_path / name)
    found = pipeline.discover_surfaces(str(tmp_path))
    assert found == sorted(found)


def test_discovery_walks_more_than_one_level(tmp_path):
    write_surface(tmp_path / "one" / "two" / "three" / "deep.vtk")
    assert len(pipeline.discover_surfaces(str(tmp_path))) == 1


def test_discovery_of_a_single_file_that_is_not_a_surface_returns_nothing(tmp_path):
    scan = tmp_path / "scan.nii.gz"
    scan.write_bytes(b"\x1f\x8b")
    assert pipeline.discover_surfaces(str(scan)) == []


def test_discovery_of_a_path_that_does_not_exist_returns_nothing(tmp_path):
    assert pipeline.discover_surfaces(str(tmp_path / "absent")) == []


@pytest.mark.parametrize("name", ["a.stl", "a.ply", "a.obj", "a.vtp", "a.nii.gz", "a.json"])
def test_a_surface_format_the_reader_cannot_open_is_not_discovered(tmp_path, name):
    """Upstream globbed `.vtk` only, and so does this. Accepting a name the
    loader then refuses is the failure mode worth pinning."""
    (tmp_path / name).write_bytes(b"")
    assert pipeline.discover_surfaces(str(tmp_path)) == []


def test_is_surface_file_agrees_with_discovery():
    assert pipeline.is_surface_file("a.vtk")
    assert pipeline.is_surface_file("A.VTK")
    assert not pipeline.is_surface_file("a.vtk.gz")
    assert not pipeline.is_surface_file("vtk")


# ---------------------------------------------------------------------------
# The attribution rescaling
# ---------------------------------------------------------------------------

def reference_bilinear(image, size):
    """Bilinear resize with HALF-PIXEL centres, written out longhand.

    This is the convention `cv2.resize(..., INTER_LINEAR)` uses and the one
    `align_corners=False` uses. It is the thing that could silently be wrong:
    `align_corners=True` is also bilinear, also plausible, and shifts every
    sample. Measured against a real OpenCV, the port agrees to 2.2e-5
    relative -- float32 rounding, not a different algorithm.
    """
    import numpy as np

    height, width = image.shape
    out_h, out_w = size
    rows = (np.arange(out_h) + 0.5) * height / out_h - 0.5
    cols = (np.arange(out_w) + 0.5) * width / out_w - 0.5
    rows = np.clip(rows, 0, height - 1)
    cols = np.clip(cols, 0, width - 1)

    r0 = np.floor(rows).astype(int); r1 = np.minimum(r0 + 1, height - 1)
    c0 = np.floor(cols).astype(int); c1 = np.minimum(c0 + 1, width - 1)
    dr = (rows - r0)[:, None]; dc = (cols - c0)[None, :]

    top = image[r0][:, c0] * (1 - dc) + image[r0][:, c1] * dc
    bottom = image[r1][:, c0] * (1 - dc) + image[r1][:, c1] * dc
    return top * (1 - dr) + bottom * dr


def test_the_resize_uses_half_pixel_centres_like_opencv():
    numpy = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")

    rng = numpy.random.default_rng(0)
    for shape in [(7, 7), (13, 17), (28, 28), (31, 29)]:
        image = rng.standard_normal(shape).astype("float32")
        got = torch.nn.functional.interpolate(
            torch.from_numpy(image)[None, None], size=(224, 224),
            mode="bilinear", align_corners=False)[0, 0].numpy()
        expected = reference_bilinear(image.astype("float64"), (224, 224))
        assert numpy.abs(got - expected).max() < 1e-4


def test_align_corners_true_would_be_a_different_answer():
    """Pins that the choice is a choice: the two conventions really do differ,
    so `align_corners=False` is load-bearing rather than decorative."""
    numpy = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")

    image = torch.arange(49, dtype=torch.float32).reshape(1, 1, 7, 7)
    a = torch.nn.functional.interpolate(image, size=(32, 32), mode="bilinear", align_corners=False)
    b = torch.nn.functional.interpolate(image, size=(32, 32), mode="bilinear", align_corners=True)
    assert numpy.abs((a - b).numpy()).max() > 0.1


def test_a_two_dimensional_map_is_accepted_as_a_batch_of_one():
    numpy = pytest.importorskip("numpy")
    scaled = pipeline.scale_attribution(numpy.zeros((8, 8), dtype="float32"))
    assert scaled.shape == (1, 224, 224)


def test_a_batch_of_several_maps_is_scaled_independently():
    numpy = pytest.importorskip("numpy")
    batch = numpy.stack([
        numpy.zeros((8, 8), dtype="float32"),
        numpy.linspace(0, 1, 64, dtype="float32").reshape(8, 8),
    ])
    scaled = pipeline.scale_attribution(batch)
    assert not scaled[0].any()
    assert scaled[1].min() < 0 < scaled[1].max()


def test_the_output_is_float32_whatever_went_in():
    numpy = pytest.importorskip("numpy")
    scaled = pipeline.scale_attribution(numpy.zeros((1, 8, 8), dtype="float64"))
    assert scaled.dtype == numpy.float32


def test_a_single_hot_pixel_does_not_flatten_the_rest():
    """This is what clipping to the 1st and 99th percentiles buys: without it
    one outlier compresses every real value into a sliver around zero.

    The outlier's own corner is not preserved -- it is clipped, which is the
    point -- so the comparison is over the rest of the frame. Without the
    clip the whole map would collapse: that is what the second assertion
    measures.
    """
    numpy = pytest.importorskip("numpy")
    image = numpy.linspace(0, 1, 4096, dtype="float32").reshape(1, 64, 64)
    with_outlier = image.copy()
    with_outlier[0, 0, 0] = 1e6

    plain = pipeline.scale_attribution(image)[0]
    clipped = pipeline.scale_attribution(with_outlier)[0]
    rest = (slice(16, None), slice(16, None))
    assert numpy.abs(plain[rest] - clipped[rest]).max() < 0.05

    # Unclipped, the same outlier squeezes every real value to one end.
    unclipped = 2 * ((with_outlier - with_outlier.min())
                     / (with_outlier.max() - with_outlier.min())) - 1
    assert unclipped[0][rest].max() < -0.999


def test_an_all_negative_map_still_spans_the_full_range():
    numpy = pytest.importorskip("numpy")
    scaled = pipeline.scale_attribution(
        numpy.linspace(-90, -40, 64, dtype="float32").reshape(1, 8, 8))
    assert scaled.min() < -0.9 and scaled.max() > 0.9


def test_a_constant_nonzero_map_normalises_to_zeros():
    numpy = pytest.importorskip("numpy")
    scaled = pipeline.scale_attribution(numpy.full((1, 8, 8), 7.5, dtype="float32"))
    assert numpy.isfinite(scaled).all() and not scaled.any()


def test_a_map_containing_a_nan_does_not_silently_produce_a_surface_of_nans():
    """A NaN in, a NaN out is acceptable; a NaN in that becomes a plausible
    number would be a lie painted on the mesh."""
    numpy = pytest.importorskip("numpy")
    image = numpy.zeros((1, 8, 8), dtype="float32")
    image[0, 0, 0] = numpy.nan
    scaled = pipeline.scale_attribution(image)
    assert numpy.isnan(scaled).any() or not scaled.any()


def test_the_resize_target_is_the_renderers_frame():
    assert pipeline.GRADCAM_IMAGE_SIZE == (224, 224)


def test_the_clip_percentiles_are_the_ones_upstream_borrowed():
    assert (pipeline.CLIP_LOW, pipeline.CLIP_HIGH) == (1, 99)


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------

def test_cpu_is_never_upgraded_to_cuda(monkeypatch):
    assert pipeline.resolve_device("cpu") == "cpu"


def test_cuda_falls_back_to_cpu_when_no_device_is_visible(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert pipeline.resolve_device("cuda") == "cpu"


def test_cuda_is_kept_when_a_device_is_visible(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert pipeline.resolve_device("cuda") == "cuda"


def test_the_device_actually_used_is_the_one_recorded(tmp_path, stubbed):
    write_surface(tmp_path / "in" / "a.vtk")
    checkpoint = tmp_path / "airways_2_class.ckpt"
    checkpoint.write_bytes(b"")
    result = run(surfaces=str(tmp_path / "in"), model=str(checkpoint),
                 output_dir=str(tmp_path / "out"), explain=False, device="cuda")
    assert json.loads(open(result["report"], encoding="utf-8").read())["device"] == "cpu"


# ---------------------------------------------------------------------------
# Network resolution
# ---------------------------------------------------------------------------

def test_the_network_is_looked_up_in_saxi_nets_lightning_not_saxi_nets(monkeypatch):
    """`shapeaxi.saxi_predict` does `getattr(saxi_nets, args.nn)` and raises
    AttributeError before touching a mesh, because both these classes live in
    `saxi_nets_lightning`. That is why the prediction loop here is upstream's
    rather than shapeaxi's own."""
    import sys
    import types

    module = types.ModuleType("shapeaxi.saxi_nets_lightning")
    seen = {}

    class Fake:
        @staticmethod
        def load_from_checkpoint(path, strict=False):
            seen["path"] = path
            seen["strict"] = strict
            return _Recorder()

    class _Recorder:
        def eval(self):
            seen["eval"] = True

        def to(self, device):
            seen["device"] = device

    module.SaxiMHAFBClassification = Fake
    shapeaxi = types.ModuleType("shapeaxi")
    shapeaxi.saxi_nets_lightning = module
    monkeypatch.setitem(sys.modules, "shapeaxi", shapeaxi)
    monkeypatch.setitem(sys.modules, "shapeaxi.saxi_nets_lightning", module)
    # The subject here is WHERE the class is looked up, not the environment
    # checks that follow it.
    monkeypatch.setattr(pipeline, "check_backbone_is_staged", lambda: None)
    monkeypatch.setattr(pipeline, "allow_checkpoint_globals", lambda: None)

    pipeline.load_network("/models/x.ckpt", "SaxiMHAFBClassification", "cpu")
    assert seen == {"path": "/models/x.ckpt", "strict": False, "eval": True, "device": "cpu"}


def test_a_network_the_installed_shapeaxi_lacks_is_named_in_the_error(monkeypatch):
    import sys
    import types

    module = types.ModuleType("shapeaxi.saxi_nets_lightning")
    shapeaxi = types.ModuleType("shapeaxi")
    shapeaxi.saxi_nets_lightning = module
    monkeypatch.setitem(sys.modules, "shapeaxi", shapeaxi)
    monkeypatch.setitem(sys.modules, "shapeaxi.saxi_nets_lightning", module)

    # A RuntimeError: the installation disagreeing with itself is the
    # server's fault, and must not be answered as a bad request.
    with pytest.raises(RuntimeError, match="SaxiMHAFBRegression") as raised:
        pipeline.load_network("/models/x.ckpt", "SaxiMHAFBRegression", "cpu")
    assert not isinstance(raised.value, ValueError)


# ---------------------------------------------------------------------------
# run(): naming, the report, and the guards
# ---------------------------------------------------------------------------

def test_the_output_directory_is_created_by_the_tool(tmp_path, stubbed):
    write_surface(tmp_path / "in" / "a.vtk")
    checkpoint = tmp_path / "airways_2_class.ckpt"
    checkpoint.write_bytes(b"")
    destination = tmp_path / "deep" / "and" / "absent"

    run(surfaces=str(tmp_path / "in"), model=str(checkpoint),
        output_dir=str(destination), explain=False)
    assert destination.is_dir()


def test_two_surfaces_with_the_same_name_in_different_folders_do_not_collide(tmp_path, stubbed):
    """Keying on the base name is how a batch loses a patient. The prediction
    rows are keyed by the path relative to the input root."""
    write_surface(tmp_path / "in" / "p1" / "scan.vtk")
    write_surface(tmp_path / "in" / "p2" / "scan.vtk")
    checkpoint = tmp_path / "airways_2_class.ckpt"
    checkpoint.write_bytes(b"")

    result = run(surfaces=str(tmp_path / "in"), model=str(checkpoint),
                 output_dir=str(tmp_path / "out"), explain=False)
    keys = sorted(entry["surface"] for entry in result["predictions"])
    assert keys == [os.path.join("p1", "scan.vtk"), os.path.join("p2", "scan.vtk")]


def test_the_prediction_order_follows_the_discovery_order(tmp_path, stubbed):
    for name in ("c.vtk", "a.vtk", "b.vtk"):
        write_surface(tmp_path / "in" / name)
    checkpoint = tmp_path / "cleft_4_class.ckpt"
    checkpoint.write_bytes(b"")

    result = run(surfaces=str(tmp_path / "in"), model=str(checkpoint),
                 output_dir=str(tmp_path / "out"), explain=False)
    assert [e["surface"] for e in result["predictions"]] == ["a.vtk", "b.vtk", "c.vtk"]


def test_the_report_is_named_in_the_outputs_mapping(tmp_path, stubbed):
    write_surface(tmp_path / "in" / "a.vtk")
    checkpoint = tmp_path / "airways_2_class.ckpt"
    checkpoint.write_bytes(b"")

    result = run(surfaces=str(tmp_path / "in"), model=str(checkpoint),
                 output_dir=str(tmp_path / "out"), explain=False)
    assert result["outputs"]["report"] == result["report"]


def test_each_explained_surface_is_named_in_the_outputs_mapping(tmp_path, stubbed):
    write_surface(tmp_path / "in" / "a.vtk")
    write_surface(tmp_path / "in" / "b.vtk")
    checkpoint = tmp_path / "airways_2_class.ckpt"
    checkpoint.write_bytes(b"")

    result = run(surfaces=str(tmp_path / "in"), model=str(checkpoint),
                 output_dir=str(tmp_path / "out"))
    assert set(result["outputs"]) == {"report", "a_pred", "b_pred"}


def test_the_report_is_valid_json_a_client_can_read(tmp_path, stubbed):
    write_surface(tmp_path / "in" / "a.vtk")
    checkpoint = tmp_path / "condyles_4_class.ckpt"
    checkpoint.write_bytes(b"")

    result = run(surfaces=str(tmp_path / "in"), model=str(checkpoint),
                 output_dir=str(tmp_path / "out"), explain=False)
    report = json.loads(open(result["report"], encoding="utf-8").read())
    assert set(report) == {"model", "anatomy", "task", "classes", "network",
                           "device", "explained", "surfaces", "predictions"}


def test_the_report_names_the_anatomy_a_clinician_would_recognise(tmp_path, stubbed):
    write_surface(tmp_path / "in" / "a.vtk")
    checkpoint = tmp_path / "condyles_4_class.ckpt"
    checkpoint.write_bytes(b"")

    result = run(surfaces=str(tmp_path / "in"), model=str(checkpoint),
                 output_dir=str(tmp_path / "out"), explain=False)
    assert "condyle" in json.loads(open(result["report"], encoding="utf-8").read())["anatomy"].lower()


def test_the_surface_count_in_the_report_matches_the_predictions(tmp_path, stubbed):
    for name in ("a.vtk", "b.vtk", "c.vtk"):
        write_surface(tmp_path / "in" / name)
    checkpoint = tmp_path / "airways_2_class.ckpt"
    checkpoint.write_bytes(b"")

    result = run(surfaces=str(tmp_path / "in"), model=str(checkpoint),
                 output_dir=str(tmp_path / "out"), explain=False)
    report = json.loads(open(result["report"], encoding="utf-8").read())
    assert report["surfaces"] == len(report["predictions"]) == 3


def test_a_suffix_is_appended_before_the_extension_not_after(tmp_path, stubbed):
    write_surface(tmp_path / "in" / "a.vtk")
    checkpoint = tmp_path / "airways_2_class.ckpt"
    checkpoint.write_bytes(b"")

    run(surfaces=str(tmp_path / "in"), model=str(checkpoint),
        output_dir=str(tmp_path / "out"), output_suffix="_gradcam")
    assert (tmp_path / "out" / "a_gradcam.vtk").exists()


def test_an_empty_suffix_leaves_the_written_surface_at_its_own_name(tmp_path, stubbed):
    write_surface(tmp_path / "in" / "a.vtk")
    checkpoint = tmp_path / "airways_2_class.ckpt"
    checkpoint.write_bytes(b"")

    run(surfaces=str(tmp_path / "in"), model=str(checkpoint),
        output_dir=str(tmp_path / "out"), output_suffix="")
    assert (tmp_path / "out" / "a.vtk").exists()


def test_a_regression_score_is_not_rounded_to_an_integer(tmp_path, monkeypatch, stubbed):
    write_surface(tmp_path / "in" / "a.vtk")
    checkpoint = tmp_path / "airways_4_regress.ckpt"
    checkpoint.write_bytes(b"")
    monkeypatch.setattr(engine, "predict", lambda *a, **k: [0.4999])

    result = run(surfaces=str(tmp_path / "in"), model=str(checkpoint),
                 output_dir=str(tmp_path / "out"), explain=False)
    assert result["predictions"][0]["score"] == pytest.approx(0.4999)


def test_a_classification_result_is_an_int_not_a_float(tmp_path, stubbed):
    write_surface(tmp_path / "in" / "a.vtk")
    checkpoint = tmp_path / "airways_2_class.ckpt"
    checkpoint.write_bytes(b"")

    result = run(surfaces=str(tmp_path / "in"), model=str(checkpoint),
                 output_dir=str(tmp_path / "out"), explain=False)
    assert isinstance(result["predictions"][0]["class"], int)


def test_the_input_folder_is_not_written_into(tmp_path, stubbed):
    """Every result goes to `output_dir`. Upstream wrote its file list into the
    output folder and read it back with mode 'a', so a second run appended to
    the first run's list."""
    write_surface(tmp_path / "in" / "a.vtk")
    checkpoint = tmp_path / "airways_2_class.ckpt"
    checkpoint.write_bytes(b"")
    before = sorted(os.listdir(tmp_path / "in"))

    run(surfaces=str(tmp_path / "in"), model=str(checkpoint),
        output_dir=str(tmp_path / "out"))
    assert sorted(os.listdir(tmp_path / "in")) == before


def test_a_second_run_into_the_same_output_folder_does_not_accumulate(tmp_path, stubbed):
    write_surface(tmp_path / "in" / "a.vtk")
    checkpoint = tmp_path / "airways_2_class.ckpt"
    checkpoint.write_bytes(b"")

    first = run(surfaces=str(tmp_path / "in"), model=str(checkpoint),
                output_dir=str(tmp_path / "out"), explain=False)
    second = run(surfaces=str(tmp_path / "in"), model=str(checkpoint),
                 output_dir=str(tmp_path / "out"), explain=False)
    assert len(first["predictions"]) == len(second["predictions"]) == 1


def test_a_single_surface_is_mounted_on_its_own_directory(tmp_path, stubbed):
    """The dataset joins its file column onto a mount point, so a single file
    has to be given a root it actually lives under."""
    surface = write_surface(tmp_path / "in" / "nested" / "a.vtk")
    checkpoint = tmp_path / "airways_2_class.ckpt"
    checkpoint.write_bytes(b"")

    result = run(surfaces=surface, model=str(checkpoint),
                 output_dir=str(tmp_path / "out"), explain=False)
    assert result["predictions"][0]["surface"] == "a.vtk"


def test_run_returns_the_three_keys_the_server_reads(tmp_path, stubbed):
    write_surface(tmp_path / "in" / "a.vtk")
    checkpoint = tmp_path / "airways_2_class.ckpt"
    checkpoint.write_bytes(b"")

    result = run(surfaces=str(tmp_path / "in"), model=str(checkpoint),
                 output_dir=str(tmp_path / "out"), explain=False)
    assert set(result) == {"outputs", "predictions", "report"}


def test_the_outputs_mapping_holds_only_paths_that_exist(tmp_path, stubbed):
    write_surface(tmp_path / "in" / "a.vtk")
    checkpoint = tmp_path / "cleft_4_class.ckpt"
    checkpoint.write_bytes(b"")

    result = run(surfaces=str(tmp_path / "in"), model=str(checkpoint),
                 output_dir=str(tmp_path / "out"))
    for path in result["outputs"].values():
        assert os.path.isfile(path), path


def test_a_missing_checkpoint_is_refused_before_any_heavy_import(tmp_path):
    """The guard has to fire without torch being touched: the server publishes
    schemas on machines that have no CUDA stack at all."""
    write_surface(tmp_path / "a.vtk")
    with pytest.raises(FileNotFoundError):
        run(surfaces=str(tmp_path), model="/nowhere/condyles_4_class.ckpt",
            output_dir=str(tmp_path / "out"))


def test_an_unknown_checkpoint_is_refused_before_the_surfaces_are_walked(tmp_path):
    with pytest.raises(ValueError, match="not a checkpoint"):
        run(surfaces=str(tmp_path / "does-not-exist"), model="mystery.ckpt",
            output_dir=str(tmp_path / "out"))


def test_run_has_only_the_annotations_the_schema_contract_allows():
    """The server builds this tool's schema from `run()`'s signature with the
    tool's own interpreter, and publishes it without importing the tool. Only
    `Path`, `str`, `int`, `float`, `bool`, `Literal[...]` and `list[...]` of
    those can be expressed; anything else fails schema generation."""
    import inspect
    import typing
    from pathlib import Path

    allowed = {Path, str, int, float, bool}
    for parameter in inspect.signature(run).parameters.values():
        annotation = parameter.annotation
        if typing.get_origin(annotation) is typing.Literal:
            assert all(isinstance(option, str) for option in typing.get_args(annotation))
        else:
            assert annotation in allowed, parameter.name


def test_run_returns_a_path_annotation_like_every_other_tool():
    import inspect
    from pathlib import Path

    assert inspect.signature(run).return_annotation is Path


def test_a_single_file_given_as_a_pathlib_path_is_accepted(tmp_path, stubbed):
    """Found by running the tool through the API rather than by a unit test:
    the runner hands a tool `pathlib.Path` for its `path` arguments, and a
    single uploaded surface reached `is_surface_file` as a Path, where
    `.lower()` does not exist. Every test until this one passed strings."""
    from pathlib import Path as P

    surface = write_surface(tmp_path / "in" / "a.vtk")
    checkpoint = tmp_path / "airways_2_class.ckpt"
    checkpoint.write_bytes(b"")

    result = run(surfaces=P(surface), model=P(checkpoint),
                 output_dir=P(tmp_path / "out"), explain=False)
    assert len(result["predictions"]) == 1


def test_discovery_accepts_a_pathlib_path_for_a_file_and_for_a_folder(tmp_path):
    from pathlib import Path as P

    surface = write_surface(tmp_path / "a.vtk")
    assert pipeline.discover_surfaces(P(tmp_path)) == [surface]
    assert pipeline.discover_surfaces(P(surface)) == [surface]


def test_is_surface_file_accepts_a_pathlib_path():
    from pathlib import Path as P

    assert pipeline.is_surface_file(P("/models/a.vtk"))
    assert not pipeline.is_surface_file(P("/models/a.nii.gz"))


def test_a_pathlib_path_is_accepted_wherever_a_path_is_declared(tmp_path, stubbed):
    """The server hands `run()` real strings, but the annotation says Path and
    a caller in another tool may well pass one."""
    from pathlib import Path as P

    write_surface(tmp_path / "in" / "a.vtk")
    checkpoint = tmp_path / "airways_2_class.ckpt"
    checkpoint.write_bytes(b"")

    result = run(surfaces=P(tmp_path / "in"), model=P(checkpoint),
                 output_dir=P(tmp_path / "out"), explain=False)
    assert len(result["predictions"]) == 1


def test_run_declares_an_output_dir_the_server_fills_in():
    import inspect

    assert "output_dir" in inspect.signature(run).parameters


def test_the_module_imports_nothing_heavy_at_module_level():
    """Import cost is paid on every schema publication, on machines that have
    no GPU. `import sadt_docshapeaxi` must not pull torch in."""
    import subprocess
    import sys

    finished = subprocess.run(
        [sys.executable, "-c",
         "import sys, sadt_docshapeaxi;"
         "heavy = [m for m in ('torch', 'shapeaxi', 'pandas', 'captum', 'vtk')"
         " if m in sys.modules];"
         "print(','.join(heavy))"],
        capture_output=True, text=True,
    )
    assert finished.returncode == 0, finished.stderr
    assert finished.stdout.strip() == ""


# ---------------------------------------------------------------------------
# The environment these checkpoints actually need
# ---------------------------------------------------------------------------

def test_the_backbone_is_refused_rather_than_downloaded(tmp_path, monkeypatch):
    """shapeaxi builds an EfficientNet around every checkpoint and fetches it
    from GitHub on first use. A server holding patient data does not make
    outbound calls mid-request, so it is staged and this refuses instead."""
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "empty-cache"))
    import torch

    torch.hub.get_dir.cache_clear() if hasattr(torch.hub.get_dir, "cache_clear") else None
    # ToolUnavailableError, so the server answers 503: no request can stage it.
    with pytest.raises(ToolUnavailableError) as raised:
        pipeline.check_backbone_is_staged()
    message = str(raised.value)
    assert pipeline.BACKBONE_FILE in message
    assert pipeline.BACKBONE_URL in message


def test_a_staged_backbone_satisfies_the_check(tmp_path, monkeypatch):
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "cache"))
    staged = tmp_path / "cache" / "hub" / "checkpoints" / pipeline.BACKBONE_FILE
    os.makedirs(staged.parent, exist_ok=True)
    staged.write_bytes(b"")
    pipeline.check_backbone_is_staged()


def test_the_backbone_path_honours_torch_home(tmp_path, monkeypatch):
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "elsewhere"))
    assert str(tmp_path / "elsewhere") in pipeline.backbone_cache_path()


def test_the_checkpoint_allowlist_names_classes_rather_than_disabling_the_check():
    """torch 2.6 defaults `weights_only=True` and these checkpoints pickle
    their training transform pipeline. `weights_only=False` would trust
    whatever the file contains; an allowlist names exactly what is trusted."""
    import inspect

    body = inspect.getsource(pipeline.allow_checkpoint_globals)
    body = body.split('"""')[-1]          # the code, not the docstring
    assert "add_safe_globals" in body
    assert "weights_only" not in body
    assert "torch.load" not in body


def test_the_allowlist_covers_the_transform_the_checkpoints_carry():
    torch = pytest.importorskip("torch")
    pytest.importorskip("shapeaxi")

    pipeline.allow_checkpoint_globals()
    allowed = torch.serialization.get_safe_globals()
    names = {getattr(entry, "__name__", str(entry)) for entry in allowed}
    assert "TrainTransform" in names
    assert "Compose" in names


def test_the_installed_shapeaxi_is_the_one_the_checkpoints_were_trained_with():
    """`MHAEncoder.__init__` changed signature at shapeaxi 2.0.0. Every
    published checkpoint carries the 1.x hyperparameters (`embed_dim`,
    `hidden_dim`, `num_heads=256`, `K`), so 2.x raises TypeError on all five.
    Upstream's own installer asks for `shapeaxi>=2.0.2`."""
    import importlib.metadata as metadata

    version = metadata.version("shapeaxi")
    assert version.split(".")[0] == "1", (
        f"shapeaxi {version} is installed; the published checkpoints need 1.x"
    )


def test_the_old_encoder_signature_is_what_the_checkpoints_expect():
    import inspect

    saxi_nets = pytest.importorskip("shapeaxi.saxi_nets")
    parameters = inspect.signature(saxi_nets.MHAEncoder.__init__).parameters
    assert {"embed_dim", "hidden_dim", "K"} <= set(parameters)


@pytest.mark.models
@pytest.mark.parametrize("checkpoint", sorted(catalog.ANALYSES))
def test_real_model_every_published_checkpoint_loads(checkpoint):
    """Run by hand against the staged checkpoints; the result goes in the PR."""
    models = os.environ.get("DOCSHAPEAXI_MODELS")
    if not models:
        pytest.skip("set DOCSHAPEAXI_MODELS")
    path = os.path.join(models, checkpoint)
    if not os.path.isfile(path):
        pytest.skip(f"{checkpoint} is not staged")

    analysis = catalog.analysis_for(checkpoint)
    network = pipeline.load_network(path, analysis.network, "cpu")
    assert network is not None


def test_the_gradcam_namespace_carries_the_class_it_is_explaining():
    """`gradcam_process` names the point array it writes after
    `args.target_class`; without it every class writes `grad_cam_max` and the
    last one wins, so a four-class mesh carries one array instead of four."""
    from sadt_docshapeaxi.engine import _Namespace

    namespace = _Namespace(device="cpu", target_class=3)
    assert (namespace.device, namespace.target_class) == ("cpu", 3)


def test_the_gradcam_namespace_defaults_to_no_target_class():
    from sadt_docshapeaxi.engine import _Namespace

    assert _Namespace(device="cpu").target_class is None


# ---------------------------------------------------------------------------
# Progress -- the load announced, then each pass counted per surface
# ---------------------------------------------------------------------------

def _events(events_file):
    """Progress records and log records, apart. Logs carry `kind: log`."""
    records = [json.loads(line) for line in events_file.read_text().splitlines() if line]
    return ([r for r in records if r.get("kind") != "log"],
            [r for r in records if r.get("kind") == "log"])


class _FakeNetwork:
    """Just enough of a shapeaxi network for `engine.predict` to loop over."""

    class hparams:
        sample_levels = [16]

    def create_mesh(self, vertices, faces, normals):
        return vertices

    def sample_points_from_meshes(self, mesh, level):
        return mesh

    def render(self, mesh):
        return mesh, None

    def __call__(self, points, views):
        import torch

        return torch.tensor([[0.1, 0.9]])


def test_the_prediction_pass_counts_surfaces_and_never_names_one(tmp_path, monkeypatch):
    """The real `engine.predict` loop, over a stand-in network and dataset:
    the count it reports is the surfaces it has actually taken."""
    torch = pytest.importorskip("torch")
    events_file = tmp_path / "events.jsonl"
    monkeypatch.setenv("SADT_PROGRESS_FILE", str(events_file))
    monkeypatch.setattr(pipeline, "resolve_device", lambda requested: "cpu")
    monkeypatch.setattr(pipeline, "load_network", lambda *a, **k: _FakeNetwork())
    monkeypatch.setattr(
        engine, "_dataset",
        lambda model, surfaces, mount_point, device: [
            (torch.zeros(3, 3), torch.zeros(1, 3, dtype=torch.long), torch.zeros(3, 3))
            for _ in surfaces
        ],
    )
    for name in ("Smith_John", "Jones_Mary"):
        write_surface(tmp_path / "in" / (name + ".vtk"))
    checkpoint = tmp_path / "airways_2_class.ckpt"
    checkpoint.write_bytes(b"")

    result = run(surfaces=str(tmp_path / "in"), model=str(checkpoint),
                 output_dir=str(tmp_path / "out"), explain=False)

    assert [entry["class"] for entry in result["predictions"]] == [1, 1]
    events, logs = _events(events_file)
    assert [event["message"] for event in events] == [
        "loading the model", "classifying surface 1 of 2", "classifying surface 2 of 2",
    ]
    assert [event["fraction"] for event in events] == [0.0, 0.0, 0.5]
    assert logs == []
    text = events_file.read_text()
    assert "Smith" not in text and "Jones" not in text and ".vtk" not in text


def test_the_explanation_takes_the_bar_from_where_the_prediction_left_it(
    tmp_path, monkeypatch, stubbed
):
    """Two passes, one bar: the explanation must not send it back to zero."""
    spans = {}
    real_predict, real_explain = engine.predict, engine.explain

    def predict(*args, span=(0.0, 1.0), **kwargs):
        spans["predict"] = span
        return real_predict(*args, span=span, **kwargs)

    def explain(*args, span=(0.0, 1.0), **kwargs):
        spans["explain"] = span
        return real_explain(*args, span=span, **kwargs)

    monkeypatch.setattr(engine, "predict", predict)
    monkeypatch.setattr(engine, "explain", explain)
    write_surface(tmp_path / "in" / "a.vtk")
    checkpoint = tmp_path / "condyles_4_class.ckpt"
    checkpoint.write_bytes(b"")

    run(surfaces=str(tmp_path / "in"), model=str(checkpoint),
        output_dir=str(tmp_path / "out"))

    assert spans["predict"][1] == spans["explain"][0]
    assert 0.0 < spans["explain"][0] < 1.0
    assert spans["explain"][1] == 1.0


# ---------------------------------------------------------------------------
# What an operator reads when a run fails
# ---------------------------------------------------------------------------

def test_an_unknown_model_names_the_argument_and_the_analyses_in_words():
    """File names are redacted on their way to a log; the anatomy and task
    pairs are what an operator can still read."""
    with pytest.raises(ToolInputError) as raised:
        catalog.analysis_for("nose_9_class.ckpt")
    message = str(raised.value)
    assert message.startswith("'model'")
    assert "Mandibular condyle (severity)" in message
    assert "Nasopharynx airway obstruction (binary, severity, regression)" in message


def test_a_missing_checkpoint_names_the_argument_and_the_analysis(tmp_path):
    write_surface(tmp_path / "a.vtk")
    with pytest.raises(FileNotFoundError) as raised:
        run(surfaces=str(tmp_path), model=str(tmp_path / "condyles_4_class.ckpt"),
            output_dir=str(tmp_path / "out"))
    assert str(raised.value).startswith("'model': the Mandibular condyle severity")


def test_an_empty_surface_is_refused_by_position_before_the_model_loads(tmp_path, monkeypatch):
    write_surface(tmp_path / "in" / "a.vtk")
    (tmp_path / "in" / "b.vtk").write_text("not a mesh")
    checkpoint = tmp_path / "airways_2_class.ckpt"
    checkpoint.write_bytes(b"")
    monkeypatch.setattr(pipeline, "load_network",
                        lambda *a, **k: pytest.fail("the model was loaded"))

    with pytest.raises(ToolInputError) as raised:
        run(surfaces=str(tmp_path / "in"), model=str(checkpoint),
            output_dir=str(tmp_path / "out"))
    message = str(raised.value)
    assert "surface 2 of 2 is empty or unreadable" in message
    assert "b.vtk" not in message


def test_same_named_surfaces_are_refused_by_count_when_explaining(tmp_path, stubbed):
    write_surface(tmp_path / "in" / "Smith" / "scan.vtk")
    write_surface(tmp_path / "in" / "Jones" / "scan.vtk")
    write_surface(tmp_path / "in" / "other.vtk")
    checkpoint = tmp_path / "airways_2_class.ckpt"
    checkpoint.write_bytes(b"")

    with pytest.raises(ToolInputError) as raised:
        run(surfaces=str(tmp_path / "in"), model=str(checkpoint),
            output_dir=str(tmp_path / "out"))
    message = str(raised.value)
    assert message.startswith("'surfaces': 2 of 3 surfaces share a file name")
    assert "scan" not in message and "Smith" not in message


def test_the_network_load_is_announced_with_what_and_where(tmp_path, monkeypatch, caplog):
    import sys
    import types

    class Exploding:
        @classmethod
        def load_from_checkpoint(cls, *args, **kwargs):
            raise RuntimeError("a long pickle preamble\nthe actual cause")

    module = types.ModuleType("shapeaxi.saxi_nets_lightning")
    module.SaxiMHAFBClassification = Exploding
    shapeaxi = types.ModuleType("shapeaxi")
    shapeaxi.saxi_nets_lightning = module
    monkeypatch.setitem(sys.modules, "shapeaxi", shapeaxi)
    monkeypatch.setitem(sys.modules, "shapeaxi.saxi_nets_lightning", module)
    monkeypatch.setattr(pipeline, "check_backbone_is_staged", lambda: None)
    monkeypatch.setattr(pipeline, "allow_checkpoint_globals", lambda: None)

    with caplog.at_level("INFO", logger="DOCShapeAXI"):
        with pytest.raises(RuntimeError) as raised:
            pipeline.load_network("/models/x.ckpt", "SaxiMHAFBClassification",
                                  "cpu", anatomy="Mandibular condyle")
    assert ("loading SaxiMHAFBClassification checkpoint for Mandibular condyle on cpu"
            in caplog.messages)
    assert str(raised.value).endswith("failed: RuntimeError: the actual cause")
    assert not isinstance(raised.value, ValueError)


def test_a_surface_that_fails_mid_pass_is_logged_by_position(tmp_path, monkeypatch, caplog):
    torch = pytest.importorskip("torch")

    class Failing(_FakeNetwork):
        calls = 0

        def __call__(self, points, views):
            Failing.calls += 1
            if Failing.calls == 2:
                raise RuntimeError("CUDA out of memory")
            return super().__call__(points, views)

    monkeypatch.setattr(pipeline, "resolve_device", lambda requested: "cpu")
    monkeypatch.setattr(pipeline, "load_network", lambda *a, **k: Failing())
    monkeypatch.setattr(
        engine, "_dataset",
        lambda model, surfaces, mount_point, device: [
            (torch.zeros(3, 3), torch.zeros(1, 3, dtype=torch.long), torch.zeros(3, 3))
            for _ in surfaces
        ],
    )
    for name in ("Smith_John", "Jones_Mary", "Doe_Jane"):
        write_surface(tmp_path / "in" / (name + ".vtk"))
    checkpoint = tmp_path / "airways_2_class.ckpt"
    checkpoint.write_bytes(b"")

    with caplog.at_level("INFO", logger="DOCShapeAXI"):
        with pytest.raises(RuntimeError, match="CUDA out of memory"):
            run(surfaces=str(tmp_path / "in"), model=str(checkpoint),
                output_dir=str(tmp_path / "out"), explain=False)
    assert ("surface 2 of 3: classification failed (RuntimeError: CUDA out of memory)"
            in caplog.messages)
    assert "Smith" not in caplog.text and ".vtk" not in caplog.text


def test_a_finished_run_logs_how_much_was_done(tmp_path, stubbed, caplog):
    write_surface(tmp_path / "in" / "a.vtk")
    write_surface(tmp_path / "in" / "b.vtk")
    checkpoint = tmp_path / "condyles_4_class.ckpt"
    checkpoint.write_bytes(b"")

    with caplog.at_level("INFO", logger="DOCShapeAXI"):
        run(surfaces=str(tmp_path / "in"), model=str(checkpoint),
            output_dir=str(tmp_path / "out"))
    assert "2 of 2 surfaces classified, 2 explained" in caplog.messages


def test_the_explanation_loader_runs_in_process():
    """A worker process's failure reaches the server as a cut traceback string
    and its log records never arrive."""
    import inspect

    assert "num_workers=0" in inspect.getsource(engine.explain)
