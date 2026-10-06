"""The claims README.md and pyproject.toml make, asserted rather than described.

Four numbers have to line up for this tool to run at all -- torch version, CUDA
variant, Python ABI, and the pytorch3d wheel tag that names the first two. A
mismatch is not an install error: the package imports fine and dies on the
first CUDA kernel. So the pins are checked here, where a change to one of them
fails a test instead of a clinician's run.
"""

import ast
import json
import subprocess
import sys
import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pytest

import sadt_crownseg
from sadt_crownseg import pipeline, run

TOOL_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = TOOL_DIR.parents[1]


def pyproject():
    return tomllib.loads((TOOL_DIR / "pyproject.toml").read_text(encoding="utf-8"))


def readme():
    return (TOOL_DIR / "README.md").read_text(encoding="utf-8")


def source_text():
    return "\n".join(
        path.read_text(encoding="utf-8") for path in (TOOL_DIR / "src").rglob("*.py")
    )


def imported_names():
    names = set()
    for path in (TOOL_DIR / "src").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module.split(".")[0])
    return names


def extra_is_installed() -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec("shapeaxi") is not None
    except (ImportError, ValueError):
        # A shadowed or half-removed distribution: `find_spec` raises rather
        # than answering, and "not installed" is the honest reading.
        return False


# ---------------------------------------------------------------------------
# The four numbers that must agree
# ---------------------------------------------------------------------------

BASE_PINS = {"torch": "2.11.0", "vtk": "9.6.2", "numpy": "2.3.2"}
EXTRA_PINS = {
    "ocnn": "2.2.1",
    "pytorch3d": "0.7.9+pt2110cu128",
    "torchvision": "0.26.0",
}


@pytest.mark.parametrize("package, pin", sorted(BASE_PINS.items()))
def test_the_base_dependencies_are_pinned_to_the_documented_version(package, pin):
    """A plain `uv sync` installs these, so the pin and the installed version
    are both checkable and both have to agree with the README."""
    assert f"{package}=={pin}" in pyproject()["project"]["dependencies"]
    assert version(package).split("+")[0] == pin
    assert f"{package} {pin}" in readme()


@pytest.mark.parametrize("package, pin", sorted(EXTRA_PINS.items()))
def test_the_engine_extra_pins_the_documented_versions(package, pin):
    """These arrive only with `--extra segmentation`, so the pin is what is
    always checkable. `pytorch3d==0.7.9+pt2110cu128` names one torch version
    and one CUDA variant exactly."""
    extra = pyproject()["project"]["optional-dependencies"]["segmentation"]
    assert f"{package}=={pin}" in extra


def test_the_pytorch3d_wheel_tag_names_the_torch_that_is_pinned():
    """Bumping torch means bumping the wheel tag in the same commit; they are
    one decision, not two. `+pt2110` is torch 2.11.0."""
    extra = pyproject()["project"]["optional-dependencies"]["segmentation"]
    wheel = next(entry for entry in extra if entry.startswith("pytorch3d"))
    torch_pin = next(
        entry for entry in pyproject()["project"]["dependencies"]
        if entry.startswith("torch==")
    )

    major, minor, _ = torch_pin.split("==")[1].split(".")
    assert f"+pt{major}{int(minor):02d}0" in wheel


def test_shapeaxi_is_asked_for_at_least_the_release_the_workaround_targets():
    """`_restore_moved_class` is written against 2.0.x, and its error message
    names 2.0.2."""
    extra = pyproject()["project"]["optional-dependencies"]["segmentation"]
    assert "shapeaxi>=2.0.2" in extra
    assert "2.0.2" in pipeline._restore_moved_class.__doc__


def test_torch_is_pinned_although_upstream_leaves_it_unpinned():
    """Deliberately stricter than upstream: an unpinned torch is not
    reproducible, and the same `uv sync` two months apart gives two different
    runtimes."""
    dependencies = pyproject()["project"]["dependencies"]
    assert "torch" not in dependencies
    assert any(entry.startswith("torch==") for entry in dependencies)


def test_torchvision_is_a_direct_dependency_of_the_extra():
    """`tool.uv.sources` applies to direct dependencies only. Left transitive
    through shapeaxi it comes from PyPI, built against the default torch, and
    importing it raises `operator torchvision::nms does not exist`."""
    extra = pyproject()["project"]["optional-dependencies"]["segmentation"]
    assert any(entry.startswith("torchvision==") for entry in extra)
    assert "torchvision" not in imported_names(), "nothing imports it by name"


@pytest.mark.parametrize("index", ["pytorch-cu128", "pytorch3d-wheels"])
def test_both_wheel_indexes_are_pinned_sources_not_extra_index_urls(index):
    """`explicit = true` is what makes them pinned sources. As extra index URLs
    uv is free to mix registries: a torch 2.11.0+cu130 paired with a
    pt2120cu126 wheel, failing at runtime on a missing libcudart."""
    entry = next(item for item in pyproject()["tool"]["uv"]["index"] if item["name"] == index)

    assert entry["explicit"] is True


@pytest.mark.parametrize(
    "package, index",
    [("torch", "pytorch-cu128"), ("torchvision", "pytorch-cu128"),
     ("pytorch3d", "pytorch3d-wheels")],
)
def test_each_compiled_package_comes_from_the_index_it_was_built_for(package, index):
    assert pyproject()["tool"]["uv"]["sources"][package] == {"index": index}


def test_the_interpreter_is_the_one_pytorch3d_is_built_for_here():
    """Upstream moved this env to 3.12 because shapeaxi 1.0.10 pinned
    grpcio==1.51.1, which has no cp312 wheel; 2.0.2 dropped that pin, so the
    reason for the move no longer applies."""
    assert pyproject()["project"]["requires-python"] == ">=3.11,<3.12"
    assert sys.version_info[:2] == (3, 11)


# ---------------------------------------------------------------------------
# What the port removed
# ---------------------------------------------------------------------------

def test_the_tool_unpacks_nothing():
    """Zip extraction is gone: the server unpacks archives before `run()`."""
    for absent in ("zipfile", "extractall", "unpack_archive"):
        assert absent not in source_text(), absent


def test_the_gpu_semaphore_is_gone():
    """Each call is its own process now, so an in-process semaphore would cap
    nothing. `MAX_CONCURRENT_GPU_JOBS` is one counter across tools, server-side."""
    assert "Semaphore" not in source_text()
    # `threading` is imported, but only for the daemon thread that counts
    # finished meshes for the progress bar; nothing here takes a lock that
    # could cap concurrency.
    assert "Lock(" not in source_text()


def test_no_temporary_directory_is_created_outside_the_output_directory():
    """Scratch space lives under `output_dir` as `.crownseg_work/`. A tool
    writing to the system temp directory leaves patient file paths where the
    server's cleanup never looks."""
    assert "tempfile" not in imported_names()
    assert "mkdtemp" not in source_text()
    assert pipeline.WORK_DIRNAME == ".crownseg_work"


def test_the_model_may_be_left_unnamed():
    """It used to be a name resolved through the data store, with a default
    read from `settings.CROWNSEG_MODEL`; then it was a required Path.

    It is optional now, and that is what lets a supervised caller omit it:
    ALI_IOS and AREG ask this tool to label meshes without knowing which
    bundle this deployment staged. Empty means "my own", found under the data
    root the server injects.
    """
    import inspect

    parameter = inspect.signature(run).parameters["model"]
    assert parameter.default == ""
    assert parameter.annotation is Path


def test_segment_crowns_returns_the_run_report():
    """Not a `CrownSegRun` object -- the server has no class of this tool's to
    import, and the report is what crosses the process boundary."""
    assert pipeline.segment_crowns.__annotations__["return"] is dict


def test_this_tool_cannot_call_another_one():
    """ALI used to import this module to segment raw meshes inline. Tools do
    not call each other any more: the server sequences Crown_Seg before ALI and
    reads `segmented_meshes`. A tool that COULD call one declares `*, sup`, and
    this one does not -- so the runner injects no supervisor into it."""
    import inspect

    assert "sup" not in inspect.signature(run).parameters
    assert not (imported_names() & {"sadt_ali", "sadt_aso", "sadt_areg", "requests"})


def test_the_tool_reads_no_server_setting():
    """A packaged tool is its own process and shares no configuration; what
    used to be `settings.CROWNSEG_MODEL` is an argument of `run()`."""
    assert "settings" not in imported_names()
    # The vendored `progress.py` is the one exception: it reads the path of
    # the progress channel the server names, which is a protocol every tool
    # speaks rather than a setting of this one.
    assert "os.environ" not in "\n".join(
        path.read_text(encoding="utf-8")
        for path in (TOOL_DIR / "src").rglob("*.py")
        if path.name != "progress.py"
    )


# ---------------------------------------------------------------------------
# The extensions and the array names the README promises
# ---------------------------------------------------------------------------

def test_the_discovered_extensions_are_the_ones_the_readme_names():
    assert pipeline.SURFACE_EXTENSIONS == (".vtk", ".stl")
    assert ".vtk`/`.stl" in readme() or "(.vtk/.stl)" in readme()


def test_the_label_arrays_that_mean_a_mesh_is_done():
    """Three spellings are in circulation; the default written by this tool is
    one of them, so its own output re-reads as segmented."""
    assert pipeline.DEFAULT_ARRAY_NAME in pipeline.LABEL_ARRAY_NAMES
    # `FDI_ID` joined them with the FDI numbering option: a mesh this tool
    # labelled in FDI has to re-read as segmented too, or a second pass would
    # segment it again.
    assert pipeline.LABEL_ARRAY_NAMES == (
        "PredictedID", "UniversalID", "Universal_ID", "FDI_ID",
    )


def test_the_default_suffix_is_the_one_the_docstring_shows():
    """The docstring's example is what `describe.py` publishes and the client
    renders under the field; an example naming another suffix is a promise the
    tool does not keep."""
    assert pipeline.DEFAULT_SUFFIX == "Seg"
    assert f"arch_{pipeline.DEFAULT_SUFFIX}.vtk" in run.__doc__


# ---------------------------------------------------------------------------
# The schema the server publishes
# ---------------------------------------------------------------------------

def test_the_declared_tool_name_is_the_api_identity():
    assert pyproject()["tool"]["sadt"]["name"] == TOOL_DIR.name == "Crown_Seg"
    assert pyproject()["tool"]["sadt"]["tool"] is True
    assert pipeline.TOOL_NAME == "Crown_Seg"


def test_every_argument_of_run_is_documented_for_the_client():
    """`describe.py` refuses to emit a schema for an argument with no
    description, and the client renders it under the field."""
    import inspect

    documented = run.__doc__.split("Args:", 1)[1]
    for name, spec in inspect.signature(run).parameters.items():
        if spec.kind is inspect.Parameter.KEYWORD_ONLY:
            # Injected by the server, never published and never rendered, so
            # there is no client to describe it to. `data_root` is the one.
            continue
        assert f"{name}:" in documented, name


def test_describe_publishes_the_schema_this_tool_promises():
    """The signature is the single source of truth for the panel a clinician
    sees, and importing this package must cost nothing -- the segmentation
    stack is an extra a CI venv deliberately does not have."""
    script = REPO_ROOT / "scripts" / "describe.py"
    if not script.is_file():
        pytest.skip("run from a full SADT-VISOR checkout")

    finished = subprocess.run(
        [sys.executable, str(script), str(TOOL_DIR)],
        capture_output=True, text=True, check=False,
    )
    assert finished.returncode == 0, finished.stderr

    schema = json.loads(finished.stdout)
    assert schema["name"] == "Crown_Seg"
    assert sorted(schema["arguments"]) == [
        "array_name", "device", "meshes", "model", "num_workers", "numbering",
        "output_dir", "skip_segmented", "suffix",
    ]
    assert schema["arguments"]["numbering"]["choices"] == ["Universal", "FDI"]
    assert schema["arguments"]["device"]["choices"] == ["cuda", "cpu"]
    assert schema["arguments"]["skip_segmented"]["default"] is True
    assert schema["arguments"]["num_workers"]["default"] == 2


def test_importing_the_package_pulls_in_no_engine():
    """`_import_dental_model_seg` is deferred precisely so that a venv without
    the extra still loads and reports its schema; only a run fails."""
    top_level = set()
    module = TOOL_DIR / "src" / "sadt_crownseg" / "__init__.py"
    for node in ast.parse(module.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Import):
            top_level.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level.add(node.module.split(".")[0])

    # `errors` is this package's own ToolInputError, not an engine: it is what
    # lets the module refuse a request without the segmentation stack loaded.
    assert top_level == {"pathlib", "typing", "pipeline", "errors"}


def test_the_pipeline_defers_every_heavy_import():
    """vtk, torch and shapeaxi are all imported inside the function that needs
    them, for the same reason."""
    top_level = set()
    module = TOOL_DIR / "src" / "sadt_crownseg" / "pipeline.py"
    for node in ast.parse(module.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Import):
            top_level.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level.add(node.module.split(".")[0])

    for heavy in ("vtk", "torch", "shapeaxi"):
        assert heavy not in top_level, heavy


def test_the_public_surface_is_run_alone():
    tree = ast.parse((TOOL_DIR / "src" / "sadt_crownseg" / "__init__.py").read_text())
    defined = {
        node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    # `_own_models` is private by name and exists for the same reason `run`'s
    # `model` became optional: a caller that names no bundle gets this tool's
    # own, resolved from the data root the server injects.
    assert {name for name in defined if not name.startswith("_")} == {"run"}
    assert sadt_crownseg.run is run


# ---------------------------------------------------------------------------
# The engine extra, when it is installed
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not extra_is_installed(), reason="needs `uv sync --extra segmentation`")
def test_the_installed_engine_is_the_documented_release():
    """Only checkable where the extra is installed; the pin above is what CI
    can see."""
    assert version("shapeaxi") == "2.0.2"
    for package, pin in EXTRA_PINS.items():
        try:
            assert version(package).split("+")[0] == pin.split("+")[0], package
        except PackageNotFoundError:  # pragma: no cover - the extra is partial
            pytest.fail(f"{package} is missing from an installed extra")
