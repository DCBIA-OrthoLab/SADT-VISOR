"""The seam between this tool and shapeaxi.

Nothing here is ported: the Slicer modules ran the `dentalmodelseg` executable
out of Slicer's own bin directory, and that executable is only the console
script of `shapeaxi.dental_model_seg`. There is no binary to shell out to, so
`main()` is called directly with the namespace its own `cml()` would have
built -- which makes that namespace the contract, and the thing to pin.
"""

import sys
import types

import pytest

from sadt_crownseg import pipeline
from sadt_crownseg.errors import ToolInputError, ToolUnavailableError


@pytest.fixture
def captured_namespace(monkeypatch):
    """Run `_run_shapeaxi` against a stand-in `main()` and keep what it got."""
    seen = {}

    class FakeModule:
        @staticmethod
        def main(args):
            seen.update(vars(args))

    monkeypatch.setattr(pipeline, "_import_dental_model_seg", lambda: FakeModule)
    return seen


def invoke(tmp_path, **overrides):
    arguments = dict(
        csv_path=str(tmp_path / "crownseg_input.csv"),
        output_dir=str(tmp_path / "out"),
        model_path=str(tmp_path / "model.pth"),
        input_root=str(tmp_path / "cohort"),
        array_name="Universal_ID",
        suffix="Seg",
        device="cpu",
        fdi=False,
        num_workers=2,
    )
    arguments.update(overrides)
    pipeline._run_shapeaxi(**arguments)


# ---------------------------------------------------------------------------
# The namespace cml() would have built
# ---------------------------------------------------------------------------

def test_the_namespace_carries_exactly_the_fields_shapeaxi_reads(tmp_path, captured_namespace):
    """An extra field is harmless; a MISSING one is an AttributeError inside
    shapeaxi's `main()`, after the model has loaded."""
    invoke(tmp_path)

    assert set(captured_namespace) == {
        "surf", "csv", "model", "suffix", "out", "num_workers",
        "crown_segmentation", "array_name", "fdi", "overwrite", "device",
        "vtk_folder",
    }


def test_the_csv_branch_is_selected_by_leaving_surf_unset(tmp_path, captured_namespace):
    """`surf` and `csv` are shapeaxi's two input modes. Setting both would make
    it segment one mesh and ignore the batch."""
    invoke(tmp_path)

    assert captured_namespace["surf"] is None
    assert captured_namespace["csv"] == str(tmp_path / "crownseg_input.csv")


def test_the_output_root_and_the_tree_root_are_two_different_arguments(
    tmp_path, captured_namespace
):
    """`out` is where results go, `vtk_folder` is the prefix stripped off each
    input path to rebuild the tree under it. Swapping them flattens a cohort."""
    invoke(tmp_path)

    assert captured_namespace["out"] == str(tmp_path / "out")
    assert captured_namespace["vtk_folder"] == str(tmp_path / "cohort")


def test_the_crown_segmentation_and_overwrite_flags_stay_off(tmp_path, captured_namespace):
    """`crown_segmentation` is shapeaxi's own post-processing pass and
    `overwrite` would let a second run replace the first's outputs in place;
    neither is what this tool asks for."""
    invoke(tmp_path)

    assert captured_namespace["crown_segmentation"] == 0
    assert captured_namespace["overwrite"] == 0


def test_the_checkpoint_and_the_array_name_are_passed_through_verbatim(
    tmp_path, captured_namespace
):
    invoke(tmp_path, array_name="PredictedID", suffix="Pred", device="cuda:0")

    assert captured_namespace["model"] == str(tmp_path / "model.pth")
    assert captured_namespace["array_name"] == "PredictedID"
    assert captured_namespace["suffix"] == "Pred"
    assert captured_namespace["device"] == "cuda:0"


# ---------------------------------------------------------------------------
# What shapeaxi prints, and what it raises
# ---------------------------------------------------------------------------

def test_shapeaxis_stdout_never_reaches_the_process_output(tmp_path, monkeypatch, capsys):
    """It prints "Saving results to <path>" for every mesh, and that path
    carries the patient's own file name -- which must not reach a log."""
    class Chatty:
        @staticmethod
        def main(args):
            print("Saving results to /data/patients/Smith_John_T1_Seg.vtk")

    monkeypatch.setattr(pipeline, "_import_dental_model_seg", lambda: Chatty)

    invoke(tmp_path)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Smith_John" not in captured.err


def test_a_real_failure_from_shapeaxi_is_left_untouched(tmp_path, monkeypatch):
    """Swallowing stdout must not swallow the exception with it: the run has to
    fail, and the server has to see why."""
    class Failing:
        @staticmethod
        def main(args):
            raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(pipeline, "_import_dental_model_seg", lambda: Failing)

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        invoke(tmp_path)


def test_stdout_is_restored_after_the_call(tmp_path, monkeypatch, capsys):
    """A redirect left in place would silence the whole worker process."""
    class Quiet:
        @staticmethod
        def main(args):
            pass

    monkeypatch.setattr(pipeline, "_import_dental_model_seg", lambda: Quiet)

    invoke(tmp_path)
    print("still here")

    assert capsys.readouterr().out == "still here\n"


def test_stdout_is_restored_even_when_shapeaxi_raises(tmp_path, monkeypatch, capsys):
    class Failing:
        @staticmethod
        def main(args):
            print("half a mesh")
            raise RuntimeError("boom")

    monkeypatch.setattr(pipeline, "_import_dental_model_seg", lambda: Failing)

    with pytest.raises(RuntimeError):
        invoke(tmp_path)
    print("still here")

    assert capsys.readouterr().out == "still here\n"


# ---------------------------------------------------------------------------
# A venv without the segmentation extra
# ---------------------------------------------------------------------------

def test_an_absent_shapeaxi_says_how_to_install_it(monkeypatch):
    """`uv sync` alone cannot run the network, and has to say so. The message
    is the only thing between a maintainer and a bare ModuleNotFoundError from
    inside shapeaxi -- and a 501 body a clinician reads."""
    monkeypatch.setitem(sys.modules, "shapeaxi", None)

    with pytest.raises(ToolUnavailableError) as failure:
        pipeline._import_dental_model_seg()

    message = str(failure.value)
    assert "uv sync --extra segmentation" in message
    assert "tools/Crown_Seg" in message
    assert "missing: shapeaxi" in message


def test_the_engine_import_probe_returns_shapeaxis_own_module(monkeypatch):
    """Run against a stand-in shapeaxi so it holds in a venv without the
    extra, which is what CI has. Nothing is patched into it any more: the
    shapeaxi pinned finds DentalModelSeg on its own."""
    shapeaxi = types.ModuleType("shapeaxi")
    shapeaxi.dental_model_seg = types.ModuleType("shapeaxi.dental_model_seg")
    monkeypatch.setitem(sys.modules, "shapeaxi", shapeaxi)
    monkeypatch.setitem(sys.modules, "shapeaxi.dental_model_seg", shapeaxi.dental_model_seg)

    assert pipeline._import_dental_model_seg() is shapeaxi.dental_model_seg


# ---------------------------------------------------------------------------
# The error classes the server maps to status codes
# ---------------------------------------------------------------------------

def test_the_input_error_is_the_one_the_server_answers_422_with():
    """`dispatch.py` maps by exception class NAME, there being no shared
    package to define a base class in. The base class matters too: a plain
    ValueError escaping the tool is answered the same way."""
    assert ToolInputError.__name__ == "ToolInputError"
    assert issubclass(ToolInputError, ValueError)


def test_the_unavailable_error_is_the_one_the_server_answers_501_with():
    assert ToolUnavailableError.__name__ == "ToolUnavailableError"
    assert issubclass(ToolUnavailableError, RuntimeError)
    assert not issubclass(ToolUnavailableError, ValueError)
