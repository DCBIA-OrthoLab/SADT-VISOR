"""A mixed batch: the meshes that already work first, the rest segmented.

A clinician sends a folder of intraoral scans. Some have been through
`Crown_Seg` and carry a tooth-label array; some have not. The landmark networks
are pointed at teeth, so the raw ones are nothing they can be asked about --
and until now that refused the WHOLE batch with a message naming a tool the
person sending the request has no reason to have heard of.

What is pinned here is the ORDER and what survives a failure, not the
inference: the labelled meshes are processed first, so their landmarks are on
disk before a second of segmentation is spent, and every way the segmentation
can fail costs the meshes it was asked about and nothing else.

The engine and the supervisor are both stubbed. Nothing here needs a GPU,
pytorch3d or a model bundle; what it needs is vtk, to write meshes that really
do and really do not carry a label array -- that being the actual question the
run branches on.
"""

import inspect
import json
import os
import re
from pathlib import Path

import pytest

from sadt_ali_ios import dispatch, engine, run
from sadt_ali_ios.errors import ToolInputError


# Copied from `test_run.py` rather than shared: CONTRIBUTING.md says a test
# helper is duplicated rather than given a package of its own, and this one
# already crossed from ALI_CBCT's suite the same way.
def write_surface(path, labelled=True):
    """A minimal .vtk polydata, optionally carrying a tooth-label array."""
    vtk = pytest.importorskip("vtk")

    points = vtk.vtkPoints()
    for coordinates in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 1)):
        points.InsertNextPoint(*coordinates)

    polys = vtk.vtkCellArray()
    for triangle in ((0, 1, 2), (0, 1, 3), (1, 2, 3), (0, 2, 3)):
        polys.InsertNextCell(3)
        for point_id in triangle:
            polys.InsertCellPoint(point_id)

    surface = vtk.vtkPolyData()
    surface.SetPoints(points)
    surface.SetPolys(polys)

    if labelled:
        labels = vtk.vtkIntArray()
        labels.SetName("Universal_ID")
        for value in (8, 8, 8, 8):
            labels.InsertNextValue(value)
        surface.GetPointData().AddArray(labels)

    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(surface)
    writer.Write()
    return str(path)


class FakeSup:
    """A supervisor, as a tool sees one. Records what it was asked for.

    Five members and no type: a tool never imports a supervisor class, so a
    dozen lines here are indistinguishable from the server's own.
    """

    def __init__(self, tmp_path, outputs=None):
        self.out = Path(tmp_path) / "out"
        self.tmp = Path(tmp_path) / "tmp"
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.outputs = outputs or {}
        self.calls = []
        self.messages = []
        self.call_index = 0

    def run(self, tool, **params):
        self.calls.append((tool, params))
        maker = self.outputs.get(tool)
        if maker is None:
            raise AssertionError(f"nothing planted for {tool!r} in this test")
        # The SUPERVISOR owns where a callee writes, exactly as the server owns
        # it for a request: the caller passes no `output_dir` and reads the path
        # back off the return value. Modelled here, because a fake that let the
        # caller name the directory would accept a caller that named one.
        assert "output_dir" not in params, (
            "a caller must not name where a supervised tool writes"
        )
        self.call_index += 1
        nested = self.tmp / "sup" / f"{self.call_index:02d}_{tool}" / "output"
        nested.mkdir(parents=True, exist_ok=True)
        return Path(maker(dict(params, output_dir=str(nested))))

    def progress(self, fraction, message):
        self.messages.append((fraction, message))

    def log(self, message):
        self.messages.append((None, message))


def crown_seg(failed=(), seen=None):
    """A stand-in for Crown_Seg: labels what it was given, reports what it did.

    The seam is the run report, so that is what this writes -- `meshes`, keyed
    by the path relative to what it was handed, each entry carrying the
    absolute path of the labelled copy or the reason there is none. `failed`
    names keys it refuses, which is how a batch with one unsegmentable mesh in
    it is built.

    `seen` collects the keys it was actually handed. Recorded HERE because the
    staging directory lives under the run's work dir, which is removed before
    `run()` returns -- a test asserting on it afterwards would be asserting on
    a folder that is correctly gone.
    """
    def make(params):
        root = str(params["meshes"])
        output_dir = str(params["output_dir"])
        records = {}
        for directory, _subdirs, names in os.walk(root):
            for name in sorted(names):
                key = os.path.relpath(os.path.join(directory, name), root)
                if seen is not None:
                    seen.append(key)
                if key in failed:
                    records[key] = {
                        "status": "failed",
                        "error": "the segmentation produced no output for this mesh",
                    }
                    continue
                # Where the real one writes its csv branch's outputs.
                produced = os.path.join(
                    output_dir, "crownseg_input_Seg", os.path.dirname(key),
                    os.path.splitext(os.path.basename(key))[0] + "_Seg.vtk",
                )
                write_surface(produced, labelled=True)
                records[key] = {"status": "segmented", "input": key,
                                "produced": [produced]}

        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "run_report.json"), "w", encoding="utf-8") as handle:
            json.dump({"tool": "Crown_Seg", "cases": records}, handle)
        return output_dir

    return make


@pytest.fixture
def stub_engine(monkeypatch):
    """Stand in for `predict_landmarks`, recording each pass it is asked for.

    Returns the list of passes, each `[(path, key), ...]` exactly as it was
    handed them -- which is what makes the ORDER of the two passes assertable.

    `check_dependencies` is stubbed with it, and belongs with it: it is the
    engine importing its whole lazy stack, and a unit test standing in for the
    engine must stand in for all of it. Left real, every mixed-batch test here
    would load torch, monai and pytorch3d to prove something about ordering.
    """
    passes = []
    monkeypatch.setattr(engine, "check_dependencies", lambda: None)

    def fake(meshes, model_path, networks=None, prediction_ID="Pred",
             output_dir=None, device=None):
        passes.append(list(meshes))
        return {
            "mode": "IOS",
            "device": device,
            "prediction_ID": prediction_ID,
            # As the real engine does: the bundle the checkpoints came from.
            "model_bundle": "ALI_IOS_Models",
            "networks": ["Occlusal"],
            "landmarks_without_model": [],
            "models_unrecognized": [],
            "cases": {
                key: {
                    "input": os.path.basename(path),
                    "status": "ok",
                    "landmarks_found": ["UR1O"],
                    "landmarks_failed": {},
                    "jaws_without_model": {},
                    "produced": [],
                    "duration_seconds": 0.1,
                }
                for path, key in meshes
            },
            "summary": {"total": len(meshes), "processed": len(meshes), "failed": 0},
            "duration_seconds": 0.1,
        }

    monkeypatch.setattr(engine, "predict_landmarks", fake)
    return passes


def a_mixed_cohort(tmp_path):
    """Two meshes already labelled, one raw, in one folder."""
    write_surface(tmp_path / "cohort" / "a_ready.vtk", labelled=True)
    write_surface(tmp_path / "cohort" / "b_ready.vtk", labelled=True)
    write_surface(tmp_path / "cohort" / "c_raw.vtk", labelled=False)
    return str(tmp_path / "cohort")


def a_bundle(tmp_path):
    """Stands in for the models directory the server hands the tool."""
    bundle = tmp_path / "models"
    bundle.mkdir(parents=True, exist_ok=True)
    return str(bundle)


# ---------------------------------------------------------------------------
# The order, which is the whole point
# ---------------------------------------------------------------------------

def test_the_ready_meshes_are_processed_before_the_raw_ones(tmp_path, stub_engine):
    """Two passes, ready ones first, and the segmentation strictly between them.

    Not one pass over a batch that was segmented up front: a run cancelled or
    timed out during the segmentation has to leave the cheap results behind,
    and that is only true if they were written before it started.
    """
    sup = FakeSup(tmp_path, {"Crown_Seg": crown_seg()})

    run(input=Path(a_mixed_cohort(tmp_path)), model=Path(a_bundle(tmp_path)),
        output_dir=tmp_path / "out", sup=sup)

    assert len(stub_engine) == 2
    assert [key for _path, key in stub_engine[0]] == ["a_ready.vtk", "b_ready.vtk"]
    assert [key for _path, key in stub_engine[1]] == ["c_raw.vtk"]
    assert [tool for tool, _params in sup.calls] == ["Crown_Seg"]


def test_a_fully_labelled_batch_never_reaches_for_the_other_tool(tmp_path, stub_engine):
    """One pass and no supervised call: the common case must cost nothing."""
    write_surface(tmp_path / "cohort" / "arch.vtk", labelled=True)
    sup = FakeSup(tmp_path, {"Crown_Seg": crown_seg()})

    run(input=tmp_path / "cohort", model=Path(a_bundle(tmp_path)),
        output_dir=tmp_path / "out", sup=sup)

    assert len(stub_engine) == 1
    assert sup.calls == []


def test_crown_seg_is_handed_the_models_directory_this_tool_was_given(tmp_path, stub_engine):
    """The server fills a hosted-model argument with the whole models folder and
    each engine recognises its own weights inside; passing it straight on is
    what lets a nested call find the crown checkpoint without this tool knowing
    where DATA/ lives, which it must not."""
    bundle = a_bundle(tmp_path)
    staged = []
    sup = FakeSup(tmp_path, {"Crown_Seg": crown_seg(seen=staged)})

    run(input=Path(a_mixed_cohort(tmp_path)), model=Path(bundle),
        output_dir=tmp_path / "out", sup=sup)

    _tool, params = sup.calls[0]
    assert params["model"] == bundle
    # Only the raw mesh was sent, under its own key: the two that were ready
    # must not be paid for twice.
    assert sorted(staged) == ["c_raw.vtk"]


def test_a_mesh_keeps_its_own_name_through_the_segmentation(tmp_path, stub_engine):
    """Crown_Seg names its output `<stem>_Seg.vtk` and the engine names the
    markups file after the mesh it read, so without renaming, a mesh segmented
    here would produce `c_raw_Seg_lm_Pred.mrk.json` where the same mesh sent
    already labelled produces `c_raw_lm_Pred.mrk.json`. ASO and AREG pair
    landmarks to scans by that stem."""
    sup = FakeSup(tmp_path, {"Crown_Seg": crown_seg()})

    run(input=Path(a_mixed_cohort(tmp_path)), model=Path(a_bundle(tmp_path)),
        output_dir=tmp_path / "out", sup=sup)

    (path, key) = stub_engine[1][0]
    assert key == "c_raw.vtk"
    assert os.path.basename(path) == "c_raw.vtk"


def test_the_run_report_names_the_meshes_segmented_on_the_fly(tmp_path, stub_engine):
    """How a mesh got its labels is the report's to say. A clinician reading
    landmarks that were placed on a segmentation nobody checked has to be able
    to find out that is what happened."""
    output = tmp_path / "out"
    sup = FakeSup(tmp_path, {"Crown_Seg": crown_seg()})

    run(input=Path(a_mixed_cohort(tmp_path)), model=Path(a_bundle(tmp_path)),
        output_dir=output, sup=sup)

    report = json.loads((output / dispatch.REPORT_NAME).read_text())
    assert report["segmented_on_the_fly"] == ["c_raw.vtk"]
    assert sorted(report["cases"]) == ["a_ready.vtk", "b_ready.vtk", "c_raw.vtk"]
    assert report["summary"] == {"total": 3, "processed": 3, "failed": 0}


def test_the_field_is_there_even_when_nothing_needed_segmenting(tmp_path, stub_engine):
    """Empty is the normal case, so a client reads one field rather than
    testing whether it exists."""
    write_surface(tmp_path / "cohort" / "arch.vtk", labelled=True)
    output = tmp_path / "out"

    run(input=tmp_path / "cohort", model=Path(a_bundle(tmp_path)), output_dir=output,
        sup=FakeSup(tmp_path, {"Crown_Seg": crown_seg()}))

    report = json.loads((output / dispatch.REPORT_NAME).read_text())
    assert report["segmented_on_the_fly"] == []


# ---------------------------------------------------------------------------
# With no supervisor, nothing changes
# ---------------------------------------------------------------------------

def test_with_no_supervisor_an_unlabelled_mesh_still_refuses_the_batch(tmp_path, stub_engine):
    """A direct call, a checkout, a server too old to inject a supervisor.

    The answer has to be the one it always was -- the 422 naming the tool to
    run -- and it has to come BEFORE any weights are loaded, which is what
    `require_labels` was doing and still does.
    """
    with pytest.raises(ToolInputError) as raised:
        run(input=Path(a_mixed_cohort(tmp_path)), model=Path(a_bundle(tmp_path)),
            output_dir=tmp_path / "out")

    message = str(raised.value)
    assert "Crown_Seg" in message
    assert "1 of 3" in message
    assert stub_engine == [], "the batch was refused after inference had started"


def test_with_no_supervisor_a_labelled_batch_is_unaffected(tmp_path, stub_engine):
    write_surface(tmp_path / "cohort" / "arch.vtk", labelled=True)

    run(input=tmp_path / "cohort", model=Path(a_bundle(tmp_path)),
        output_dir=tmp_path / "out")

    assert [key for _path, key in stub_engine[0]] == ["arch.vtk"]


# ---------------------------------------------------------------------------
# Failure costs the mesh it belongs to, never the batch
# ---------------------------------------------------------------------------

def test_one_unsegmentable_mesh_does_not_sink_the_batch(tmp_path, stub_engine):
    write_surface(tmp_path / "cohort" / "a_ready.vtk", labelled=True)
    write_surface(tmp_path / "cohort" / "b_raw.vtk", labelled=False)
    write_surface(tmp_path / "cohort" / "c_raw.vtk", labelled=False)
    output = tmp_path / "out"
    sup = FakeSup(tmp_path, {"Crown_Seg": crown_seg(failed=("b_raw.vtk",))})

    run(input=tmp_path / "cohort", model=Path(a_bundle(tmp_path)), output_dir=output,
        sup=sup)

    report = json.loads((output / dispatch.REPORT_NAME).read_text())
    assert report["segmented_on_the_fly"] == ["c_raw.vtk"]
    assert report["summary"] == {"total": 3, "processed": 2, "failed": 1}
    # Reported as that one mesh failing, with the reason its own tool gave.
    assert report["cases"]["b_raw.vtk"]["status"] == "failed"
    assert "no output" in report["cases"]["b_raw.vtk"]["error"]
    assert report["cases"]["c_raw.vtk"]["status"] == "ok"


def test_a_failed_segmentation_keeps_the_landmarks_already_written(tmp_path, stub_engine):
    """The whole call failing -- no crown checkpoint in the bundle, the engine's
    extra not installed, the process killed -- must not throw away the landmarks
    the first pass already wrote. That is the ordering's entire purpose."""
    output = tmp_path / "out"

    def explode(params):
        raise RuntimeError("Crown_Seg: no crown-segmentation checkpoint in 'ALI_IOS_Models'")

    sup = FakeSup(tmp_path, {"Crown_Seg": explode})

    run(input=Path(a_mixed_cohort(tmp_path)), model=Path(a_bundle(tmp_path)),
        output_dir=output, sup=sup)

    report = json.loads((output / dispatch.REPORT_NAME).read_text())
    assert report["summary"] == {"total": 3, "processed": 2, "failed": 1}
    assert report["segmented_on_the_fly"] == []
    # The reason travels to the mesh it cost, rather than to the server log.
    assert "no crown-segmentation checkpoint" in report["cases"]["c_raw.vtk"]["error"]


def test_a_batch_that_is_only_unsegmentable_meshes_fails_with_the_reason(tmp_path, stub_engine):
    """Nothing carried labels and nothing could be given any, so there is no
    partial result to keep and the segmentation is the whole story. Failing
    with "produced no landmarks" and no reason is what this avoids."""
    write_surface(tmp_path / "cohort" / "raw.vtk", labelled=False)

    def explode(params):
        raise RuntimeError("Crown_Seg: the segmentation extra is not installed")

    with pytest.raises(RuntimeError) as raised:
        run(input=tmp_path / "cohort", model=Path(a_bundle(tmp_path)),
            output_dir=tmp_path / "out",
            sup=FakeSup(tmp_path, {"Crown_Seg": explode}))

    assert "the segmentation extra is not installed" in str(raised.value)


def test_a_pass_that_produces_nothing_does_not_cost_the_other(tmp_path, monkeypatch):
    """`predict_landmarks` raises when NO mesh of the batch it was given
    produced a landmark. With two passes that must not end the run: the meshes
    segmented here still have landmarks worth returning."""
    output = tmp_path / "out"
    passes = []

    def fake(meshes, model_path, networks=None, prediction_ID="Pred",
             output_dir=None, device=None):
        passes.append(list(meshes))
        if len(passes) == 1:
            raise RuntimeError("ALI produced no landmarks for any mesh. First error: x")
        return {
            "mode": "IOS", "device": device, "prediction_ID": prediction_ID,
            "networks": ["Occlusal"], "landmarks_without_model": [],
            "models_unrecognized": [],
            "cases": {key: {"input": os.path.basename(path), "status": "ok",
                            "landmarks_found": ["UR1O"], "landmarks_failed": {},
                            "jaws_without_model": {}, "produced": [],
                            "duration_seconds": 0.1}
                      for path, key in meshes},
            "summary": {"total": len(meshes), "processed": len(meshes), "failed": 0},
            "duration_seconds": 0.1,
        }

    monkeypatch.setattr(engine, "check_dependencies", lambda: None)
    monkeypatch.setattr(engine, "predict_landmarks", fake)

    run(input=Path(a_mixed_cohort(tmp_path)), model=Path(a_bundle(tmp_path)),
        output_dir=output, sup=FakeSup(tmp_path, {"Crown_Seg": crown_seg()}))

    report = json.loads((output / dispatch.REPORT_NAME).read_text())
    assert report["cases"]["c_raw.vtk"]["status"] == "ok"
    assert report["summary"]["total"] == 1, "only the pass that produced anything is reported"


def test_the_landmark_stack_is_checked_before_a_segmentation_is_paid_for(tmp_path, monkeypatch):
    """A batch where NOTHING arrived labelled has not reached the engine yet, so
    a venv without pytorch3d would segment the whole cohort and only then say
    the landmark engine was never installed."""
    from sadt_ali_ios.errors import ToolUnavailableError

    write_surface(tmp_path / "cohort" / "raw.vtk", labelled=False)
    order = []

    def unavailable():
        order.append("check")
        raise ToolUnavailableError("ALI's IOS engine needs pytorch3d.")

    monkeypatch.setattr(engine, "check_dependencies", unavailable)
    sup = FakeSup(tmp_path, {"Crown_Seg": lambda params: order.append("segment")})

    with pytest.raises(ToolUnavailableError):
        run(input=tmp_path / "cohort", model=Path(a_bundle(tmp_path)),
            output_dir=tmp_path / "out", sup=sup)

    assert order == ["check"], "the cohort was segmented for an engine that cannot run"


# ---------------------------------------------------------------------------
# The shape the generator and the server read
# ---------------------------------------------------------------------------

def test_the_supervisor_is_keyword_only_and_unannotated():
    """That shape is the marker: `describe.py` keeps it out of the schema and
    publishes `"supervisor": true` instead, so a runner that cannot inject one
    refuses the tool rather than calling it and failing halfway."""
    parameter = inspect.signature(run).parameters["sup"]

    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.annotation is inspect.Parameter.empty
    assert parameter.default is None


def test_the_tool_it_calls_is_named_by_a_literal_the_generator_can_read():
    """`describe.py` derives the schema's `calls` by READING this source, and
    the server refuses to start when a declared call names a tool it does not
    serve. A name it cannot see is a name nothing checks."""
    source = open(dispatch.__file__, encoding="utf-8").read()

    # The two spellings the generator resolves: a module-level string constant,
    # and a `sup.run()` whose first argument is that constant.
    assert re.search(r'^CROWN_TOOL = "%s"$' % dispatch.CROWN_TOOL, source, re.M)
    assert re.search(r"sup\.run\(\s*CROWN_TOOL\b", source)


def test_one_unreadable_mesh_does_not_sink_the_batch(tmp_path, stub_engine):
    """Telling a labelled mesh from a raw one means reading every mesh in the
    batch, and that read is the first thing a corrupt file breaks -- before a
    single landmark has been placed. It is put with the ones needing labels, so
    Crown_Seg fails it alone and says why, and the cohort still returns."""
    cohort = tmp_path / "cohort"
    write_surface(cohort / "good.vtk", labelled=True)
    (cohort / "corrupt.vtk").write_text("# vtk DataFile Version 3.0\n", encoding="utf-8")

    output = tmp_path / "out"
    sup = FakeSup(tmp_path, {"Crown_Seg": crown_seg(failed=["corrupt.vtk"])})

    run(input=cohort, model=Path(a_bundle(tmp_path)), output_dir=output, sup=sup)

    report = json.loads((output / dispatch.REPORT_NAME).read_text())
    assert report["cases"]["good.vtk"]["status"] == "ok"
    assert report["cases"]["corrupt.vtk"]["status"] == "failed"
    # The engine was asked about the readable mesh, and only about it.
    assert [key for _path, key in stub_engine[0]] == ["good.vtk"]


def test_the_report_names_the_bundle_the_weights_came_from(tmp_path, stub_engine):
    """`model` is handed the whole of `DATA/ALI/models/` when the caller names
    no bundle, so naming the bundle after the ARGUMENT reports "models" -- the
    one thing this field exists not to say. The engine names it from a
    checkpoint it actually loaded, and that answer must survive."""
    output = tmp_path / "out"
    sup = FakeSup(tmp_path, {"Crown_Seg": crown_seg()})

    run(input=Path(a_mixed_cohort(tmp_path)), model=Path(a_bundle(tmp_path)),
        output_dir=output, sup=sup)

    report = json.loads((output / dispatch.REPORT_NAME).read_text())
    assert report["model_bundle"] == "ALI_IOS_Models"
