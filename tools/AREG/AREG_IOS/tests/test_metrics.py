"""What the registration measured, shown to the person who started it.

The ICP aligned the two patches and said nothing about how well. These lines
reach the user's console as `metric` lines: the cost the ICP minimises where
it started and where it ended, the surface distance it left, and with the
mucogingival band how far each landmark still is from its T1 position.
"""

import json

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk

from sadt_areg_common import catalogs
from sadt_areg_ios import dispatch, icp, progress, surfaces
from sadt_areg_ios import pipeline as ios_pipeline

ENTRY = {
    "status": "ok",
    "metrics": {
        "icp": {"cost": "point-to-point",
                "cost_milestones": [(0, 1.2345), (1000, 0.0123)],
                "iterations": 1000,
                "surface_mean_mm": 0.0871, "surface_std_mm": 0.0642,
                "surface_max_mm": 0.41, "points": 5000},
        "landmark_displacement": {"labels": ["LL6MG", "L0MG", "LR6MG"],
                                  "distances_mm": [0.5, 0.5, 2.0]},
    },
}


def _bowl(rotation_deg=0.0, shift=(0.0, 0.0, 0.0)):
    """Points of an asymmetric bowl, rotated about z and shifted, as (N, 3)."""
    x, y = np.meshgrid(np.arange(-15.0, 16.0), np.arange(-10.0, 11.0))
    points = np.column_stack([x.ravel(), y.ravel(),
                              0.05 * x.ravel() ** 2 + 0.02 * y.ravel() ** 2
                              + 0.01 * x.ravel() * y.ravel()])
    angle = np.radians(rotation_deg)
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.0],
                         [np.sin(angle), np.cos(angle), 0.0],
                         [0.0, 0.0, 1.0]])
    return points @ rotation.T + np.asarray(shift)


def _cloud(points):
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(np.ascontiguousarray(points, dtype=np.float64), deep=True))
    cloud = vtk.vtkPolyData()
    cloud.SetPoints(vtk_points)
    vertices = vtk.vtkCellArray()
    for index in range(len(points)):
        vertices.InsertNextCell(1)
        vertices.InsertCellPoint(index)
    cloud.SetVerts(vertices)
    return cloud


def test_each_quantity_on_its_own_line():
    assert dispatch.metric_lines(ENTRY) == [
        "ICP cost, mean squared point-to-point distance: 1.2345 at the start > 0.0123 at the end mm²",
        "surface distance after registration: 0.09 ± 0.06 mm",
        "landmark displacement T1 to T2 after registration: 1.00 ± 0.71 mm, "
        "max 2.00 mm (LR6MG), 3 pairs",
    ]


def test_without_landmarks_there_is_no_landmark_line():
    entry = {"metrics": {"icp": ENTRY["metrics"]["icp"]}}
    lines = dispatch.metric_lines(entry)
    assert len(lines) == 2
    assert not any("landmark" in line for line in lines)


def test_a_failed_subject_has_no_metric_line():
    assert dispatch.metric_lines({"status": "failed", "reason": "no point"}) == []


def test_no_line_carries_an_em_dash_or_an_arrow_or_runs_long():
    for line in dispatch.metric_lines(ENTRY):
        assert "—" not in line and "->" not in line and "→" not in line
        assert len(line) < progress.MAX_MESSAGE


def test_the_alignment_reports_its_cost_where_it_started_and_ended():
    """A bowl turned by 5 degrees: the centroids match, the shapes do not, so
    the cost starts well above zero and the ICP takes it to nothing."""
    target = _cloud(_bowl())
    source = _cloud(_bowl(rotation_deg=5.0, shift=(2.0, -1.0, 0.5)))
    alignment = {}
    matrix = icp.align(source, target, stats=alignment)
    measured = icp.measure(source, target, matrix, iterations=alignment["iterations"])

    assert alignment["iterations"] == icp.MAX_ITERATIONS
    (first_i, first), (last_i, last) = measured["cost_milestones"]
    assert (first_i, last_i) == (0, icp.MAX_ITERATIONS)
    assert first > 0.1
    assert last < 0.01
    assert measured["surface_mean_mm"] < 0.1
    assert measured["points"] == source.GetNumberOfPoints()


def test_the_surface_distance_is_the_closest_point_distance():
    """A cloud one millimetre above an identical one: every point is 1 mm away."""
    target = _cloud(_bowl())
    measured = icp.measure(_cloud(_bowl(shift=(0.0, 0.0, 1.0))), target, np.eye(4))
    assert np.isclose(measured["surface_mean_mm"], 1.0, atol=1e-6)
    assert np.isclose(measured["surface_std_mm"], 0.0, atol=1e-6)


class _LandmarkPainter:
    """A patch on every point, and landmarks per scan as the MGL painter has."""

    array_name = "Bottom_MGL"

    def __init__(self, landmarks):
        self.landmarks = landmarks

    def landmarks_for(self, scan_path):
        return self.landmarks["T1" if "T1" in scan_path else "T2"]

    def __call__(self, surface, scan_path, where=None):
        array = numpy_to_vtk(np.ones(surface.GetNumberOfPoints(), dtype=np.int64), deep=True)
        array.SetName(self.array_name)
        surface.GetPointData().AddArray(array)
        return surface, None


def test_the_metrics_are_in_the_report_entry(tmp_path):
    """Two timepoints the same up to a translation: the landmarks the same
    translation moved end where they started, one of them 3 mm off."""
    shift = np.array([1.0, 2.0, 0.0])
    for timepoint, offset in (("T1", np.zeros(3)), ("T2", shift)):
        surfaces.write_surface(_cloud(_bowl(shift=offset)),
                               str(tmp_path / timepoint / f"P1_{timepoint}_Lower.vtk"))
    t1 = {"LL6MG": np.array([-10.0, 0.0, 5.0]), "L0MG": np.array([0.0, -8.0, 1.3]),
          "LR6MG": np.array([10.0, 0.0, 5.0])}
    t2 = {label: position + shift for label, position in t1.items()}
    t2["LR6MG"] = t2["LR6MG"] + (0.0, 0.0, 3.0)
    matched = ios_pipeline.pair(str(tmp_path / "T1"), str(tmp_path / "T2"), "Reg",
                                registered_jaw=catalogs.JAW_LOWER, carry_other=False)

    entry = ios_pipeline.register_patient(
        jaws=matched.matched["P1"], painter=_LandmarkPainter({"T1": t1, "T2": t2}),
        registered_jaw=catalogs.JAW_LOWER, output_dir=str(tmp_path / "out"),
        relative_key="P1", suffix="Reg",
    )

    metrics = json.loads(json.dumps(entry["metrics"]))  # what the JSON report holds
    assert metrics["icp"]["surface_mean_mm"] < 0.01
    moved = metrics["landmark_displacement"]
    assert moved["labels"] == ["LL6MG", "L0MG", "LR6MG"]
    assert np.allclose(moved["distances_mm"], [0.0, 0.0, 3.0], atol=0.01)


def test_the_lines_reach_the_user_as_metrics_under_their_subject(tmp_path, monkeypatch):
    stream = tmp_path / "events.jsonl"
    monkeypatch.setenv(progress.VARIABLE, str(stream))

    dispatch.report_metrics("subject 2 of 5", ENTRY)

    records = [json.loads(line) for line in stream.read_text().splitlines()]
    assert {r["level"] for r in records} == {"metric"}
    assert {r["audience"] for r in records} == {"user"}
    assert records[0]["message"] == "subject 2 of 5"
    assert [r["message"] for r in records[1:]] == dispatch.metric_lines(ENTRY)
    assert all(len(r["message"]) <= progress.MAX_MESSAGE for r in records)


def test_nothing_is_sent_for_a_subject_without_metrics(tmp_path, monkeypatch):
    stream = tmp_path / "events.jsonl"
    monkeypatch.setenv(progress.VARIABLE, str(stream))
    dispatch.report_metrics("subject 1 of 1", {"status": "failed"})
    assert not stream.exists() or stream.read_text() == ""


def test_a_run_sends_one_block_per_subject_and_no_name(tmp_path, monkeypatch):
    stream = tmp_path / "events.jsonl"
    monkeypatch.setenv(progress.VARIABLE, str(stream))
    for subject in ("P1", "P2"):
        for timepoint in ("T1", "T2"):
            surfaces.write_surface(_cloud(_bowl()),
                                   str(tmp_path / timepoint / f"{subject}_{timepoint}_Lower.vtk"))
    (tmp_path / "mgl").mkdir()
    landmarks = {"L0MG": np.array([0.0, -8.0, 1.3])}
    monkeypatch.setattr(ios_pipeline, "MGLPainter", lambda root, height, **_: _LandmarkPainter(
        {"T1": landmarks, "T2": landmarks}))

    dispatch.register(t1_path=str(tmp_path / "T1"), t2_path=str(tmp_path / "T2"),
                      automation=catalogs.AUTOMATION_SEMI, ios_patch=catalogs.PATCH_MGL,
                      mgl_landmarks_path=str(tmp_path / "mgl"),
                      output_dir=str(tmp_path / "out"))

    records = [json.loads(line) for line in stream.read_text().splitlines()]
    metrics = [r["message"] for r in records if r.get("level") == "metric"]
    assert metrics[0] == "patient 1/2" and metrics[4] == "patient 2/2"
    assert len(metrics) == 8
    assert not any("P1" in r["message"] or "P2" in r["message"] for r in records)
    report = json.loads((tmp_path / "out" / dispatch.REPORT_NAME).read_text())
    assert all("metrics" in entry for entry in report["patients"].values())
