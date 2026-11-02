"""The metrics spreadsheet AREG IOS writes when asked.

Built from the report the run already wrote, so the spreadsheet can never
disagree with it: one row per patient, one per landmark.
"""

import inspect
import json
import typing
import zipfile

import numpy as np
import pytest

from sadt_areg_common import catalogs
import sadt_areg_ios
from sadt_areg_ios import dispatch, surfaces
from sadt_areg_ios import pipeline as ios_pipeline
from sadt_areg_ios.layout import LAYOUT

from test_metrics import _LandmarkPainter, _bowl, _cloud

REPORT = {
    "patch": catalogs.PATCH_MGL,
    "patients": {
        "case_a": {
            "status": "ok",
            "metrics": {
                "icp": {"cost": "point-to-point",
                        "cost_milestones": [[0, 1.2345], [1000, 0.0123]],
                        "iterations": 1000,
                        "surface_mean_mm": 0.0871, "surface_std_mm": 0.0642,
                        "surface_max_mm": 0.41, "points": 5000},
                "landmark_displacement": {"labels": ["LL6MG", "L0MG", "LR6MG"],
                                          "distances_mm": [0.5, 0.5, 2.0]},
            },
        },
        "case_b": {"status": "failed", "reason": "no landmark file for this scan"},
    },
}


def test_one_row_per_patient_with_what_the_console_showed():
    registered, failed = dispatch.metric_tables(REPORT)["cases"]
    assert registered == {
        "patient": "case_a", "status": "ok", "region": "mucogingival",
        "icp_cost": "point-to-point",
        "icp_cost_start_mm2": 1.2345, "icp_cost_end_mm2": 0.0123,
        "surface_mean_mm": 0.0871, "surface_std_mm": 0.0642,
        "landmark_displacement_mean_mm": pytest.approx(1.0),
        "landmark_displacement_std_mm": pytest.approx(np.std([0.5, 0.5, 2.0])),
        "landmark_displacement_max_mm": 2.0,
        "landmark_displacement_max_label": "LR6MG",
    }
    assert failed == {"patient": "case_b", "status": "failed", "region": "mucogingival",
                      "error": "no landmark file for this scan"}


def test_one_row_per_landmark():
    assert dispatch.metric_tables(REPORT)["landmarks"] == [
        {"patient": "case_a", "landmark": "LL6MG", "displacement_mm": 0.5},
        {"patient": "case_a", "landmark": "L0MG", "displacement_mm": 0.5},
        {"patient": "case_a", "landmark": "LR6MG", "displacement_mm": 2.0},
    ]


def test_the_palate_has_no_landmark_row():
    entry = {"status": "ok", "metrics": {"icp": REPORT["patients"]["case_a"]["metrics"]["icp"]}}
    tables = dispatch.metric_tables({"patch": catalogs.PATCH_PALATE, "patients": {"case_a": entry}})
    assert tables["landmarks"] == []
    (row,) = tables["cases"]
    assert row["region"] == "palate"
    assert not any(column.startswith("landmark_") for column in row)


def _cohort(tmp_path, monkeypatch):
    """Two subjects on the mucogingival band, the second one 3 mm off on LR6MG."""
    shift = np.array([1.0, 2.0, 0.0])
    for subject in ("P1", "P2"):
        for timepoint, offset in (("T1", np.zeros(3)), ("T2", shift)):
            surfaces.write_surface(_cloud(_bowl(shift=offset)),
                                   str(tmp_path / timepoint / f"{subject}_{timepoint}_Lower.vtk"))
    (tmp_path / "mgl").mkdir()
    t1 = {"LL6MG": np.array([-10.0, 0.0, 5.0]), "L0MG": np.array([0.0, -8.0, 1.3]),
          "LR6MG": np.array([10.0, 0.0, 5.0])}
    t2 = {label: position + shift for label, position in t1.items()}
    monkeypatch.setattr(ios_pipeline, "MGLPainter", lambda root, height, **_: _LandmarkPainter(
        {"T1": t1, "T2": t2}))


def _register(tmp_path, outputs):
    return dispatch.register(t1_path=str(tmp_path / "T1"), t2_path=str(tmp_path / "T2"),
                             automation=catalogs.AUTOMATION_SEMI, ios_patch=catalogs.PATCH_MGL,
                             mgl_landmarks_path=str(tmp_path / "mgl"),
                             output_dir=str(tmp_path / "out"), outputs=outputs)


def test_nothing_is_written_unless_the_option_is_ticked(tmp_path, monkeypatch):
    _cohort(tmp_path, monkeypatch)
    _register(tmp_path, outputs=[])
    out = tmp_path / "out"
    assert not list(out.glob(catalogs.METRICS_STEM + "*"))
    assert "metrics_files" not in json.loads((out / dispatch.REPORT_NAME).read_text())


def test_the_ticked_option_writes_the_spreadsheet_beside_the_report(tmp_path, monkeypatch):
    _cohort(tmp_path, monkeypatch)
    _register(tmp_path, outputs=[catalogs.OUTPUT_METRICS])
    out = tmp_path / "out"
    report = json.loads((out / dispatch.REPORT_NAME).read_text())

    stem = catalogs.METRICS_STEM
    assert sorted(report["metrics_files"]) == sorted(
        [f"{stem}.xlsx", f"{stem}_cases.csv", f"{stem}_landmarks.csv"])
    assert all((out / name).is_file() for name in report["metrics_files"])

    cases = (out / f"{stem}_cases.csv").read_text(encoding="utf-8-sig").splitlines()
    assert cases[0].split(",")[:3] == ["patient", "status", "region"]
    assert [line.split(",")[0] for line in cases[1:]] == ["P1", "P2"]
    landmarks = (out / f"{stem}_landmarks.csv").read_text(encoding="utf-8-sig").splitlines()
    assert landmarks[0] == "patient,landmark,displacement_mm"
    assert len(landmarks) == 1 + 2 * 3

    with zipfile.ZipFile(out / f"{stem}.xlsx") as archive:
        assert archive.testzip() is None
        workbook = archive.read("xl/workbook.xml").decode()
        assert 'name="cases"' in workbook and 'name="landmarks"' in workbook
        assert {"xl/worksheets/sheet1.xml", "xl/worksheets/sheet2.xml"} <= set(archive.namelist())


def test_run_hands_the_option_to_the_engine(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(sadt_areg_ios, "main", lambda **kwargs: seen.update(kwargs))
    sadt_areg_ios.run(t1=tmp_path, t2=tmp_path, output_dir=tmp_path / "out",
                      outputs=[catalogs.OUTPUT_METRICS])
    assert seen["outputs"] == [catalogs.OUTPUT_METRICS]


def test_the_option_is_offered_and_nothing_is_ticked():
    parameter = inspect.signature(sadt_areg_ios.run).parameters["outputs"]
    (choices,) = typing.get_args(parameter.annotation)
    assert typing.get_args(choices) == (catalogs.OUTPUT_METRICS,)
    assert parameter.default == []
    assert "outputs:" in inspect.getdoc(sadt_areg_ios.run)
    assert LAYOUT["outputs"] == catalogs.outputs_layout("Outputs", catalogs.OUTPUT_METRICS)
