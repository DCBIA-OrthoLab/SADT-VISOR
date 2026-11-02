"""The metrics spreadsheet: what each orientation measured, kept for a cohort.

Ticking "Metrics spreadsheet (.xlsx + .csv)" writes `ASO_metrics.xlsx` and one
.csv per table beside `ASO_report.json`, built from that report: one row per
scan, one row per scan and point.
"""

import csv
import inspect
import json
import os
import subprocess
import sys
import typing
import zipfile
from pathlib import Path

import numpy as np
import pytest

from sadt_aso import catalogs, dispatch, run, user_metrics
from sadt_aso.errors import ToolInputError

from test_run import _REFERENCE_POINTS, _cbct_case, _cbct_reference

CBCT_REPORT = {
    "modality": catalogs.MODALITY_CBCT,
    "automation": catalogs.AUTOMATION_SEMI,
    "cases": {
        "case_b": {"status": "failed", "reason": "no landmark file for this scan"},
        "case_a": {
            "status": "ok",
            "metrics": {
                "points": list(user_metrics.POINTS_LANDMARK),
                "labels": ["Ba", "N", "S"],
                "left_out": {"RPo": "rejected as an outlier",
                             "Gn": "not in the input landmark file"},
                "stages": [
                    {"after": user_metrics.STEP_THREE_POINT, "centred": True,
                     "distances_mm": [1.0, 2.0, 3.0]},
                    {"after": user_metrics.STEP_ORIENTATION, "centred": True,
                     "distances_mm": [0.5, 0.5, 2.0]},
                ],
                "icp": {"iterations": 1000, "rms_closest_mm": 0.4321},
                "rotation_deg": 17.04,
                "translation_mm": 13.86,
            },
        },
    },
}

IOS_REPORT = {
    "modality": catalogs.MODALITY_IOS,
    "automation": catalogs.AUTOMATION_FULLY,
    "cases": {
        "case_c": {
            "status": "ok",
            "jaws": {
                "Upper": {"status": "ok", "metrics": {
                    "points": list(user_metrics.POINTS_CENTROID),
                    "labels": ["UL1", "UL6", "UR6"],
                    "stages": [
                        {"after": user_metrics.STEP_OCCLUSAL_PLANE,
                         "distances_mm": [4.0, 2.0, 3.0]},
                        {"after": user_metrics.STEP_THREE_POINT,
                         "distances_mm": [1.0, 1.0, 1.0]},
                        {"after": user_metrics.STEP_ORIENTATION,
                         "distances_mm": [0.1, 0.3, 0.2]},
                    ],
                    "icp": {"iterations": 100, "rms_closest_mm": 0.2},
                    "rotation_deg": 12.0, "translation_mm": 6.2,
                }},
                "Lower": {"status": "ok", "metrics": {
                    "moved_with": "Upper", "rotation_deg": 12.0, "translation_mm": 6.2,
                }},
            },
        },
    },
}


def test_the_cases_table_has_one_row_per_scan_with_each_stage_summarised():
    cases = dispatch.metric_tables(CBCT_REPORT)["cases"]
    assert [row["patient"] for row in cases] == ["case_a", "case_b"]
    ok, failed = cases
    assert "jaw" not in ok
    assert ok["modality"] == "CBCT" and ok["status"] == "ok" and ok["error"] is None
    assert ok["points"] == "landmarks" and ok["n_points"] == 3
    assert ok["left_out"] == "Gn, RPo"
    assert np.isclose(ok["three_point_fit_mean_mm"], 2.0)
    assert np.isclose(ok["three_point_fit_std_mm"], np.std([1.0, 2.0, 3.0]))
    assert ok["three_point_fit_max_mm"] == 3.0 and ok["three_point_fit_max_label"] == "S"
    assert np.isclose(ok["orientation_mean_mm"], 1.0)
    assert ok["orientation_max_label"] == "S"
    assert ok["centres_matched"] == "yes"
    assert ok["icp_rms_closest_mm"] == 0.4321
    assert ok["rotation_deg"] == 17.04 and ok["translation_mm"] == 13.86
    assert failed == {"patient": "case_b", "modality": "CBCT", "status": "failed",
                      "error": "no landmark file for this scan"}


def test_the_landmarks_table_has_one_row_per_point_and_the_left_out_ones():
    points = dispatch.metric_tables(CBCT_REPORT)["landmarks"]
    assert [(p["point"], p["in_fit"]) for p in points] == [
        ("Ba", "yes"), ("N", "yes"), ("S", "yes"), ("Gn", "no"), ("RPo", "no")]
    assert points[2]["three_point_fit_mm"] == 3.0 and points[2]["orientation_mm"] == 2.0
    assert points[4]["left_out_because"] == "rejected as an outlier"
    assert "orientation_mm" not in points[4]


def test_an_ios_table_has_a_row_per_jaw_and_the_occlusal_plane_stage_first():
    tables = dispatch.metric_tables(IOS_REPORT)
    upper, lower = tables["cases"]
    assert (upper["jaw"], lower["jaw"]) == ("Upper", "Lower")
    assert upper["points"] == "teeth" and upper["centres_matched"] == "no"
    assert upper["occlusal_plane_max_label"] == "UL1"
    keys = list(upper)
    assert (keys.index("occlusal_plane_mean_mm") < keys.index("three_point_fit_mean_mm")
            < keys.index("orientation_mean_mm"))
    assert lower["moved_with"] == "Upper" and "n_points" not in lower
    assert [p["point"] for p in tables["landmarks"]] == ["UL1", "UL6", "UR6"]
    assert {p["jaw"] for p in tables["landmarks"]} == {"Upper"}


def _orient(tmp_path, **kwargs):
    _cbct_case(tmp_path / "input", key="patient1")
    return run(
        input=tmp_path / "input",
        reference=_cbct_reference(tmp_path),
        modality=catalogs.MODALITY_CBCT,
        automation=catalogs.AUTOMATION_SEMI,
        cbct_landmarks=list(_REFERENCE_POINTS),
        output_dir=tmp_path / "out",
        **kwargs,
    )


def _report(output_dir) -> dict:
    with open(os.path.join(str(output_dir), dispatch.REPORT_NAME), encoding="utf-8") as handle:
        return json.load(handle)


def test_nothing_is_written_when_the_spreadsheet_is_not_ticked(tmp_path):
    output_dir = _orient(tmp_path)
    assert not list(Path(output_dir).glob(catalogs.METRICS_STEM + "*"))
    assert "metrics_files" not in _report(output_dir)


def test_ticked_the_spreadsheet_is_written_beside_the_report(tmp_path):
    output_dir = _orient(tmp_path, outputs=[catalogs.OUTPUT_METRICS])
    report = _report(output_dir)
    assert report["metrics_files"] == [
        "ASO_metrics.xlsx", "ASO_metrics_cases.csv", "ASO_metrics_landmarks.csv"]
    for name in report["metrics_files"]:
        assert os.path.getsize(os.path.join(str(output_dir), name)) > 0

    with open(os.path.join(str(output_dir), "ASO_metrics_cases.csv"),
              encoding="utf-8-sig", newline="") as handle:
        (row,) = list(csv.DictReader(handle))
    assert row["patient"] == "patient1" and row["status"] == "ok"
    assert row["n_points"] == str(len(_REFERENCE_POINTS))
    assert abs(float(row["orientation_max_mm"])) < 1e-3
    assert abs(float(row["rotation_deg"]) - 17.0) < 1e-3

    with open(os.path.join(str(output_dir), "ASO_metrics_landmarks.csv"),
              encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert sorted(r["point"] for r in rows) == sorted(_REFERENCE_POINTS)


def test_the_xlsx_is_a_zip_holding_both_sheets(tmp_path):
    output_dir = _orient(tmp_path, outputs=[catalogs.OUTPUT_METRICS])
    with zipfile.ZipFile(os.path.join(str(output_dir), "ASO_metrics.xlsx")) as archive:
        assert archive.testzip() is None
        names = set(archive.namelist())
        workbook = archive.read("xl/workbook.xml").decode("utf-8")
        cases = archive.read("xl/worksheets/sheet1.xml").decode("utf-8")
    assert {"[Content_Types].xml", "xl/worksheets/sheet1.xml",
            "xl/worksheets/sheet2.xml"} <= names
    assert 'name="cases"' in workbook and 'name="landmarks"' in workbook
    assert "patient1" in cases and "orientation_mean_mm" in cases


def test_an_unknown_output_is_refused(tmp_path):
    with pytest.raises(ToolInputError, match="outputs"):
        _orient(tmp_path, outputs=["Something else"])


def test_the_published_option_is_the_catalogs_own_and_nothing_is_ticked():
    hint = typing.get_type_hints(run)["outputs"]
    assert list(typing.get_args(typing.get_args(hint)[0])) == list(catalogs.OUTPUT_CHOICES)
    assert inspect.signature(run).parameters["outputs"].default == []
    assert not any(catalogs.OUTPUT_CHOICES.values())


def test_the_option_is_in_the_schema_under_outputs():
    """Out of process, the way the server reads it: describe.py on this tool."""
    root = Path(__file__).resolve().parents[3]
    completed = subprocess.run(
        [sys.executable, str(root / "scripts" / "describe.py"),
         str(Path(__file__).resolve().parents[1])],
        capture_output=True, text=True, timeout=300,
    )
    assert completed.returncode == 0, completed.stderr
    arguments = json.loads(completed.stdout)["arguments"]
    spec = arguments["outputs"]
    assert spec["choices"] == [catalogs.OUTPUT_METRICS]
    assert spec["default"] == [] and not spec["required"]
    assert spec["section"] == arguments["output_suffix"]["section"] == "Outputs"
    assert spec["label"] == "Extra outputs"
    assert spec["note"] == (
        "Saved beside the result. The result and its report are always written.")
    assert set(spec["option_help"]) == {catalogs.OUTPUT_METRICS}
