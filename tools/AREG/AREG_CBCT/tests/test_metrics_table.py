"""The metrics spreadsheet AREG CBCT writes when asked.

Built from the report the run already wrote, so the spreadsheet can never
disagree with it: one row per patient and region. No landmark sheet: this
tool has no landmarks on both timepoints.
"""

import csv
import inspect
import json
import os
import typing
import zipfile

from conftest import cohort
from sadt_areg_cbct import dispatch
from sadt_areg_cbct.layout import LAYOUT
from sadt_areg_common import catalogs

import sadt_areg_cbct

REPORT = {"patients": {"case_a": {"status": "ok", "regions": {
    "CB": {
        "status": "ok",
        "region": "Cranial base",
        "engine": "elastix",
        "registration_cost": {
            "metric": "Mattes mutual information",
            "lower_is_better": True,
            "engine": "elastix",
            "computed_on": "the cranial base mask",
            "resolutions": [
                {"iterations": 40, "milestones": [[0, -0.2], [39, -0.4]]},
                {"iterations": 25, "milestones": [[0, -0.5], [10, -0.62], [24, -0.65]]},
            ],
        },
        "rigid_transform": {"rotation_deg": 3.2, "translation_mm": 1.75},
    },
    "MAND": {"status": "failed", "reason": "no Mandible mask matched"},
}}}}


def test_one_row_per_region_with_what_the_console_showed():
    cranial, mandible = dispatch.metric_tables(REPORT)["cases"]
    assert cranial == {
        "patient": "case_a", "region": "Cranial base", "status": "ok",
        "engine": "elastix", "cost_metric": "Mattes mutual information",
        "cost_computed_on": "the cranial base mask",
        "cost_start": -0.5, "cost_end": -0.65,
        "iterations_finest": 25, "iterations_total": 65,
        "rotation_deg": 3.2, "translation_mm": 1.75,
    }
    # A failure written before any entry existed has no `region` key; the
    # code it is filed under names it.
    assert mandible == {"patient": "case_a", "region": "Mandible", "status": "failed",
                        "error": "no Mandible mask matched"}


def test_there_is_no_landmark_table():
    assert list(dispatch.metric_tables(REPORT)) == ["cases"]


def test_the_option_is_offered_in_the_signature_and_the_layout():
    hint = inspect.signature(sadt_areg_cbct.run).parameters["outputs"].annotation
    (literal,) = typing.get_args(hint)
    assert catalogs.OUTPUT_METRICS in typing.get_args(literal)
    assert catalogs.OUTPUT_METRICS in str(LAYOUT["outputs"])


def _semi(root, outputs):
    return dispatch.register(
        t1_path=os.path.join(str(root), "T1"),
        t2_path=os.path.join(str(root), "T2"),
        t1_masks_path=os.path.join(str(root), "masks"),
        automation=catalogs.AUTOMATION_SEMI,
        regions=["Cranial base"],
        output_dir=os.path.join(str(root), "out"),
        outputs=outputs,
    )


def test_the_spreadsheet_is_written_beside_the_report_when_ticked(tmp_path):
    cohort(tmp_path, size=32)
    run = _semi(tmp_path, [catalogs.OUTPUT_METRICS])
    out = run.output_dir
    expected = ["AREG_metrics.xlsx", "AREG_metrics_cases.csv"]
    assert sorted(run.report["metrics_files"]) == expected
    for name in expected:
        assert os.path.isfile(os.path.join(out, name))
    with open(os.path.join(out, dispatch.REPORT_NAME)) as handle:
        assert sorted(json.load(handle)["metrics_files"]) == expected

    with open(os.path.join(out, "AREG_metrics_cases.csv"), encoding="utf-8-sig") as handle:
        (row,) = list(csv.DictReader(handle))
    assert (row["patient"], row["region"], row["status"]) == ("P1", "Cranial base", "ok")
    assert row["cost_metric"] == "Mattes mutual information"
    assert float(row["cost_end"]) <= float(row["cost_start"])
    assert int(row["iterations_total"]) >= int(row["iterations_finest"]) > 0
    assert row["rotation_deg"] and row["translation_mm"]

    with zipfile.ZipFile(os.path.join(out, "AREG_metrics.xlsx")) as archive:
        names = archive.namelist()
        workbook = archive.read("xl/workbook.xml").decode("utf-8")
    assert "xl/worksheets/sheet1.xml" in names
    assert "xl/worksheets/sheet2.xml" not in names
    assert 'name="cases"' in workbook


def test_nothing_is_written_when_not_ticked(tmp_path):
    cohort(tmp_path, size=32)
    run = _semi(tmp_path, [])
    assert "metrics_files" not in run.report
    assert not [name for name in os.listdir(run.output_dir) if name.startswith("AREG_metrics")]
