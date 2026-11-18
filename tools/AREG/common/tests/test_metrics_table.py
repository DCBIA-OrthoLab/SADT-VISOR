"""The metrics spreadsheet: an .xlsx Excel opens and the same tables as .csv.

Written with the standard library alone, so what is pinned here is the part a
spreadsheet package would otherwise guarantee: a valid workbook, one sheet per
table, the header first, numbers as numbers, and text a file name could carry
escaped rather than breaking the XML.
"""

import csv
import re
import zipfile

from sadt_areg_common import metrics_table


def _sheet(path, number):
    with zipfile.ZipFile(path) as archive:
        return archive.read("xl/worksheets/sheet{}.xml".format(number)).decode("utf-8")


def test_one_workbook_and_one_csv_per_table(tmp_path):
    written = metrics_table.write(str(tmp_path), "AREG_metrics", {
        "cases": [{"patient": "a", "mean_mm": 1.5}],
        "landmarks": [{"patient": "a", "landmark": "UR1O", "after_icp_mm": 0.25}],
    })
    assert [p.rsplit("/", 1)[1] for p in written] == [
        "AREG_metrics.xlsx", "AREG_metrics_cases.csv", "AREG_metrics_landmarks.csv"]
    with zipfile.ZipFile(written[0]) as archive:
        names = set(archive.namelist())
        workbook = archive.read("xl/workbook.xml").decode("utf-8")
    assert {"xl/worksheets/sheet1.xml", "xl/worksheets/sheet2.xml", "xl/styles.xml"} <= names
    assert re.findall(r'<sheet name="([^"]+)"', workbook) == ["cases", "landmarks"]


def test_numbers_stay_numbers_and_text_is_escaped(tmp_path):
    path = metrics_table.write(str(tmp_path), "m", {
        "cases": [{"patient": "a<b>&c\x01", "mean_mm": 1.5, "n": 3, "ok": True}]})[0]
    sheet = _sheet(path, 1)
    assert "<v>1.5</v>" in sheet and "<v>3</v>" in sheet
    assert "a&lt;b&gt;&amp;c" in sheet and "\x01" not in sheet
    assert ">yes<" in sheet


def test_a_missing_value_is_an_empty_cell(tmp_path):
    written = metrics_table.write(str(tmp_path), "m", {"cases": [
        {"patient": "a", "surface_mean_mm": 0.2},
        {"patient": "b", "error": "failed"},
        {"patient": "c", "surface_mean_mm": float("nan")},
    ]})
    with open(written[1], encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == ["patient", "surface_mean_mm", "error"], "columns in first-seen order"
    assert rows[1]["surface_mean_mm"] == "" and rows[2]["surface_mean_mm"] == ""
    assert "nan" not in _sheet(written[0], 1).lower()


def test_an_empty_table_is_no_sheet_and_nothing_at_all_is_no_file(tmp_path):
    written = metrics_table.write(str(tmp_path), "m", {"cases": [{"a": 1}], "landmarks": []})
    assert len(written) == 2
    assert metrics_table.write(str(tmp_path / "none"), "m", {"cases": []}) == []


def test_the_csv_carries_a_byte_order_mark_for_excel(tmp_path):
    written = metrics_table.write(str(tmp_path), "m", {"cases": [{"value": "±"}]})
    assert open(written[1], "rb").read().startswith(b"\xef\xbb\xbf")
