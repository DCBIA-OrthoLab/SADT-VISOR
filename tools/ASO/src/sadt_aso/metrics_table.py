"""A run's metrics as a spreadsheet: one .xlsx with a sheet per table, and a .csv each.

What a registration or an orientation measured is printed to the clinician's
console as it happens; this is the same numbers kept for later, one row per
case for a cohort's table and one row per landmark for the statistics. The
.xlsx opens in Excel as it is; the .csv files are the same tables for R,
pandas or SPSS.

Standard library only. An .xlsx is a zip of a handful of XML parts, and
writing those few directly costs less than adding a spreadsheet package to
every tool's lock (and to its vulnerability scan) for two sheets of numbers.

This file is vendored, identical, into each tool that writes the table (see
scripts/tests/test_vendored_copies.py): change every copy.
"""

import csv
import os
import re
import zipfile
from xml.sax.saxutils import escape

# Characters XML 1.0 does not allow at all; a file name can carry them.
_INVALID_XML = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f]")


def write(directory: str, stem: str, tables: dict) -> list:
    """Write `tables` as `<stem>.xlsx` plus `<stem>_<sheet>.csv`; return the paths.

    `tables` is `{sheet name: [row dict, ...]}`, in the order the sheets
    should appear. The columns of a sheet are the keys of its rows, in the
    order they first appear, so a row may leave a column out (an arch without
    an ICP has no surface distance) and its cell is left empty. A sheet with
    no row is skipped.
    """
    os.makedirs(directory, exist_ok=True)
    sheets = [(name, rows, _columns(rows)) for name, rows in tables.items() if rows]
    if not sheets:
        return []
    written = []
    for name, rows, columns in sheets:
        path = os.path.join(directory, "{}_{}.csv".format(stem, name))
        with open(path, "w", newline="", encoding="utf-8-sig") as handle:
            # utf-8-sig: the byte-order mark is what makes Excel read "±" and
            # accented names as UTF-8 instead of the locale's code page.
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({key: _csv_value(row.get(key)) for key in columns})
        written.append(path)
    path = os.path.join(directory, stem + ".xlsx")
    _write_xlsx(path, sheets)
    written.insert(0, path)
    return written


def _columns(rows) -> list:
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    return columns


def _csv_value(value):
    if value is None:
        return ""
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return ""
        return repr(round(value, 6))
    return value


def _column_letter(index: int) -> str:
    letters = ""
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters


def _cell(reference: str, value, header: bool = False) -> str:
    style = ' s="1"' if header else ""
    if value is None or value == "":
        return ""
    if isinstance(value, bool):
        value = "yes" if value else "no"
    if isinstance(value, (int, float)):
        if value != value or value in (float("inf"), float("-inf")):
            return ""  # NaN and infinities are not numbers a sheet can hold
        return '<c r="{}"{}><v>{}</v></c>'.format(reference, style, repr(value))
    text = escape(_INVALID_XML.sub("", str(value)))
    return '<c r="{}" t="inlineStr"{}><is><t xml:space="preserve">{}</t></is></c>'.format(
        reference, style, text)


def _sheet_xml(rows, columns) -> str:
    lines = []
    header = "".join(_cell("{}1".format(_column_letter(i)), name, header=True)
                     for i, name in enumerate(columns))
    lines.append('<row r="1">{}</row>'.format(header))
    for number, row in enumerate(rows, start=2):
        cells = "".join(_cell("{}{}".format(_column_letter(i), number), row.get(name))
                        for i, name in enumerate(columns))
        lines.append('<row r="{}">{}</row>'.format(number, cells))
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" '
            'activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>'
            '<sheetData>{}</sheetData></worksheet>'.format("".join(lines)))


def _write_xlsx(path: str, sheets) -> None:
    main = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    rel = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    package = "http://schemas.openxmlformats.org/package/2006/relationships"
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        + "".join('<Override PartName="/xl/worksheets/sheet{}.xml" ContentType="application/'
                  'vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'.format(i)
                  for i in range(1, len(sheets) + 1))
        + '</Types>')
    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="{}"><Relationship Id="rId1" Type="{}/officeDocument" '
        'Target="xl/workbook.xml"/></Relationships>'.format(package, rel))
    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="{}" xmlns:r="{}"><sheets>{}</sheets></workbook>'.format(
            main, rel, "".join('<sheet name="{}" sheetId="{}" r:id="rId{}"/>'.format(
                escape(name[:31]), i, i) for i, (name, _rows, _columns) in enumerate(sheets, 1))))
    workbook_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="{}">{}<Relationship Id="rId{}" Type="{}/styles" '
        'Target="styles.xml"/></Relationships>'.format(
            package, "".join('<Relationship Id="rId{}" Type="{}/worksheet" '
                             'Target="worksheets/sheet{}.xml"/>'.format(i, rel, i)
                             for i in range(1, len(sheets) + 1)),
            len(sheets) + 1, rel))
    # Two cell formats: the default, and bold for the header row.
    styles = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="{}"><fonts count="2"><font><sz val="11"/><name val="Calibri"/></font>'
        '<font><b/><sz val="11"/><name val="Calibri"/></font></fonts>'
        '<fills count="2"><fill><patternFill patternType="none"/></fill>'
        '<fill><patternFill patternType="gray125"/></fill></fills>'
        '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="2"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
        '<xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/></cellXfs>'
        '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
        '</styleSheet>'.format(main))
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", root_rels)
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        archive.writestr("xl/styles.xml", styles)
        for i, (_name, rows, columns) in enumerate(sheets, 1):
            archive.writestr("xl/worksheets/sheet{}.xml".format(i), _sheet_xml(rows, columns))
