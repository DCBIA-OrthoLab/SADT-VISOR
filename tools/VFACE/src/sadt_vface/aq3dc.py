"""Running a list of measurements over a cohort, and writing the table.

Upstream's AQ3DC: a measurement list is an Excel sheet naming, per row, a kind
of measurement and the landmarks it is taken on. This reads that list, computes
each row for each patient, and writes one table per region.

**A measurement carries a timepoint per landmark, not per row.** `Distance
between 2 points T1 T2` is the distance from a point at T1 to a point at T2 --
which, in an asymmetry assessment, is the distance between a landmark and its
own mirror image after registration. That is the measurement the whole pipeline
exists to make, and it is why the timepoint sits on the point.

What is NOT carried over from upstream: the Qt `keep_sign` checkbox, which the
batch path sets to checked for every measurement and never reads again. Signs
are always meaningful here, which is what `keep_sign=True` means.
"""

import logging
import os
from dataclasses import dataclass

from .errors import ToolInputError
from . import measure

logger = logging.getLogger(__name__)

T1 = "T1"
T2 = "T2"

# The columns upstream writes, in its order. A row fills the ones its kind has
# and carries "x" in the rest: a distance has no yaw, and the post-processing
# reads that "x" as "this row does not answer that question".
COLUMNS = (
    "Patient", "Type of measurement", "Landmarks",
    "R-L Component", "R-L Meaning",
    "A-P Component", "A-P Meaning",
    "S-I Component", "S-I Meaning",
    "3D Distance",
    "Yaw Component", "Yaw Meaning",
    "Pitch Component", "Pitch Meaning",
    "Roll Component", "Roll Meaning",
)

NOT_APPLICABLE = "x"

# How a measurement list spells each kind, with the timepoints appended. These
# are the values a caller puts in the "Type of measurement" column, so they are
# upstream's strings exactly -- a list written for the Slicer module has to work
# here unchanged.
SPELLINGS = {
    "Distance between 2 points T1": (measure.DISTANCE_2_POINTS, (T1, T1)),
    "Distance between 2 points T2": (measure.DISTANCE_2_POINTS, (T2, T2)),
    "Distance between 2 points T1 T2": (measure.DISTANCE_2_POINTS, (T1, T2)),
    "Distance point line T1": (measure.DISTANCE_POINT_LINE, (T1, T1, T1)),
    "Distance point line T2": (measure.DISTANCE_POINT_LINE, (T2, T2, T2)),
    "Angle between 2 lines T1": (measure.ANGLE_2_LINES, (T1, T1, T1, T1)),
    "Angle between 2 lines T2": (measure.ANGLE_2_LINES, (T2, T2, T2, T2)),
    "Angle line T1 and line T2": (measure.ANGLE_LINE_T1_T2, (T1, T1, T2, T2)),
}

_POINTS_EXPECTED = {
    measure.DISTANCE_2_POINTS: 2,
    measure.DISTANCE_POINT_LINE: 3,
    measure.ANGLE_2_LINES: 4,
    measure.ANGLE_LINE_T1_T2: 4,
}


@dataclass(frozen=True)
class Measurement:
    """One row of a measurement list: what to compute, on which points, when."""

    kind: str
    spelling: str
    names: tuple
    times: tuple

    def __post_init__(self):
        if self.kind not in _POINTS_EXPECTED:
            raise ToolInputError(f"'{self.kind}' is not a measurement this tool makes.")
        if len(self.names) != _POINTS_EXPECTED[self.kind]:
            raise ToolInputError(
                f"'{self.spelling}' takes {_POINTS_EXPECTED[self.kind]} landmark(s), "
                f"got {len(self.names)}: {list(self.names)}"
            )

    @property
    def label(self) -> str:
        """The `Landmarks` cell, which is how the post-processing finds this row.

        A distance is written `A - B` and an angle `A-B / C-D`. The two
        spellings are not decoration: `postprocess` rebuilds this string from a
        feature column's name to look the row up, so the separators are part of
        the contract between the two steps.
        """
        if self.kind in (measure.DISTANCE_2_POINTS, measure.DISTANCE_POINT_LINE):
            if self.kind == measure.DISTANCE_2_POINTS:
                return f"{self.names[0]} - {self.names[1]}"
            return f"{self.names[0]} - {self.names[1]}-{self.names[2]}"
        return f"{self.names[0]}-{self.names[1]} / {self.names[2]}-{self.names[3]}"


def _blank_row(patient: str, entry: Measurement) -> dict:
    row = {name: NOT_APPLICABLE for name in COLUMNS}
    row["Patient"] = patient
    row["Type of measurement"] = entry.spelling
    row["Landmarks"] = entry.label
    return row


def read_measurement_list(path: str) -> list:
    """The measurements an Excel sheet asks for.

    Two shapes, both upstream's: `Type of measurement` with `Point 1` and
    `Point 2 / Line`, or with `Line 1` and `Line 2` -- the second naming each
    line as `A-B`, which is split on the dash. Every sheet of the workbook is
    read, so a list split across tabs works.
    """
    import pandas as pd

    try:
        sheets = pd.read_excel(path, sheet_name=None)
    except Exception as exc:  # noqa: BLE001 - pandas raises a family of these
        raise ToolInputError(
            f"'{os.path.basename(path)}' could not be read as a measurement list: {exc}"
        ) from exc

    found = []
    for name, sheet in sheets.items():
        columns = set(sheet.columns)
        if {"Type of measurement", "Point 1", "Point 2 / Line"} <= columns:
            rows = sheet[["Type of measurement", "Point 1", "Point 2 / Line"]]
            for kind, first, second in rows.itertuples(index=False):
                found.append(_build(kind, [first, second], path, name))
        elif {"Type of measurement", "Line 1", "Line 2"} <= columns:
            rows = sheet[["Type of measurement", "Line 1", "Line 2"]]
            for kind, line1, line2 in rows.itertuples(index=False):
                found.append(_build(
                    kind, str(line1).split("-") + str(line2).split("-"), path, name
                ))
        else:
            logger.warning(
                "VFACE: a sheet of the measurement list names neither "
                "'Point 1'/'Point 2 / Line' nor 'Line 1'/'Line 2'; it is skipped"
            )

    found = [entry for entry in found if entry is not None]
    if not found:
        raise ToolInputError(
            f"'{os.path.basename(path)}' holds no measurement this tool can make. A "
            "sheet needs a 'Type of measurement' column beside either "
            "'Point 1'/'Point 2 / Line' or 'Line 1'/'Line 2'."
        )
    return found


def _build(spelling, names, path, sheet):
    """One `Measurement`, or None when the row cannot be one.

    A bad row is dropped with a reason rather than ending the read: a
    measurement list is hand-written, and one typo in row 40 must not cost the
    other 39.
    """
    spelling = str(spelling).strip()
    if spelling not in SPELLINGS:
        logger.warning("VFACE: '%s' is not a measurement this tool makes; that row "
                       "of the list is skipped", spelling)
        return None

    kind, times = SPELLINGS[spelling]
    cleaned = [str(name).strip() for name in names if str(name).strip() not in ("", "nan")]
    try:
        return Measurement(kind=kind, spelling=spelling, names=tuple(cleaned), times=times)
    except ToolInputError as exc:
        logger.warning("VFACE: a row of the measurement list is skipped -- %s", exc)
        return None


def landmarks_needed(measurements) -> list:
    """Every landmark the list names, once, in a stable order.

    This is what the landmark tool is asked for. It spawns one search agent per
    landmark at about a minute each, so asking for a region's whole catalogue
    instead would spend hours finding points no measurement reads.

    Midpoints are expanded to the two landmarks they join: `Mid_ROr_LOr` is not
    a point anything predicts, it is one this tool computes.
    """
    wanted = []
    for entry in measurements:
        for name in entry.names:
            for part in _sources_of(name):
                if part not in wanted:
                    wanted.append(part)
    return wanted


def _sources_of(name: str) -> tuple:
    """The predicted landmarks a name is built from. A midpoint names two."""
    if name.startswith("Mid_"):
        parts = [part for part in name.split("_")[1:] if part]
        return tuple(parts) if parts else (name,)
    return (name,)


def with_midpoints(points: dict, measurements) -> dict:
    """The patient's points, plus every midpoint the list asks for.

    A midpoint is the mean of the two landmarks it joins, and it is computed
    rather than predicted -- nothing places a point between the two orbitals.
    Absent when either of its two is.
    """
    import numpy as np

    complete = dict(points)
    for entry in measurements:
        for name in entry.names:
            if not name.startswith("Mid_") or name in complete:
                continue
            sources = _sources_of(name)
            if len(sources) >= 2 and all(source in complete for source in sources):
                complete[name] = list(
                    np.mean([complete[source] for source in sources], axis=0)
                )
    return complete


def compute_one(patient: str, positions: dict, entry: Measurement) -> dict:
    """One row: the measurement, for one patient, from `{timepoint: {label: xyz}}`.

    Raises `MeasurementError` when the landmarks are there but the geometry is
    degenerate, and `KeyError` when one of them is not there at all. The caller
    separates the two: a missing landmark is a gap in the cohort, a degenerate
    one is a mistake in the list.
    """
    points = [positions[time][name] for name, time in zip(entry.names, entry.times)]
    row = _blank_row(patient, entry)

    if entry.kind == measure.DISTANCE_2_POINTS:
        lr, ap, si, norm = measure.distance_between_points(points[0], points[1])
    elif entry.kind == measure.DISTANCE_POINT_LINE:
        lr, ap, si, norm = measure.distance_point_to_line(points[0], points[1], points[2])
    else:
        yaw, pitch, roll = measure.angles_between_lines(*points)
        row["Yaw Component"] = abs(yaw)
        row["Pitch Component"] = abs(pitch)
        row["Roll Component"] = abs(roll)

        if entry.kind == measure.ANGLE_LINE_T1_T2:
            # One anatomy seen at two times: what a sign says is which way it
            # turned, not which way it points.
            meanings = measure.cross_timepoint_angle_meanings(yaw, pitch, roll)
        elif all(measure.is_dental(name) for name in entry.names):
            meanings = measure.dental_angle_meanings(entry.names, yaw, pitch, roll)
        else:
            # Upstream leaves an angle between skeletal landmarks unlabelled:
            # there is no dental table for it and no clock to read it against.
            meanings = None
        if meanings is not None:
            row["Yaw Meaning"], row["Pitch Meaning"], row["Roll Meaning"] = meanings
        return row

    row["R-L Component"] = abs(lr)
    row["A-P Component"] = abs(ap)
    row["S-I Component"] = abs(si)
    row["3D Distance"] = norm

    meanings = None
    if (entry.kind == measure.DISTANCE_2_POINTS
            and measure.is_dental(entry.names[0]) and measure.is_dental(entry.names[1])):
        meanings = measure.dental_distance_meanings(entry.names[0], entry.names[1], lr, ap, si)
    if meanings is None:
        # None, not the line's first landmark, for a point-to-line distance.
        # Upstream's second operand there is the Line OBJECT, and indexing it
        # gives None -- so a point-to-line measurement never reads a side off
        # the line, however the line is named. See `measure._side_of`.
        second = entry.names[1] if entry.kind == measure.DISTANCE_2_POINTS else None
        meanings = measure.skeletal_distance_meanings(entry.names[0], second, lr, ap, si)
    row["R-L Meaning"], row["A-P Meaning"], row["S-I Meaning"] = meanings
    return row


def compute_cohort(t1_landmarks: dict, t2_landmarks: dict, measurements,
                   report: dict = None, summary: dict = None) -> list:
    """Every measurement for every patient present at both timepoints.

    `t1_landmarks` and `t2_landmarks` are `{patient: {label: position}}`. A
    patient in one and not the other is reported and skipped: a measurement
    between a landmark and its mirror needs both sides of the pair, and half of
    one is not a smaller answer, it is no answer.

    What was skipped is logged once per MEASUREMENT, with how many patients it
    was skipped for and why, rather than once per patient and measurement: a
    landmark the search never finds would otherwise write one identical line
    per patient for every measurement that uses it, and bury the one line that
    says which landmark it was.

    `summary`, when given, is filled with the counts a caller needs to explain
    an empty result: `patients` (at both timepoints), `only_one` (at one), and
    `skipped`, `{reason: count}` over every patient and measurement.
    """
    from collections import Counter

    rows = []
    both = sorted(set(t1_landmarks) & set(t2_landmarks))
    only_one = sorted(set(t1_landmarks) ^ set(t2_landmarks))
    if only_one:
        logger.warning("VFACE: %d patient(s) have landmarks at only one timepoint "
                       "and are not measured", len(only_one))
        if report is not None:
            report.setdefault("not_measured", {}).update({
                patient: "present at only one timepoint" for patient in only_one
            })

    # {measurement label: Counter of reasons}, so each measurement is logged once.
    skipped: dict = {}
    for patient in both:
        positions = {
            T1: with_midpoints(t1_landmarks[patient], measurements),
            T2: with_midpoints(t2_landmarks[patient], measurements),
        }
        for entry in measurements:
            try:
                rows.append(compute_one(patient, positions, entry))
            except KeyError as missing:
                reason = f"landmark {missing.args[0]} absent"
                skipped.setdefault(entry.label, Counter())[reason] += 1
                if report is not None:
                    report.setdefault("landmarks_absent", {}).setdefault(
                        patient, []
                    ).append(f"{entry.label} ({missing.args[0]})")
            except measure.MeasurementError as exc:
                reason = f"degenerate: {exc}"
                skipped.setdefault(entry.label, Counter())[reason] += 1
                if report is not None:
                    report.setdefault("measurements_degenerate", {}).setdefault(
                        patient, []
                    ).append(f"{entry.label}: {exc}")

    for label, reasons in skipped.items():
        reason, _count = reasons.most_common(1)[0]
        logger.warning("VFACE: %s skipped for %d of %d patient(s) (%s)",
                       label, sum(reasons.values()), len(both), reason)

    if summary is not None:
        totals = Counter()
        for reasons in skipped.values():
            totals.update(reasons)
        summary.update(patients=len(both), only_one=len(only_one), skipped=dict(totals))
    return rows


def write_table(rows, path: str) -> str:
    """The measurement table, as the Excel the next step reads."""
    import pandas as pd

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pd.DataFrame(rows, columns=list(COLUMNS)).to_excel(path, index=False)
    return path
