"""What an orientation measured, shown to the person who started it.

The landmark distances to the reference after each step, the ICP's own end
state and the rotation and translation applied reach the user's console as
`metric` lines, and the per-scan report entry carries the same numbers.
"""

import json
import os

import numpy as np

from sadt_aso import catalogs, dispatch, geometry, progress, user_metrics

from test_run import _REFERENCE_POINTS, _cbct_case, _cbct_reference

METRICS = {
    "points": ["landmark", "landmarks"],
    "labels": ["Ba", "N", "S"],
    "left_out": {"RPo": "rejected as an outlier", "Gn": "not in the input landmark file"},
    "stages": [
        {"after": user_metrics.STEP_THREE_POINT, "distances_mm": [1.0, 2.0, 3.0]},
        {"after": user_metrics.STEP_ORIENTATION, "distances_mm": [0.5, 0.5, 2.0]},
    ],
    "icp": {"iterations": 1000, "rms_closest_mm": 0.4321, "translation_applied": False},
    "rotation_deg": 17.04,
    "translation_mm": 13.86,
}


def test_each_quantity_on_its_own_line():
    assert user_metrics.metric_lines(METRICS) == [
        "Gn left out of the fit: not in the input landmark file",
        "RPo left out of the fit: rejected as an outlier",
        "landmark distance to the reference after the initial three-point fit: "
        "2.00 ± 0.82 mm, max 3.00 mm (S), 3 landmarks",
        "landmark distance to the reference after orientation: "
        "1.00 ± 0.71 mm, max 2.00 mm (S), 3 landmarks",
        "ICP closest-point RMS distance at its final pose: 0.43 mm",
        "orientation applied: rotation 17.0°, translation 13.9 mm",
    ]


def test_tooth_centroids_are_named_as_teeth():
    metrics = dict(METRICS, points=list(user_metrics.POINTS_CENTROID), left_out={},
                   labels=["UR6", "UL1", "UL6"],
                   icp={"iterations": 100, "rms_closest_mm": 0.0})
    lines = user_metrics.metric_lines(metrics)
    assert lines[1] == ("tooth centroid distance to the reference after orientation: "
                        "1.00 ± 0.71 mm, max 2.00 mm (UL6), 3 teeth")
    assert lines[2] == "ICP closest-point RMS distance at its final pose: 0.00 mm"


def test_a_driven_jaw_says_it_was_not_fitted():
    lines = user_metrics.metric_lines({"moved_with": "Upper", "rotation_deg": 12.0,
                                       "translation_mm": 6.2})
    assert lines == ["moved with the upper jaw's transform, not fitted on its own",
                     "orientation applied: rotation 12.0°, translation 6.2 mm"]


def test_nothing_measured_says_nothing():
    assert user_metrics.metric_lines({}) == []
    assert user_metrics.metric_lines(None) == []


def test_no_line_carries_an_arrow_or_an_em_dash_and_all_fit():
    for line in user_metrics.metric_lines(METRICS):
        assert "—" not in line and "->" not in line and "→" not in line
        assert len(line) < progress.MAX_MESSAGE


def test_the_rotation_and_translation_of_a_matrix():
    matrix = np.eye(4)
    matrix[:3, :3] = geometry.rotation_matrix(np.array([0.2, 1.0, 0.3]), np.radians(17.0))
    matrix[:3, 3] = (3.0, 4.0, 0.0)
    rotation, translation = user_metrics.rotation_and_translation(matrix)
    assert np.isclose(rotation, 17.0)
    assert np.isclose(translation, 5.0)


def test_the_closest_point_rms_uses_the_nearest_target_point():
    moved = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)]
    target = [(10.0, 0.0, 1.0), (0.0, 0.0, 2.0)]
    assert np.isclose(user_metrics.closest_point_rms(moved, target), np.sqrt((4.0 + 1.0) / 2))


def test_the_lines_reach_the_user_as_metrics_under_their_scan(tmp_path, monkeypatch):
    stream = tmp_path / "events.jsonl"
    monkeypatch.setenv(progress.VARIABLE, str(stream))

    user_metrics.report_metrics("patient 2/5", METRICS)

    records = [json.loads(line) for line in stream.read_text().splitlines()]
    assert {r["level"] for r in records} == {"metric"}
    assert {r["audience"] for r in records} == {"user"}
    assert records[0]["message"] == "patient 2/5"
    assert [r["message"] for r in records[1:]] == user_metrics.metric_lines(METRICS)


def test_a_cbct_run_reports_the_distance_its_known_transform_leaves(tmp_path, monkeypatch):
    """The landmarks are the reference rotated by 17 degrees and shifted by
    (12, -8, 5) mm; the 16-voxel scan is then recentred by (-8, -8, -8) mm.
    The CBCT orientation undoes the rotation and applies no translation of its
    own, so every landmark ends the same |(4, -16, -3)| = 16.76 mm from its
    reference counterpart: an offset between the two centred frames, not an
    error. With the centres matched, the distance that grades the rotation is
    zero; the move applied is 17 degrees and the recentring's 13.86 mm."""
    stream = tmp_path / "events.jsonl"
    monkeypatch.setenv(progress.VARIABLE, str(stream))
    _cbct_case(tmp_path / "input", key="patient1")

    run = dispatch.orient(
        input_path=str(tmp_path / "input"),
        reference_path=_cbct_reference(tmp_path),
        modality=catalogs.MODALITY_CBCT,
        automation=catalogs.AUTOMATION_SEMI,
        cbct_landmarks=list(_REFERENCE_POINTS),
        output_dir=str(tmp_path / "out"),
    )

    expected = 0.0
    metrics = run.patients["patient1"]["metrics"]
    final = metrics["stages"][-1]
    assert final["after"] == user_metrics.STEP_ORIENTATION
    assert np.allclose(final["distances_mm"], expected, atol=1e-3)
    assert sorted(metrics["labels"]) == sorted(_REFERENCE_POINTS)
    assert np.isclose(metrics["rotation_deg"], 17.0, atol=1e-3)
    assert np.isclose(metrics["translation_mm"], np.linalg.norm([8.0, 8.0, 8.0]), atol=1e-3)

    with open(os.path.join(run.output_dir, dispatch.REPORT_NAME), encoding="utf-8") as handle:
        written = json.load(handle)["cases"]["patient1"]["metrics"]
    assert np.allclose(written["stages"][-1]["distances_mm"], expected, atol=1e-3)

    records = [json.loads(line) for line in stream.read_text().splitlines()]
    shown = [r["message"] for r in records
             if r.get("level") == "metric" and r.get("audience") == "user"]
    assert shown[0] == "patient 1/1"
    assert ("landmark distance to the reference after orientation, centres matched: "
            "0.00 ± 0.00 mm, max 0.00 mm") in " ".join(shown)
    assert "orientation applied: rotation 17.0°, translation 13.9 mm" in shown
    assert not any("patient1" in message for message in shown)
