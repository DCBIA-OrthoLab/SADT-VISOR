"""What the registration measured, shown to the person who started it.

The landmark gaps, the ICP's cost and how close the crowns ended were computed
on every run and written only to the operator's log. They now reach the user's
console as `metric` lines, as quantities a reader can use: means with their
spread, the worst landmark pair, and the cost the ICP actually minimises.
"""

import json

import numpy as np
import pyvista as pv

from sadt_areg_ioscbct import dispatch, geometry, progress

DETAIL = {
    "prealignment": {"accepted": True, "residual_mm": 1.284,
                     "landmarks_fitted": ["UR1O", "UR3O", "UR6O"]},
    "icp": {"fitness": 0.934, "cost": "point-to-plane",
            "cost_milestones": [(0, 0.3341), (3, 0.0612), (9, 0.0451)],
            "surface_mean_mm": 0.183, "surface_std_mm": 0.121,
            "landmark_gaps": {"labels": ["UR1O", "UR3O", "UR6O"],
                              "before_mm": [1.0, 2.0, 3.0],
                              "after_mm": [0.5, 0.5, 2.0]}},
}


def test_the_milestones_show_the_cost_falling():
    history = [2.0, 1.9, 1.5, 1.0, 0.99, 0.98, 0.6, 0.59, 0.59]
    milestones = geometry.error_milestones(history)

    assert milestones[0] == (0, 2.0)
    assert milestones[-1] == (8, 0.59), "the last iteration is where it ended"
    costs = [cost for _i, cost in milestones]
    assert costs == sorted(costs, reverse=True)
    assert (4, 0.99) not in milestones, "a 1% step is noise, not a milestone"


def test_a_long_descent_stays_a_few_numbers():
    milestones = geometry.error_milestones([10.0 * 0.8 ** i for i in range(300)], most=6)
    assert len(milestones) == 6
    assert milestones[0][0] == 0 and milestones[-1][0] == 299


def test_an_iteration_without_correspondences_is_skipped():
    assert geometry.error_milestones([float("inf"), 1.0, 0.5]) == [(1, 1.0), (2, 0.5)]
    assert geometry.error_milestones([float("inf")]) == []


def test_each_quantity_on_its_own_line():
    assert dispatch.metric_lines(DETAIL) == [
        "landmark distance after the rigid landmark fit: 2.00 ± 0.82 mm, "
        "max 3.00 mm (UR6O), 3 pairs",
        "landmark distance after the ICP: 1.00 ± 0.71 mm, max 2.00 mm (UR6O), 3 pairs",
        "ICP cost, mean squared point-to-plane distance: 0.3341 > 0.0612 > 0.0451 mm², "
        "10 iterations",
        "crown to CBCT surface distance: 0.18 ± 0.12 mm, over 93% of the crown points",
    ]


def test_every_landmark_left_out_is_named_with_its_reason():
    detail = dict(DETAIL, landmarks_missing={"UL6O": "CBCT"},
                  prealignment=dict(DETAIL["prealignment"], landmarks_excluded={
                      "LL1O": "4.9 mm from where the other 5 put it",
                      "UL1O": "not on a tooth in the CBCT"}))
    lines = dispatch.metric_lines(detail)
    assert "LL1O left out of the rigid fit: 4.9 mm from where the other 5 put it" in lines
    assert "UL1O left out of the rigid fit: not on a tooth in the CBCT" in lines
    assert "UL6O left out of the rigid fit: not found on the CBCT side" in lines


def test_the_fit_records_why_it_dropped_a_pair():
    rng = np.random.default_rng(0)
    moving = rng.uniform(-20, 20, size=(6, 3))
    fixed = moving + (1.0, 2.0, 3.0)
    fixed[2] += (6.0, 0.0, 0.0)
    labels = ["A", "B", "C", "D", "E", "F"]
    excluded = {}
    _matrix, keep = geometry.fit_rigid_without_outliers(
        moving, fixed, labels, on_enamel={"F": False}, excluded=excluded)
    assert excluded["F"] == "not on a tooth in the CBCT"
    assert excluded["C"].endswith("from where the other 4 put it")
    assert 2 not in keep and 5 not in keep


def test_without_a_surface_there_is_only_the_landmark_fit():
    detail = {"prealignment": {"accepted": False, "residual_mm": 9.1,
                               "landmarks_fitted": ["UR1O"]}, "icp": None}
    assert dispatch.metric_lines(detail) == [
        "landmark distance after the rigid landmark fit: 9.10 mm RMS, 1 pairs, "
        "too far apart to be used"]


def test_the_icp_reports_the_cost_it_minimises():
    """On a plane shifted 2 mm along its normal, the point-to-plane cost starts
    at 4 mm^2 and falls to nothing."""
    plane = pv.Plane(i_size=40, j_size=40, i_resolution=40, j_resolution=40).triangulate()
    target = geometry.Target.from_mesh(plane)
    moved = plane.translate((0, 0, 2.0), inplace=False)
    _matrix, stats = geometry.icp_point_to_plane(moved, target, max_dist=5.0)

    first, last = stats["cost_milestones"][0][1], stats["cost_milestones"][-1][1]
    assert abs(first - 4.0) < 0.05
    assert last < 0.01
    assert stats["surface_mean_mm"] < 0.1


def test_the_lines_reach_the_user_as_metrics_under_their_arch(tmp_path, monkeypatch):
    stream = tmp_path / "events.jsonl"
    monkeypatch.setenv(progress.VARIABLE, str(stream))

    dispatch.report_metrics("patient 1/1, arch U", DETAIL)

    records = [json.loads(line) for line in stream.read_text().splitlines()]
    assert {r["level"] for r in records} == {"metric"}
    assert {r["audience"] for r in records} == {"user"}
    assert records[0]["message"] == "patient 1/1, arch U"
    assert all(len(r["message"]) <= progress.MAX_MESSAGE for r in records)
    assert np.isclose(len(records), 1 + len(dispatch.metric_lines(DETAIL)))
