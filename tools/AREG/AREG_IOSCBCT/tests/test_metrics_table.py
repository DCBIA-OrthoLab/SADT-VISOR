"""The metrics spreadsheet AREG IOS-CBCT writes when asked.

Built from the report the run already wrote, so the spreadsheet can never
disagree with it: one row per patient and arch, one per landmark.
"""

from sadt_areg_ioscbct import dispatch

REPORT = {"patients": {"case_a": {"meshes": {
    "case_a_U_Seg.vtk": {
        "status": "ok",
        "prealignment": {"accepted": True, "residual_mm": 1.3,
                         "landmarks_fitted": ["UR1O", "UR3O", "UR6O"],
                         "landmarks_excluded": {"UL1O": "not on a tooth in the CBCT"}},
        "landmarks_missing": {"UL6O": "IOS"},
        "icp": {"fitness": 0.95, "cost": "point-to-plane",
                "cost_milestones": [[0, 0.3], [7, 0.03]],
                "surface_mean_mm": 0.17, "surface_std_mm": 0.1,
                "landmark_gaps": {"labels": ["UR1O", "UR3O", "UR6O"],
                                  "before_mm": [1.0, 2.0, 3.0], "after_mm": [1.0, 1.0, 4.0]}},
    },
    "case_a_L_Seg.vtk": {"status": "failed", "error": "no landmark file"},
}}}}


def test_one_row_per_arch_with_what_the_console_showed():
    cases = dispatch.metric_tables(REPORT)["cases"]
    upper, lower = cases
    assert (upper["patient"], upper["arch"], upper["status"]) == ("case_a", "U", "ok")
    assert upper["landmarks_in_fit"] == 3
    assert upper["landmarks_left_out"] == "UL1O, UL6O"
    assert upper["rigid_fit_landmark_mean_mm"] == 2.0
    assert upper["after_icp_landmark_max_mm"] == 4.0
    assert upper["after_icp_landmark_max_label"] == "UR6O"
    assert (upper["icp_cost_start_mm2"], upper["icp_cost_end_mm2"], upper["icp_iterations"]) == (0.3, 0.03, 8)
    assert upper["crown_points_matched_pct"] == 95.0
    assert lower == {"patient": "case_a", "arch": "L", "status": "failed",
                     "error": "no landmark file"}


def test_one_row_per_landmark_used_or_left_out():
    landmarks = dispatch.metric_tables(REPORT)["landmarks"]
    used = [row for row in landmarks if row["in_fit"]]
    left = {row["landmark"]: row["left_out_because"] for row in landmarks if not row["in_fit"]}
    assert [(r["landmark"], r["after_rigid_fit_mm"], r["after_icp_mm"]) for r in used] == [
        ("UR1O", 1.0, 1.0), ("UR3O", 2.0, 1.0), ("UR6O", 3.0, 4.0)]
    assert left == {"UL1O": "not on a tooth in the CBCT",
                    "UL6O": "not found on the IOS side"}


def test_the_option_is_offered():
    import inspect
    import sadt_areg_ioscbct

    annotation = str(inspect.signature(sadt_areg_ioscbct.run).parameters["outputs"].annotation)
    assert "Metrics spreadsheet (.xlsx + .csv)" in annotation
