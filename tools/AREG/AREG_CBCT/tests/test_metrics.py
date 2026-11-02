"""What the registration measured, shown to the person who started it.

The optimiser's cost was computed on every run and written only to a log that
was deleted on return. It now reaches the user's console as `metric` lines,
with the rigid transform that was applied, and is kept in the report entry of
every region.
"""

import json
import os

import numpy as np
import SimpleITK as sitk
import torch

from conftest import cohort, displaced, full_mask, phantom
from sadt_areg_cbct import dispatch, elastix, gpu_rigid, metrics, pipeline, progress
from sadt_areg_common import catalogs

# The header and two rows of a real elastix.log, then what follows a table.
HEADER = ("1:ItNr\t1a:SrchDirNr\t1b:LineItNr\t2:Metric\t3:StepLength\t4a:||Gradient||\t"
          "4b:||SearchDir||\t4c:DirGradient\t5:Phase\t6a:Wolfe1\t6b:Wolfe2\t"
          "7:LinSrchStopCondition\tTime[ms]")
LOG = "\n".join([
    "Resolution: 0",
    "Initialization of AdvancedMattesMutualInformation metric took: 10 ms.",
    HEADER,
    "0\t0\t0\t-1.063390\t1.0\t0.2\t0.2\t-0.04\tMain\ttrue\ttrue\tWolfeSatisfied\t28.9",
    "1\t1\t0\t-1.118338\t1.0\t0.1\t0.2\t-0.04\tMain\ttrue\ttrue\tWolfeSatisfied\t8.2",
    "Time spent in resolution 0 (ITK initialization and iterating): 0.0698",
    "Stopping condition: Error in LineSearch.",
    "Resolution: 1",
    HEADER,
    "0\t0\t1\t-1.661622\t0.18\t0.7\t0.8\t0.44\tMain\ttrue\ttrue\tWolfeSatisfied\t28.3",
    "1\t1\t0\t-1.686469\t0.18\t0.7\t0.5\t0.13\tMain\ttrue\ttrue\tWolfeSatisfied\t8.1",
    "2\t2\t0\t-1.753147\t0.18\t0.6\t0.7\t0.02\tMain\ttrue\ttrue\tWolfeSatisfied\t8.1",
    "Time spent in resolution 1 (ITK initialization and iterating): 0.288",
    "Final metric value  = -1.753",
])

ENTRY = {
    "status": "ok",
    "region": "Cranial base",
    "engine": "elastix",
    "registration_cost": {
        "metric": "Mattes mutual information", "lower_is_better": True, "engine": "elastix",
        "computed_on": "the cranial base mask",
        "resolutions": [
            {"iterations": 40, "milestones": [[0, -0.4012], [39, -0.5233]]},
            {"iterations": 25, "milestones": [[0, -0.5871], [24, -0.6015]]},
            {"iterations": 12, "milestones": [[0, -0.6102], [3, -0.6188], [11, -0.6203]]},
        ],
    },
    "rigid_transform": {"rotation_deg": 2.4567, "translation_mm": 3.25},
}


# ---------------------------------------------------------------------------
# Reading the cost
# ---------------------------------------------------------------------------

def test_elastix_log_tables_are_read_one_list_per_resolution():
    assert elastix.parse_iterations(LOG) == [
        [-1.06339, -1.118338], [-1.661622, -1.686469, -1.753147]]


def test_a_log_without_a_table_has_no_cost():
    assert elastix.parse_iterations("Resolution: 0\nnothing ran\n") == []
    assert elastix.read_cost("/nonexistent/elastix/output") == []


def test_iteration_info_files_win_over_the_log(tmp_path):
    (tmp_path / "elastix.log").write_text(LOG)
    for level, values in ((1, ["-0.5", "-0.6"]), (0, ["-0.1"])):
        rows = [f"{i}\t0\t0\t{v}\t1\t1\t1\t1\tMain\ttrue\ttrue\tok\t1" for i, v in enumerate(values)]
        (tmp_path / f"IterationInfo.0.R{level}.txt").write_text("\n".join([HEADER] + rows))
    assert elastix.read_cost(str(tmp_path)) == [[-0.1], [-0.5, -0.6]]


def test_an_elastix_registration_hands_back_its_cost():
    fixed = phantom(size=32)
    moving, _truth = displaced(fixed)
    masked, _note = elastix.apply_mask(fixed, full_mask(fixed))
    cost = []
    elastix.register(masked, moving, cost=cost)
    assert len(cost) == int(elastix._RIGID_PARAMETERS["NumberOfResolutions"][0])
    assert all(level and all(value < 0 for value in level) for level in cost), \
        "elastix minimises the negative mutual information"


def test_the_gpu_engine_records_the_best_cost_it_kept_at_every_iteration(monkeypatch):
    """Modelled on the CPU, as the uphill test in test_gpu_rigid: the value
    falls, rises, then falls below where it was."""
    walk = iter([3.0, 1.0, 2.0, 0.5] * 100)

    def gradient(self, level, params):
        params.grad = torch.zeros_like(params)
        return next(walk)

    monkeypatch.setattr(gpu_rigid._Registration, "_gradient", gradient)
    monkeypatch.setattr(gpu_rigid, "ITERATIONS", (4, 4, 4))
    fixed = phantom(size=24)
    cost = []
    gpu_rigid._Registration(fixed, fixed, torch.device("cpu")).run(cost)
    assert cost == [[3.0, 1.0, 1.0, 0.5]] * 3


def test_a_failure_on_the_card_leaves_only_elastixs_cost(monkeypatch):
    def half_done(fixed, moving, cost=None):
        cost.append([9.0])
        raise RuntimeError("CUDA out of memory")

    def elastix_register(fixed, moving, cost=None):
        cost.append([-1.0, -2.0])
        return "elastix transform"

    monkeypatch.setattr(gpu_rigid, "available", lambda: True)
    monkeypatch.setattr(gpu_rigid, "register", half_done)
    monkeypatch.setattr(elastix, "register", elastix_register)
    cost = []
    assert pipeline._register(None, None, cost) == ("elastix transform", "elastix")
    assert cost == [[-1.0, -2.0]]


# ---------------------------------------------------------------------------
# Thinning
# ---------------------------------------------------------------------------

def test_the_milestones_show_the_cost_falling():
    values = [-1.0, -1.01, -1.3, -1.6, -1.605, -1.61, -1.9, -1.9001, -1.9002]
    milestones = metrics.cost_milestones(values)

    assert milestones[0] == [0, -1.0]
    assert milestones[-1] == [8, -1.9002], "the last iteration is where it ended"
    costs = [cost for _i, cost in milestones]
    assert costs == sorted(costs, reverse=True)
    assert [1, -1.01] not in milestones, "a step of 1% of the descent is noise"


def test_a_long_descent_stays_a_few_numbers():
    milestones = metrics.cost_milestones([-1.0 + 0.8 ** i for i in range(300)], most=6)
    assert len(milestones) == 6
    assert milestones[0][0] == 0 and milestones[-1][0] == 299


def test_a_cost_that_never_fell_is_its_first_and_last_value():
    assert metrics.cost_milestones([-0.5, -0.4, -0.45]) == [[0, -0.5], [2, -0.45]]
    assert metrics.cost_milestones([-0.5]) == [[0, -0.5]]
    assert metrics.cost_milestones([float("nan")]) == []


def test_a_descent_of_a_few_ten_thousandths_is_not_several_improvements():
    assert metrics.cost_milestones([-0.5976, -0.5979, -0.5980, -0.5981]) == [
        [0, -0.5976], [3, -0.5981]]


def test_two_milestones_equal_once_rounded_are_one():
    assert metrics.cost_milestones([-1.0, -1.77840, -1.77843]) == [[0, -1.0], [2, -1.7784]]


def test_the_summary_keeps_every_resolution_and_where_it_was_computed():
    summary = metrics.summarise_cost([[-1.0, -2.0], [], [-3.0]], "gpu", "the mandible mask")
    assert summary["engine"] == "gpu" and summary["lower_is_better"] is True
    assert summary["computed_on"] == "the mandible mask"
    assert [level["iterations"] for level in summary["resolutions"]] == [2, 1]
    assert metrics.summarise_cost([], "gpu") is None


def test_the_rigid_motion_is_the_angle_and_the_centre_displacement():
    transform = sitk.Euler3DTransform()
    transform.SetCenter((-120.0, 40.0, 75.0))
    transform.SetRotation(0.0, 0.0, np.radians(3.0))
    transform.SetTranslation((3.0, 4.0, 0.0))
    assert metrics.rigid_motion(transform) == {"rotation_deg": 3.0, "translation_mm": 5.0}


# ---------------------------------------------------------------------------
# The lines
# ---------------------------------------------------------------------------

def test_each_quantity_on_its_own_line():
    assert metrics.metric_lines(ENTRY) == [
        "registration cost, Mattes mutual information (lower is better), finest of 3 "
        "resolutions: -0.6102 > -0.6188 > -0.6203, 12 iterations (77 in all)",
        "rigid transform applied: rotation 2.5°, translation 3.2 mm",
    ]


def test_a_cost_that_rose_at_the_end_says_so():
    entry = {"registration_cost": {"resolutions": [
        {"iterations": 3, "milestones": [[0, -1.0], [1, -1.5], [2, -1.4]]}]}}
    assert metrics.metric_lines(entry) == [
        "registration cost, Mattes mutual information (lower is better): "
        "-1.0000 > -1.5000 < -1.4000, 3 iterations"]


def test_the_header_names_the_position_and_the_region_never_a_file():
    assert metrics.header(2, 5, ENTRY) == (
        "patient 2/5, cranial base, cost computed on the T1 inside the cranial base mask")


def test_a_failed_registration_has_no_lines():
    assert metrics.metric_lines({"status": "failed", "reason": "x"}) == []


def test_the_lines_reach_the_user_as_metrics_under_their_case(tmp_path, monkeypatch):
    stream = tmp_path / "events.jsonl"
    monkeypatch.setenv(progress.VARIABLE, str(stream))

    metrics.report_metrics(metrics.header(1, 1, ENTRY), ENTRY)

    records = [json.loads(line) for line in stream.read_text().splitlines()]
    assert {r["level"] for r in records} == {"metric"}
    assert {r["audience"] for r in records} == {"user"}
    assert records[0]["message"].startswith("patient 1/1, cranial base")
    assert [r["message"] for r in records[1:]] == metrics.metric_lines(ENTRY)
    assert all(len(r["message"]) < 200 for r in records)
    assert not any("—" in r["message"] or "->" in r["message"] for r in records)


def test_a_run_reports_and_announces_what_each_registration_measured(tmp_path, monkeypatch):
    stream = tmp_path / "events.jsonl"
    monkeypatch.setenv(progress.VARIABLE, str(stream))
    cohort(tmp_path, subjects=("P1", "P2"), size=32)

    run = dispatch.register(
        t1_path=os.path.join(str(tmp_path), "T1"), t2_path=os.path.join(str(tmp_path), "T2"),
        t1_masks_path=os.path.join(str(tmp_path), "masks"),
        automation=catalogs.AUTOMATION_SEMI, regions=["Cranial base"],
        output_dir=os.path.join(str(tmp_path), "out"))

    entries = [patient["regions"]["CB"] for patient in run.patients.values()]
    for entry in entries:
        assert entry["registration_cost"]["resolutions"], "the cost is in the report"
        assert entry["registration_cost"]["engine"] == entry["engine"]
        # The phantom's T2 is the T1 rotated by a few degrees and moved about
        # 2.2 mm (conftest.displaced). How much of that an engine recovers on
        # a 32-voxel phantom differs -- elastix on a CPU runner found under a
        # degree -- and is the registration tests' business, not this one's:
        # here, only that a real, bounded motion is reported.
        assert 0.0 < entry["rigid_transform"]["rotation_deg"] < 10.0
        assert 0.0 < entry["rigid_transform"]["translation_mm"] < 10.0
    saved = json.loads(open(os.path.join(str(tmp_path), "out", dispatch.REPORT_NAME)).read())
    assert any("rigid_transform" in region for patient in saved["patients"].values()
               for region in patient["regions"].values())

    records = [json.loads(line) for line in stream.read_text().splitlines()]
    shown = [r["message"] for r in records if r.get("level") == "metric"]
    assert {r["audience"] for r in records if r.get("level") == "metric"} == {"user"}
    assert shown[0] == "patient 1/2, cranial base, cost computed on the T1 inside the cranial base mask"
    assert shown[3].startswith("patient 2/2, cranial base")
    assert shown[1].startswith("registration cost, Mattes mutual information (lower is better)")
    assert shown[2].startswith("rigid transform applied: rotation ")
    assert not any("P1" in line or "P2" in line for line in shown)
