"""run(): the chain through a fake supervisor, and the contract around it."""

import inspect
import json
import typing

import pytest

from conftest import FakeSup
from sadt_tmj_crop import crop, run, tools
from sadt_tmj_crop.errors import SupervisorRequired, ToolInputError


def test_a_run_writes_the_crops_the_report_and_the_quick_view(tmp_path, sup):
    out = run(scans=tmp_path / "in", output_dir=tmp_path / "out", sup=sup)

    for suffix in ("_R.nii.gz", "_L.nii.gz", ".json", "_qc.png"):
        assert (out / f"a{suffix}").stat().st_size > 0, suffix
    report = json.loads((out / "a.json").read_text())
    assert {side: report["sides"][side]["status"] for side in "RL"} == {"R": "ok", "L": "ok"}
    assert not [p for p in out.iterdir() if p.name.startswith(".tmj_")], "the staging folder is removed"


def test_the_two_tools_are_asked_for_what_the_crop_needs(tmp_path, sup):
    run(scans=tmp_path / "in", output_dir=tmp_path / "out", sup=sup)

    aso = sup.asked("ASO")
    assert aso["frame"] == "Frankfurt horizontal"
    assert aso["automation"] == "Fully-Automated" and aso["modality"] == "CBCT"
    # No model and no output folder: each tool resolves its own, and the supervisor places the output.
    assert not {"model", "landmark_model", "reference", "output_dir"} & set(aso)
    amasss = sup.asked("AMASSS")
    assert amasss["structures"] == ["MAND"] and amasss["merge"] == ["SEPARATE"]
    assert not {"model", "output_dir"} & set(amasss)


def test_a_batch_calls_each_tool_once(tmp_path):
    sup = FakeSup(tmp_path, names=("a", "b", "c"))

    out = run(scans=tmp_path / "in", output_dir=tmp_path / "out", sup=sup)

    assert [name for name, _ in sup.calls] == ["ASO", "AMASSS"]
    assert sorted(p.name for p in out.glob("*_R.nii.gz")) == ["a_R.nii.gz", "b_R.nii.gz", "c_R.nii.gz"]


def test_without_a_supervisor_the_run_is_refused_up_front(tmp_path):
    with pytest.raises(SupervisorRequired, match="ASO"):
        run(scans=tmp_path / "missing", output_dir=tmp_path / "out")


def test_an_aso_without_reference_positions_is_named_as_the_problem(tmp_path):
    sup = FakeSup(tmp_path, with_reference=False)

    with pytest.raises(ToolInputError, match="reference positions"):
        run(scans=tmp_path / "in", output_dir=tmp_path / "out", sup=sup)


def test_a_scan_ASO_could_not_orient_gets_a_report_and_the_rest_goes_on(tmp_path):
    sup = FakeSup(tmp_path, names=("a", "b"))
    original = sup.run

    def run_with_one_failure(tool, **params):
        folder = original(tool, **params)
        if tool == "ASO":
            report = json.loads((folder / "ASO_report.json").read_text())
            report["cases"]["b"].update(status="failed", produced=[])
            (folder / "ASO_report.json").write_text(json.dumps(report))
        return folder

    sup.run = run_with_one_failure
    out = run(scans=tmp_path / "in", output_dir=tmp_path / "out", sup=sup)

    assert (out / "a_R.nii.gz").exists() and not (out / "b_R.nii.gz").exists()
    assert "orientation failed" in json.loads((out / "b.json").read_text())["status"]


def test_a_batch_that_oriented_nothing_raises_instead_of_returning_an_empty_folder(tmp_path):
    sup = FakeSup(tmp_path, aso_status="failed")

    with pytest.raises(ToolInputError, match="oriented none"):
        run(scans=tmp_path / "in", output_dir=tmp_path / "out", sup=sup)


def test_a_missing_mandible_mask_is_reported_per_scan_and_fails_the_batch_when_alone(tmp_path):
    sup = FakeSup(tmp_path, mandible=False)

    with pytest.raises(ToolInputError, match="could be cropped"):
        run(scans=tmp_path / "in", output_dir=tmp_path / "out", sup=sup)
    assert "no mandible mask" in json.loads((tmp_path / "out" / "a.json").read_text())["status"]


def test_the_published_choices_are_the_ones_the_crop_understands():
    hint = typing.get_type_hints(run)["scale_by"]
    assert list(typing.get_args(hint)) == ["none", "mandible-volume"]
    assert crop.CropSettings().scale_by in typing.get_args(hint)


def test_the_supervisor_is_keyword_only_and_not_part_of_the_published_arguments():
    parameter = inspect.signature(run).parameters["sup"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


def test_the_calls_it_makes_are_the_ones_it_declares():
    source = inspect.getsource(tools)
    assert 'sup.run("ASO"' in source and 'sup.run("AMASSS"' in source
    assert 'sup.run("Convert"' in source


def test_the_box_scaling_options_are_hidden_from_the_panel():
    """A clinician must not pick the box size; the API still accepts it."""
    import inspect

    from sadt_tmj_crop import layout, run

    parameters = inspect.signature(run).parameters
    for name in ("scale_by", "scale"):
        assert name in parameters
        assert layout.LAYOUT[name] == {"hidden": True}
    hidden = {name for name, hints in layout.LAYOUT.items() if hints.get("hidden")}
    # keep_intermediate is injected by the server, not part of run()'s signature.
    assert hidden == {"scale_by", "scale", "keep_intermediate"}


def _names(folder):
    return sorted(p.name for p in folder.iterdir() if p.is_file())


def test_by_default_only_the_crops_the_report_and_the_quick_view_are_returned(tmp_path, sup):
    out = run(scans=tmp_path / "in", output_dir=tmp_path / "out", sup=sup)

    assert _names(out) == sorted(["run_report.json", "a.json", "a_L.nii.gz", "a_R.nii.gz", "a_qc.png"])


def test_the_oriented_scan_and_the_mandible_are_returned_when_asked(tmp_path, sup):
    out = run(scans=tmp_path / "in", output_dir=tmp_path / "out", sup=sup,
              outputs=["Oriented scan", "Mandible segmentation"])

    assert {"a_oriented.nii.gz", "a_mandible.nii.gz"} <= set(_names(out))
    # Real files, not links into a staging folder that is deleted at the end.
    for name in ("a_oriented.nii.gz", "a_mandible.nii.gz"):
        assert not (out / name).is_symlink() and (out / name).stat().st_size > 0


def test_each_extra_output_is_independent(tmp_path, sup):
    out = run(scans=tmp_path / "in", output_dir=tmp_path / "out", sup=sup, outputs=["Oriented scan"])
    assert "a_oriented.nii.gz" in _names(out) and "a_mandible.nii.gz" not in _names(out)

    out = run(scans=tmp_path / "in", output_dir=tmp_path / "out2", sup=sup, outputs=["Mandible segmentation"])
    assert "a_mandible.nii.gz" in _names(out) and "a_oriented.nii.gz" not in _names(out)


@pytest.mark.parametrize("sides,present,absent", [
    ("Left and Right", "RL", ""), ("Left", "L", "R"), ("Right", "R", "L"),
])
def test_only_the_requested_joints_are_cropped_and_reported(tmp_path, sup, sides, present, absent):
    out = run(scans=tmp_path / "in", output_dir=tmp_path / "out", sup=sup, sides=sides)

    for side in present:
        assert (out / f"a_{side}.nii.gz").stat().st_size > 0
    for side in absent:
        assert not (out / f"a_{side}.nii.gz").exists()
    report = json.loads((out / "a.json").read_text())
    assert set(report["sides"]) == set(present)
    assert (out / "a_qc.png").stat().st_size > 0


def test_an_unknown_choice_of_sides_is_refused(tmp_path, sup):
    with pytest.raises(ToolInputError, match="sides"):
        run(scans=tmp_path / "in", output_dir=tmp_path / "out", sup=sup, sides="Both")


def test_the_choices_of_sides_published_are_the_ones_accepted():
    from sadt_tmj_crop.dispatch import SIDES

    published = typing.get_args(inspect.signature(run).parameters["sides"].annotation)
    assert set(published) == set(SIDES)


def test_a_requested_joint_that_is_not_found_does_not_count_as_a_crop(tmp_path):
    """One side asked for and not found leaves nothing: the batch must say so."""
    sup = FakeSup(tmp_path, mandible=False)

    with pytest.raises(ToolInputError):
        run(scans=tmp_path / "in", output_dir=tmp_path / "out", sup=sup, sides="Left")


def test_the_extra_outputs_published_are_the_ones_the_catalog_names():
    """`run()` spells the options out for the schema; the catalog is what the code tests."""
    from sadt_tmj_crop import catalogs

    published = typing.get_args(typing.get_args(
        inspect.signature(run).parameters["outputs"].annotation)[0])
    assert set(published) == set(catalogs.OUTPUT_HELP)
    from sadt_tmj_crop import layout
    assert set(layout.LAYOUT["outputs"]["option_help"]) == set(published)


def test_nothing_extra_is_ticked_by_default():
    assert inspect.signature(run).parameters["outputs"].default == []


def test_an_unknown_extra_output_is_refused(tmp_path, sup):
    with pytest.raises(ToolInputError, match="extra output"):
        run(scans=tmp_path / "in", output_dir=tmp_path / "out", sup=sup, outputs=["Landmarks"])


# --- A study sent as one folder: subfolders, repeated names, odd names, damaged scans -------


def test_two_scans_with_the_same_name_in_different_subfolders_are_kept_apart(tmp_path):
    """A follow-up study as one folder: T1/case and T2/case must neither collide nor swap."""
    sup = FakeSup(tmp_path, names=("T1/case", "T2/case"))

    out = run(scans=tmp_path / "in", output_dir=tmp_path / "out", sup=sup, sides="Right")

    assert _names(out) == ["T1_case.json", "T1_case_R.nii.gz", "T1_case_qc.png",
                           "T2_case.json", "T2_case_R.nii.gz", "T2_case_qc.png",
                           "run_report.json"]
    # Still one call to each tool for the whole folder.
    assert [name for name, _ in sup.calls] == ["ASO", "AMASSS"]


def test_names_stay_plain_when_the_subfolders_are_not_needed_to_tell_scans_apart(tmp_path):
    sup = FakeSup(tmp_path, names=("T1/a", "T2/b"))

    out = run(scans=tmp_path / "in", output_dir=tmp_path / "out", sup=sup, sides="Right")

    assert {"a_R.nii.gz", "b_R.nii.gz"} <= set(_names(out))


def test_a_name_that_looks_like_a_glob_pattern_is_found_like_any_other(tmp_path):
    sup = FakeSup(tmp_path, names=("scan[1]", "ok"))

    out = run(scans=tmp_path / "in", output_dir=tmp_path / "out", sup=sup, sides="Right")

    assert {"scan[1]_R.nii.gz", "ok_R.nii.gz"} <= set(_names(out))


def test_a_scan_that_breaks_the_crop_does_not_cost_the_rest_of_the_batch(tmp_path, monkeypatch):
    """Any error, not only a ValueError: a truncated file raised EOFError and lost every case."""
    sup = FakeSup(tmp_path, names=("a", "b", "c"))
    real = crop.crop_scan

    def crop_scan(scan, landmarks, mask, reference, *args, **kwargs):
        if kwargs["name"] == "b":
            raise EOFError("Compressed file ended before the end-of-stream marker: /data/p/secret.nii.gz")
        return real(scan, landmarks, mask, reference, *args, **kwargs)

    monkeypatch.setattr(crop, "crop_scan", crop_scan)
    out = run(scans=tmp_path / "in", output_dir=tmp_path / "out", sup=sup, sides="Right")

    assert (out / "a_R.nii.gz").exists() and (out / "c_R.nii.gz").exists()
    report = json.loads((out / "b.json").read_text())
    assert report["status"] == "crop failed: EOFError"
    assert "secret" not in json.dumps(report), "the message of a library error can quote a path"


def test_the_reason_ASO_gave_reaches_the_scan_report_without_a_file_name(tmp_path):
    sup = FakeSup(tmp_path, names=("a", "b"),
                  reasons={"b": "could not read /data/clinic/p_0042.nii.gz: bad header"})
    original = sup.run

    def run_with_one_failure(tool, **params):
        folder = original(tool, **params)
        if tool == "ASO":
            report = json.loads((folder / "ASO_report.json").read_text())
            report["cases"]["b"].update(status="failed", produced=[])
            (folder / "ASO_report.json").write_text(json.dumps(report))
        return folder

    sup.run = run_with_one_failure
    out = run(scans=tmp_path / "in", output_dir=tmp_path / "out", sup=sup)

    status = json.loads((out / "b.json").read_text())["status"]
    assert status.startswith("orientation failed:") and "bad header" not in status.split("<file>")[0]
    assert "<file>" in status and "p_0042" not in status and "/data" not in status
    assert any("orientation failed" in str(m) for _, m in sup.messages), "told to the operator too"


def test_when_nothing_is_oriented_the_error_names_the_reason_without_a_path(tmp_path):
    sup = FakeSup(tmp_path, aso_status="failed",
                  reasons={"a": "could not read /data/clinic/p_0042.nii.gz"})

    with pytest.raises(ToolInputError) as error:
        run(scans=tmp_path / "in", output_dir=tmp_path / "out", sup=sup)

    assert "could not read" in str(error.value)
    assert "p_0042" not in str(error.value) and "/data" not in str(error.value)


@pytest.mark.parametrize("options", [
    {"scale_by": "mandible_volume"}, {"scale_by": "bogus"},
    {"scale": -1.0}, {"scale": float("nan")}, {"scale": float("inf")}, {"scale": 1e-6},
])
def test_a_scaling_option_that_means_nothing_is_refused_not_ignored(tmp_path, sup, options):
    with pytest.raises(ToolInputError, match="scale"):
        run(scans=tmp_path / "in", output_dir=tmp_path / "out", sup=sup, **options)


def test_the_documented_scaling_options_are_accepted(tmp_path, sup):
    run(scans=tmp_path / "in", output_dir=tmp_path / "out1", sup=sup,
        scale_by="mandible-volume", sides="Right")
    run(scans=tmp_path / "in", output_dir=tmp_path / "out2", sup=sup, scale=1.1, sides="Right")


def test_the_scaling_choices_published_are_the_ones_validated():
    published = typing.get_args(inspect.signature(run).parameters["scale_by"].annotation)
    assert set(published) == set(crop.SCALE_BY)


def test_the_refusal_without_a_supervisor_does_not_advise_what_run_cannot_take():
    with pytest.raises(SupervisorRequired) as error:
        run(scans="x", output_dir="y")
    assert "VISOR server" in str(error.value)
    assert "yourself" not in str(error.value)


# --- The run report every tool writes beside its results -------------------------------------


def _report(out):
    return json.loads((out / "run_report.json").read_text())


def _study(tmp_path, *relative):
    """A folder holding empty files at these paths (the fake supervisor reads none)."""
    for path in relative:
        target = tmp_path / "in" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"")
    return tmp_path / "in"


def test_the_report_says_what_each_case_was_given_and_what_it_produced(tmp_path):
    scans = _study(tmp_path, "T1/case.nii.gz", "T2/case.nii.gz")
    sup = FakeSup(tmp_path, names=("T1/case", "T2/case"))

    out = run(scans=scans, output_dir=tmp_path / "out", sup=sup, outputs=["Oriented scan"])

    report = _report(out)
    assert report["summary"] == {"cases": 2, "cropped": 2, "partial": 0, "failed": 0}
    assert set(report["cases"]) == {"T1_case", "T2_case"}
    # `input` is the path the caller sent, relative to what it sent: the server narrows a replay by it.
    assert report["cases"]["T1_case"]["input"] == "T1/case.nii.gz"
    assert report["cases"]["T2_case"]["input"] == "T2/case.nii.gz"
    assert report["cases"]["T1_case"]["status"] == "ok"
    assert sorted(report["cases"]["T1_case"]["produced"]) == [
        "T1_case.json", "T1_case_L.nii.gz", "T1_case_R.nii.gz", "T1_case_oriented.nii.gz",
        "T1_case_qc.png"]


def test_every_file_in_the_output_belongs_to_exactly_one_case_of_the_report(tmp_path):
    """The server relates a file to its case through this list: a file nobody lists is orphaned."""
    sup = FakeSup(tmp_path, names=("a", "b"))

    out = run(scans=_study(tmp_path, "a.nii.gz", "b.nii.gz"), output_dir=tmp_path / "out", sup=sup,
              outputs=["Oriented scan", "Mandible segmentation"], sides="Left")

    listed = [path for entry in _report(out)["cases"].values() for path in entry["produced"]]
    assert sorted(listed) == sorted(p for p in _names(out) if p != "run_report.json")
    assert len(listed) == len(set(listed))
    for path in listed:
        assert (out / path).stat().st_size > 0, path


def test_a_file_sent_alone_is_its_own_input(tmp_path):
    scans = _study(tmp_path, "one.nii.gz") / "one.nii.gz"

    out = run(scans=scans, output_dir=tmp_path / "out", sup=FakeSup(tmp_path, names=("one",)))

    assert _report(out)["cases"]["one"]["input"] == "one.nii.gz"


def test_a_failed_case_is_in_the_report_with_its_reason_and_its_own_report_file(tmp_path):
    sup = FakeSup(tmp_path, names=("a", "b"), reasons={"b": "could not read /data/x/p_9.nii.gz"})
    original = sup.run

    def run_with_one_failure(tool, **params):
        folder = original(tool, **params)
        if tool == "ASO":
            report = json.loads((folder / "ASO_report.json").read_text())
            report["cases"]["b"].update(status="failed", produced=[])
            (folder / "ASO_report.json").write_text(json.dumps(report))
        return folder

    sup.run = run_with_one_failure
    out = run(scans=_study(tmp_path, "a.nii.gz", "b.nii.gz"), output_dir=tmp_path / "out", sup=sup)

    report = _report(out)
    assert report["summary"] == {"cases": 2, "cropped": 1, "partial": 0, "failed": 1}
    assert report["cases"]["b"]["produced"] == ["b.json"]
    assert report["cases"]["b"]["status"].startswith("orientation failed")
    assert "p_9" not in json.dumps(report) and "/data" not in json.dumps(report)
    assert report["cases"]["b"]["input"] == "b.nii.gz"


def test_the_report_carries_no_path_from_a_library_error(tmp_path, monkeypatch):
    sup = FakeSup(tmp_path, names=("a", "b"))
    real = crop.crop_scan

    def crop_scan(scan, landmarks, mask, reference, *args, **kwargs):
        if kwargs["name"] == "b":
            raise OSError("cannot read /data/clinic/secret.nii.gz")
        return real(scan, landmarks, mask, reference, *args, **kwargs)

    monkeypatch.setattr(crop, "crop_scan", crop_scan)
    out = run(scans=_study(tmp_path, "a.nii.gz", "b.nii.gz"), output_dir=tmp_path / "out", sup=sup,
              sides="Right")

    text = (out / "run_report.json").read_text()
    assert "secret" not in text and "/data" not in text
    assert _report(out)["cases"]["b"]["status"] == "crop failed: OSError"


def test_a_batch_that_cropped_nothing_still_raises_and_leaves_no_report(tmp_path):
    sup = FakeSup(tmp_path, mandible=False)

    with pytest.raises(ToolInputError):
        run(scans=_study(tmp_path, "a.nii.gz"), output_dir=tmp_path / "out", sup=sup)

    assert not (tmp_path / "out" / "run_report.json").exists()


def test_the_report_follows_the_servers_reading_of_it(tmp_path):
    """`execution/reports.py` reads `cases` -> `produced` (names) and `input`; same shape, no more."""
    out = run(scans=_study(tmp_path, "a.nii.gz"), output_dir=tmp_path / "out", sup=FakeSup(tmp_path))

    for name, entry in _report(out)["cases"].items():
        assert isinstance(name, str) and isinstance(entry["input"], str)
        assert all(isinstance(path, str) and "/" not in path for path in entry["produced"])


def test_an_oriented_scan_in_another_format_goes_through_convert_once(tmp_path):
    sup = FakeSup(tmp_path, names=("a", "b"), aso_extension=".nrrd")

    out = run(scans=tmp_path / "in", output_dir=tmp_path / "out", sup=sup)

    assert [tool for tool, _ in sup.calls].count("Convert") == 1
    assert {"a", "b"} <= {p.name.split("_")[0] for p in out.glob("*_R.nii.gz")}
    assert not [p for p in out.iterdir() if p.name.startswith(".tmj_")]


def test_a_nifti_oriented_scan_never_calls_convert(tmp_path, sup):
    run(scans=tmp_path / "in", output_dir=tmp_path / "out", sup=sup)

    assert "Convert" not in [tool for tool, _ in sup.calls]


def test_convert_is_asked_for_nifti_and_given_no_output_folder(tmp_path):
    sup = FakeSup(tmp_path, aso_extension=".nrrd")
    run(scans=tmp_path / "in", output_dir=tmp_path / "out", sup=sup)

    asked = sup.asked("Convert")
    assert asked["to"] == "nii.gz" and "output_dir" not in asked


def test_a_scan_convert_could_not_read_is_reported_and_alone_it_raises(tmp_path):
    sup = FakeSup(tmp_path, aso_extension=".nrrd", convert_fails=True)

    with pytest.raises(ToolInputError, match="converted to NIfTI"):
        run(scans=tmp_path / "in", output_dir=tmp_path / "out", sup=sup)


def test_a_case_with_one_joint_missing_is_partial_not_ok_and_the_panel_hears_of_it(tmp_path, monkeypatch):
    sup = FakeSup(tmp_path)
    real = crop.crop_scan

    def one_side_missing(*args, **kwargs):
        result = real(*args, **kwargs)
        result.report["sides"]["L"] = {"status": "no mandible near porion"}
        return result

    monkeypatch.setattr(crop, "crop_scan", one_side_missing)
    out = run(scans=tmp_path / "in", output_dir=tmp_path / "out", sup=sup)

    entry = json.loads((out / "run_report.json").read_text())["cases"]["a"]
    assert entry["status"] == "partial: L: no mandible near porion"
    assert any("partial" in message for _, message in sup.messages)


def test_a_case_that_fails_late_leaves_none_of_its_crops_behind(tmp_path, monkeypatch):
    sup = FakeSup(tmp_path, names=("a", "b"))

    def failing_image(result, reference, name, folder):
        if name == "b":
            raise OSError("disk full")

    monkeypatch.setattr(crop, "write_quality_image", failing_image)
    out = run(scans=tmp_path / "in", output_dir=tmp_path / "out", sup=sup)

    assert not (out / "b_R.nii.gz").exists() and not (out / "b_L.nii.gz").exists()
    assert (out / "a_R.nii.gz").exists() or (out / "a.json").exists()
    assert any("crop failed" in message for _, message in sup.messages)


def test_names_that_differ_only_by_case_stay_apart_and_never_stop_the_batch(tmp_path):
    from sadt_tmj_crop.dispatch import _unique_names

    names = _unique_names([("Case", "", {}, None, None, "Case"), ("case", "", {}, None, None, "case")])
    assert len({n.casefold() for n in names}) == 2
    clash = _unique_names([("x", "a", {}, None, None, "a/x"), ("x", "", {}, None, None, "x"),
                           ("a_x", "", {}, None, None, "a_x")])
    assert len({n.casefold() for n in clash}) == 3


def test_a_convert_that_fails_costs_only_the_scans_that_needed_it(tmp_path):
    sup = FakeSup(tmp_path, aso_extension=".nrrd")
    sup.run_original = sup.run

    def broken(tool, **params):
        if tool == "Convert":
            raise RuntimeError("boom")
        return sup.run_original(tool, **params)

    sup.run = broken
    with pytest.raises(ToolInputError, match="converted to NIfTI|oriented none"):
        run(scans=tmp_path / "in", output_dir=tmp_path / "out", sup=sup)


def test_the_crop_clips_and_rounds_voxels_above_the_int16_range(tmp_path):
    import nibabel as nib
    import numpy as np
    from conftest import reference_ras, write_scene

    USED = ["Ba", "S", "N", "RPo", "LPo", "ROr", "LOr"]

    scan, markups, mask = write_scene(tmp_path / "in", "a")
    image = nib.load(scan)
    nib.save(nib.Nifti1Image(np.asarray(image.dataobj) * 60.0, image.affine), scan)  # peaks at 60000
    out = tmp_path / "out"
    out.mkdir()
    crop.crop_scan(scan, markups, mask, reference_ras(), USED, {}, crop.CropSettings(), "a", str(out))
    data = np.asarray(nib.load(out / "a_R.nii.gz").dataobj)
    assert data.max() == 32767 and data.min() >= 0, "a wrapped value would be negative"


def test_a_failing_convert_leaves_the_nifti_scans_of_a_mixed_batch_ok(tmp_path):
    sup = FakeSup(tmp_path, names=("a", "b"), aso_extension=".nrrd")
    real_run = sup.run

    def mixed(tool, **params):
        result = real_run(tool, **params)
        if tool == "ASO":  # b stays NIfTI: ASO kept its input's format
            (result / "b_Or.nrrd").rename(result / "b_Or.nii.gz")
            report = json.loads((result / "ASO_report.json").read_text())
            report["cases"]["b"]["produced"][0] = "b_Or.nii.gz"
            (result / "ASO_report.json").write_text(json.dumps(report))
        if tool == "Convert":
            raise RuntimeError("boom")
        return result

    sup.run = mixed
    out = run(scans=tmp_path / "in", output_dir=tmp_path / "out", sup=sup)
    cases = json.loads((out / "run_report.json").read_text())["cases"]
    assert cases["b"]["status"] == "ok" and cases["a"]["status"].startswith("conversion failed")


def test_a_reason_quoting_a_file_name_with_spaces_keeps_none_of_it():
    from sadt_tmj_crop.dispatch import _safe_reason

    text = _safe_reason("could not read /work/in/Aaa Bbb Ccc.nii.gz: bad header")
    assert "Bbb" not in text and "Ccc" not in text and "Aaa" not in text
