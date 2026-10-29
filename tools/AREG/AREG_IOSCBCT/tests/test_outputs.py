"""`outputs`: what a run hands back beside the meshes, all in the CBCT's space.

Registration mode, so everything runs for real. The cohort's CBCT landmarks
are the intraoral ones shifted by a known vector, so the registration is that
translation and the moved intraoral landmarks must land exactly on the CBCT's.
"""

import json
import os
import typing

import numpy as np
import pytest
from conftest import (
    UPPER_LABELS,
    cohort,
    moved,
    read_report,
    run_registration,
    tree_of,
    write_flat_landmarks,
    write_markups,
)
from sadt_areg_common import catalogs
from sadt_areg_ioscbct import pipeline, run

# What a Registration run can write: nothing in it was oriented, so the
# orientation landmarks are tested on their own below.
EVERYTHING = [catalogs.OUTPUT_LANDMARKS, catalogs.OUTPUT_CBCT_SCAN, catalogs.OUTPUT_TRANSFORMS]
OFFERED = [catalogs.OUTPUT_LANDMARKS, catalogs.OUTPUT_ORIENTATION_LANDMARKS,
           catalogs.OUTPUT_CBCT_SCAN, catalogs.OUTPUT_TRANSFORMS]


def _folder(tmp_path):
    return tmp_path / "out" / "P001"


def _scene(tmp_path):
    return json.loads((_folder(tmp_path) / "scene.json").read_text())


def test_the_options_are_the_catalogs():
    annotation = typing.get_type_hints(run)["outputs"]
    assert list(typing.get_args(typing.get_args(annotation)[0])) == OFFERED
    assert set(OFFERED) <= set(catalogs.OUTPUT_HELP)


def test_a_cbct_nobody_oriented_has_no_orientation_landmarks(tmp_path):
    cohort(tmp_path)
    run_registration(tmp_path, outputs=[catalogs.OUTPUT_ORIENTATION_LANDMARKS])
    assert set(os.listdir(_folder(tmp_path))) == {
        "P001_T2_U_Reg.vtk", "P001_T2_U_Reg_matrix.npy", "scene.json"}


def test_an_oriented_cbct_hands_back_the_landmarks_aso_oriented_it_on(tmp_path):
    """ASO writes `<patient>_lm_<suffix>.mrk.json` beside the scan it oriented.
    Asked for, it is copied beside the meshes, in the space they were
    registered into, and the scene lists it as what it is."""
    from sadt_areg_ioscbct import dispatch

    cohort(tmp_path)
    write_markups(tmp_path / "cbct" / "P_0001_lm_T2.mrk.json", {"Ba": [0.0, 0.0, 0.0]})
    (tmp_path / "out").mkdir()
    dispatch.register(
        str(tmp_path / "ios"), str(tmp_path / "cbct"), str(tmp_path / "ios_lm"),
        str(tmp_path / "cbct_lm"), str(tmp_path / "out"), "Reg", {"patients": {}}, 1.0,
        carry_cbct=True, outputs=[catalogs.OUTPUT_ORIENTATION_LANDMARKS])
    assert (_folder(tmp_path) / "P_0001_lm_T2.mrk.json").is_file()
    assert {"path": "P_0001_lm_T2.mrk.json", "kind": "markups",
            "role": "orientation landmarks"} in _scene(tmp_path)["files"]


def test_nothing_ticked_adds_only_the_scene(tmp_path):
    cohort(tmp_path)
    run_registration(tmp_path)
    assert tree_of(tmp_path / "out") == sorted([
        "AREG_report.json",
        os.path.join("P001", "P001_T2_U_Reg.vtk"),
        os.path.join("P001", "P001_T2_U_Reg_matrix.npy"),
        os.path.join("P001", "scene.json"),
    ])
    assert _scene(tmp_path)["files"] == [
        {"path": "P001_T2_U_Reg.vtk", "kind": "model", "role": "registered"}]


def test_everything_ticked_writes_the_whole_scene(tmp_path):
    cohort(tmp_path)
    run_registration(tmp_path, outputs=EVERYTHING)
    assert tree_of(tmp_path / "out") == sorted([
        "AREG_report.json",
        os.path.join("P001", "P001_T2_U_Reg.vtk"),
        os.path.join("P001", "P001_T2_U_Reg_matrix.npy"),
        os.path.join("P001", "P001_T2_U_Reg_lm.mrk.json"),
        os.path.join("P001", "P001_T2_U_Reg_transform.tfm"),
        os.path.join("P001", "P_0001_T2.nii.gz"),
        os.path.join("P001", "P_0001_T2_lm_Pred.mrk.json"),
        os.path.join("P001", "scene.json"),
    ])
    assert _scene(tmp_path) == {"version": 1, "space": "the CBCT", "files": [
        {"path": "P_0001_T2.nii.gz", "kind": "volume", "role": "reference"},
        {"path": "P_0001_T2_lm_Pred.mrk.json", "kind": "markups", "role": "reference landmarks"},
        {"path": "P001_T2_U_Reg.vtk", "kind": "model", "role": "registered"},
        {"path": "P001_T2_U_Reg_lm.mrk.json", "kind": "markups", "role": "moved landmarks"},
        {"path": "P001_T2_U_Reg_transform.tfm", "kind": "transform", "role": "registration"},
    ]}


@pytest.mark.parametrize("option, written", [
    (catalogs.OUTPUT_LANDMARKS, {"P001_T2_U_Reg_lm.mrk.json", "P_0001_T2_lm_Pred.mrk.json"}),
    (catalogs.OUTPUT_CBCT_SCAN, {"P_0001_T2.nii.gz"}),
    (catalogs.OUTPUT_TRANSFORMS, {"P001_T2_U_Reg_transform.tfm"}),
])
def test_each_option_writes_its_own_files_only(tmp_path, option, written):
    cohort(tmp_path)
    run_registration(tmp_path, outputs=[option])
    base = {"P001_T2_U_Reg.vtk", "P001_T2_U_Reg_matrix.npy", "scene.json"}
    assert set(os.listdir(_folder(tmp_path))) == base | written


def test_the_moved_landmarks_land_on_the_cbct_landmarks(tmp_path):
    cohort(tmp_path)
    run_registration(tmp_path, outputs=[catalogs.OUTPUT_LANDMARKS])

    moved_ios = pipeline.read_landmarks(str(_folder(tmp_path) / "P001_T2_U_Reg_lm.mrk.json"))
    expected = moved(UPPER_LABELS)
    assert set(moved_ios) == set(expected)
    for label, position in expected.items():
        assert moved_ios[label] == pytest.approx(position, abs=1e-5)


def test_the_moved_landmarks_keep_their_file_as_it_was(tmp_path):
    """Only the positions change: a markups file keeps its coordinate system."""
    cohort(tmp_path)
    run_registration(tmp_path, outputs=[catalogs.OUTPUT_LANDMARKS])

    payload = json.loads((_folder(tmp_path) / "P001_T2_U_Reg_lm.mrk.json").read_text())
    assert payload["markups"][0]["coordinateSystem"] == "LPS"
    assert [point["label"] for point in payload["markups"][0]["controlPoints"]] == list(UPPER_LABELS)


def test_plain_json_landmarks_stay_plain_json(tmp_path):
    cohort(tmp_path)
    os.remove(tmp_path / "ios_lm" / "P001_T2_U_lm_Pred.mrk.json")
    write_flat_landmarks(tmp_path / "ios_lm" / "P001_T2_U_lm.json", UPPER_LABELS)
    run_registration(tmp_path, outputs=[catalogs.OUTPUT_LANDMARKS])

    payload = json.loads((_folder(tmp_path) / "P001_T2_U_Reg_lm.json").read_text())
    first = next(iter(UPPER_LABELS))
    assert set(payload) == set(UPPER_LABELS)
    assert payload[first] == pytest.approx(moved(UPPER_LABELS)[first], abs=1e-5)


def test_the_tfm_is_the_inverse_of_the_point_moving_matrix(tmp_path):
    """Written the way AREG_IOS writes its own: a `.tfm` maps the result back."""
    import SimpleITK as sitk

    cohort(tmp_path)
    run_registration(tmp_path, outputs=[catalogs.OUTPUT_TRANSFORMS])

    matrix = np.load(str(_folder(tmp_path) / "P001_T2_U_Reg_matrix.npy"))
    transform = sitk.AffineTransform(
        sitk.ReadTransform(str(_folder(tmp_path) / "P001_T2_U_Reg_transform.tfm")))
    written = np.eye(4)
    written[:3, :3] = np.array(transform.GetMatrix()).reshape(3, 3)
    written[:3, 3] = transform.GetTranslation()
    assert written == pytest.approx(np.linalg.inv(matrix), abs=1e-6)


def test_the_report_names_what_was_written(tmp_path):
    cohort(tmp_path)
    run_registration(tmp_path, outputs=EVERYTHING)

    patient = read_report(tmp_path / "out")["patients"]["P001"]
    assert patient["cbct_copy"] == os.path.join("P001", "P_0001_T2.nii.gz")
    assert patient["cbct_landmarks"] == [os.path.join("P001", "P_0001_T2_lm_Pred.mrk.json")]
    mesh = patient["meshes"]["P001_T2_U.vtk"]
    assert mesh["landmarks"] == os.path.join("P001", "P001_T2_U_Reg_lm.mrk.json")
    assert mesh["transform"] == os.path.join("P001", "P001_T2_U_Reg_transform.tfm")


def test_both_arches_share_one_copy_of_the_cbct_landmarks(tmp_path):
    cohort(tmp_path, jaws=("U", "L"), landmarks=UPPER_LABELS)
    write_markups(tmp_path / "ios_lm" / "P001_T2_L_lm_Pred.mrk.json", UPPER_LABELS)
    run_registration(tmp_path, outputs=[catalogs.OUTPUT_LANDMARKS])

    roles = [(entry["path"], entry["role"]) for entry in _scene(tmp_path)["files"]]
    assert [path for path, role in roles if role == "reference landmarks"] == [
        "P_0001_T2_lm_Pred.mrk.json"]
    assert sorted(path for path, role in roles if role == "registered") == [
        "P001_T2_L_Reg.vtk", "P001_T2_U_Reg.vtk"]
