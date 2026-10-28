"""scene.json and the landmark move every engine shares."""

import json

import pytest
from sadt_areg_common import scene

SHIFT = [[1, 0, 0, 3.0], [0, 1, 0, -2.0], [0, 0, 1, 1.0], [0, 0, 0, 1]]


def _markups(path, points):
    path.write_text(json.dumps({"markups": [{"coordinateSystem": "LPS", "controlPoints": [
        {"label": label, "position": position} for label, position in points.items()]}]}))
    return path


def test_markups_move_and_keep_their_file(tmp_path):
    source = _markups(tmp_path / "a.mrk.json", {"UR6O": [1.0, 2.0, 3.0]})
    scene.move_landmarks(str(source), SHIFT, str(tmp_path / "b.mrk.json"))
    payload = json.loads((tmp_path / "b.mrk.json").read_text())
    assert payload["markups"][0]["coordinateSystem"] == "LPS"
    assert payload["markups"][0]["controlPoints"][0]["position"] == pytest.approx([4.0, 0.0, 4.0])


def test_plain_landmarks_stay_plain(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps({"UR6O": [1, 2, 3], "note": "kept"}))
    scene.move_landmarks(str(tmp_path / "a.json"), SHIFT, str(tmp_path / "b.json"))
    payload = json.loads((tmp_path / "b.json").read_text())
    assert payload == {"UR6O": pytest.approx([4.0, 0.0, 4.0]), "note": "kept"}


def test_a_rotation_is_applied_too(tmp_path):
    turn = [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    source = _markups(tmp_path / "a.mrk.json", {"P": [1.0, 0.0, 0.0]})
    scene.move_landmarks(str(source), turn, str(tmp_path / "b.mrk.json"))
    moved = json.loads((tmp_path / "b.mrk.json").read_text())["markups"][0]["controlPoints"][0]
    assert moved["position"] == pytest.approx([0.0, 1.0, 0.0])


def test_the_scene_lists_files_relative_to_its_folder(tmp_path):
    for name in ("scan.nii.gz", "arch.vtk", "arch_lm.mrk.json", "arch.tfm"):
        (tmp_path / name).write_text("x")
    scene.write(str(tmp_path), "the CBCT", [
        (str(tmp_path / "scan.nii.gz"), "reference"),
        (str(tmp_path / "arch.vtk"), "registered"),
        (str(tmp_path / "arch_lm.mrk.json"), "moved"),
        (str(tmp_path / "arch.tfm"), "transform"),
        (str(tmp_path / "arch.vtk"), "registered"),
    ])
    written = json.loads((tmp_path / scene.SCENE_NAME).read_text())
    assert written == {"version": 1, "space": "the CBCT", "files": [
        {"path": "scan.nii.gz", "kind": "volume", "role": "reference"},
        {"path": "arch.vtk", "kind": "model", "role": "registered"},
        {"path": "arch_lm.mrk.json", "kind": "markups", "role": "moved"},
        {"path": "arch.tfm", "kind": "transform", "role": "transform"},
    ]}


def test_a_missing_file_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError, match="not written"):
        scene.write(str(tmp_path), "the CBCT", [(str(tmp_path / "gone.vtk"), "registered")])
