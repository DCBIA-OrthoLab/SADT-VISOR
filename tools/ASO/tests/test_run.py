"""Unit tests for the ASO tool.

No GPU and no real models: the landmark tool is stood in for by a fake
supervisor, and everything else -- discovery, pairing, recentring, registration,
output naming, the report -- runs for real against synthetic volumes and meshes.

Every test docstring names the defect of the original CLIs it prevents coming
back.

    cd tools/ASO && uv run pytest
"""

import json
import logging
import os
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk
import vtk

from sadt_aso import dispatch, run
from sadt_aso import catalogs, geometry, markups
from sadt_aso.cbct import icp as cbct_icp
from sadt_aso.cbct import pipeline as cbct_pipeline
from sadt_aso.errors import SupervisorRequired, ToolInputError
from sadt_aso.ios import icp as ios_icp
from sadt_aso.ios import pipeline as ios_pipeline
from sadt_aso.ios import surfaces


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

# Seven points in a plausible skull-ish arrangement: no three are collinear, so
# the coarse alignment can always find a usable triplet.
_REFERENCE_POINTS = {
    "Ba": np.array([0.0, -30.0, -20.0]),
    "S": np.array([0.0, -10.0, 5.0]),
    "N": np.array([0.0, 25.0, 10.0]),
    "RPo": np.array([-35.0, -20.0, -5.0]),
    "LPo": np.array([35.0, -20.0, -5.0]),
    "ROr": np.array([-25.0, 20.0, 0.0]),
    "LOr": np.array([25.0, 20.0, 0.0]),
}


def _rotate(points: dict, axis, degrees: float, offset=(0.0, 0.0, 0.0)) -> dict:
    matrix = geometry.rotation_matrix(np.array(axis, dtype=float), np.radians(degrees))
    return {
        name: matrix @ point + np.array(offset, dtype=float)
        for name, point in points.items()
    }


def _write_scan(path, size=(16, 16, 16), spacing=(1.0, 1.0, 1.0)):
    array = np.zeros(size[::-1], dtype=np.int16)
    array[4:12, 4:12, 4:12] = 800
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(spacing)
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    sitk.WriteImage(image, str(path), useCompression=True)
    return str(path)


def _write_markups(path, landmarks: dict):
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    return markups.write_landmarks(landmarks, str(path))


def _write_mesh(path, centroids: dict, array_name="Universal_ID", points_per_tooth=6):
    """A mesh with `points_per_tooth` points clustered around each centroid,
    labelled with that tooth's universal id."""
    points = vtk.vtkPoints()
    labels = vtk.vtkIntArray()
    labels.SetName(array_name)
    vertices = vtk.vtkCellArray()

    offsets = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0]],
        dtype=float,
    )
    for tooth_id, centroid in centroids.items():
        # Symmetric offsets, so the cluster's mean IS the centroid.
        for offset in offsets[:points_per_tooth]:
            point_id = points.InsertNextPoint(*(np.asarray(centroid) + offset))
            labels.InsertNextValue(int(tooth_id))
            vertices.InsertNextCell(1)
            vertices.InsertCellPoint(point_id)

    mesh = vtk.vtkPolyData()
    mesh.SetPoints(points)
    mesh.SetVerts(vertices)
    mesh.GetPointData().AddArray(labels)
    mesh.GetPointData().SetActiveScalars(array_name)

    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    return surfaces.write_surface(mesh, str(path))


_UPPER_CENTROIDS = {
    3: np.array([-20.0, 0.0, 0.0]),    # UR6
    9: np.array([0.0, 25.0, 2.0]),     # UL1
    14: np.array([20.0, 0.0, 0.0]),    # UL6
}
_LOWER_CENTROIDS = {
    19: np.array([18.0, -1.0, -12.0]),   # LL6
    25: np.array([0.0, 24.0, -10.0]),    # LR1
    30: np.array([-18.0, -1.0, -12.0]),  # LR6
}


def _cbct_case(root, key="patient1", points=None, extension=".nii.gz"):
    """One CBCT patient plus its landmarks, rotated away from the reference."""
    points = _REFERENCE_POINTS if points is None else points
    moved = _rotate(points, (0.2, 1.0, 0.3), 17.0, offset=(12.0, -8.0, 5.0))
    scan = _write_scan(os.path.join(str(root), f"{key}_scan{extension}"))
    _write_markups(os.path.join(str(root), f"{key}_lm.mrk.json"), moved)
    return scan


def _cbct_reference(root):
    path = os.path.join(str(root), "gold", "reference_lm.mrk.json")
    _write_markups(path, _REFERENCE_POINTS)
    return os.path.join(str(root), "gold")


def _ios_reference(root, with_surfaces=True, with_markups=False):
    folder = os.path.join(str(root), "gold_ios")
    for jaw, centroids in (("U", _UPPER_CENTROIDS), ("L", _LOWER_CENTROIDS)):
        if with_surfaces:
            _write_mesh(os.path.join(folder, f"Gold_{jaw}_Seg.vtk"), centroids)
        if with_markups:
            _write_markups(
                os.path.join(folder, f"Gold_{jaw}_lm.mrk.json"),
                _ios_landmarks(centroids),
            )
    return folder


def _ios_landmarks(centroids: dict) -> dict:
    """One 'O' landmark per tooth, at its centroid."""
    return {f"{catalogs.TOOTH_NAMES[tooth]}O": point for tooth, point in centroids.items()}


def _ios_case(root, key="P1", array_name="Universal_ID", with_markups=False, jaws=("U", "L")):
    moved = {}
    for jaw, centroids in (("U", _UPPER_CENTROIDS), ("L", _LOWER_CENTROIDS)):
        if jaw not in jaws:
            continue
        rotated = {
            tooth: geometry.rotation_matrix(np.array([0.1, 0.2, 1.0]), np.radians(12.0))
            @ point
            + np.array([5.0, -3.0, 2.0])
            for tooth, point in centroids.items()
        }
        moved[jaw] = rotated
        _write_mesh(
            os.path.join(str(root), f"{key}_{jaw}_Seg.vtk"), rotated, array_name=array_name
        )
        if with_markups:
            _write_markups(
                os.path.join(str(root), f"{key}_{jaw}_lm.mrk.json"), _ios_landmarks(rotated)
            )
    return moved


def _run_aso(tmp_path, **kwargs):
    """`run()` with an output directory chosen for the test.

    Returns it as a string, because that is what the assertions below read.
    """
    output_dir = kwargs.pop("output_dir", None) or (tmp_path / "out")
    return str(run(output_dir=output_dir, **kwargs))


def _report(output_dir: str) -> dict:
    with open(os.path.join(output_dir, dispatch.REPORT_NAME)) as handle:
        return json.load(handle)


# ---------------------------------------------------------------------------
# The published options and the catalogs cannot drift apart
# ---------------------------------------------------------------------------

def _choices(argument):
    """The `Literal` options `run()` publishes for one argument."""
    import typing

    hint = typing.get_type_hints(run)[argument]
    if typing.get_origin(hint) is list:
        hint = typing.get_args(hint)[0]
    return list(typing.get_args(hint))


def test_the_published_options_are_the_catalogs_own():
    """`Literal` takes literals only, so it cannot be built from the catalogs.
    That makes the signature a second declaration of the same sets, and this is
    what keeps them honest: an option added to one and not the other would be
    unselectable from the client, or offered and then refused."""
    assert _choices("cbct_landmarks") == list(catalogs.CBCT_LANDMARKS)
    assert _choices("ios_teeth") == list(catalogs.TOOTH_IDS)
    assert _choices("ios_landmark_types") == list(catalogs.IOS_LANDMARK_TYPES)
    assert _choices("ios_occlusion") == list(catalogs.OCCLUSION_CHOICES)
    assert _choices("modality") == list(catalogs.MODALITY_CHOICES)
    assert _choices("automation") == list(catalogs.AUTOMATION_CHOICES)


def test_the_published_defaults_are_the_catalogs_own():
    """A default outside its option list gives the client a picker that cannot
    produce the value the tool starts from."""
    import inspect

    parameters = inspect.signature(run).parameters
    assert set(parameters["cbct_landmarks"].default) == set(catalogs.DEFAULT_CBCT_LANDMARKS)
    assert set(parameters["ios_teeth"].default) == set(catalogs.DEFAULT_TEETH)
    assert set(parameters["ios_landmark_types"].default) == set(
        catalogs.DEFAULT_IOS_LANDMARK_TYPES
    )
    for argument in ("cbct_landmarks", "ios_teeth", "ios_landmark_types", "ios_jaws"):
        for value in parameters[argument].default:
            assert value in _choices(argument), (argument, value)


def test_mode_specific_arguments_are_optional():
    """A required argument belonging to the inactive mode would block every
    call in the other one -- the schema cannot say "only in mode X"."""
    import inspect

    parameters = inspect.signature(run).parameters
    optional = (
        "modality", "automation", "landmarks", "landmark_model", "cbct_landmarks",
        "ios_teeth", "ios_landmark_types", "ios_jaws", "ios_occlusion", "dicom_input",
        "output_suffix",
    )
    for name in optional:
        assert parameters[name].default is not inspect.Parameter.empty, name
    for name in ("input", "reference", "output_dir"):
        assert parameters[name].default is inspect.Parameter.empty, name


def test_the_supervisor_is_keyword_only_and_unannotated():
    """That shape is what `describe.py` reads to keep it out of the schema. A
    client never sends it, and a `Path` annotation would put a file picker on
    every ASO form."""
    import inspect

    parameter = inspect.signature(run).parameters["sup"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.annotation is inspect.Parameter.empty
    # Defaulted, so the tool stays callable standalone with landmarks supplied.
    assert parameter.default is None


def test_catalogs_have_no_duplicates_and_valid_defaults():
    """ALI shipped a UI saying "UR3OI" against a CLI expecting "UR3OIP". One
    vocabulary, defined once, with defaults that are members of it."""
    assert len(catalogs.CBCT_LANDMARKS) == len(set(catalogs.CBCT_LANDMARKS))
    assert len(catalogs.TOOTH_IDS) == 32
    assert sorted(catalogs.TOOTH_IDS.values()) == list(range(1, 33))
    for default in catalogs.DEFAULT_CBCT_LANDMARKS:
        assert default in catalogs.CBCT_LANDMARK_CHOICES
    for default in catalogs.DEFAULT_TEETH:
        assert default in catalogs.TOOTH_CHOICES
    # Three teeth per jaw: what the coarse alignment needs, out of the box.
    per_jaw = catalogs.split_by_jaw(catalogs.DEFAULT_TEETH)
    assert len(per_jaw["Upper"]) == 3 and len(per_jaw["Lower"]) == 3


def test_an_unknown_option_is_refused_not_dropped():
    """`Literal` is published, not enforced -- the runner calls run(**params)
    from a JSON object -- so a stale client naming an option that no longer
    exists must be told, not handed a narrower run than it asked for."""
    with pytest.raises(ToolInputError, match="NoSuchPoint"):
        dispatch._selected(["Ba", "NoSuchPoint"], catalogs.CBCT_LANDMARK_CHOICES, "cbct_landmarks")


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

def test_angle_is_clamped_at_both_ends():
    """np.arccos of a dot product rounding to 1.0000000002 is NaN, which then
    propagates through the rotation matrix into a scan full of zeros. The
    original clamped the upper end only."""
    for vector in ([1.0, 2.0, 3.0], [3.0, 1.0, 2.0]):
        angle, _ = geometry.angle_and_axis(np.array(vector), np.array(vector))
        assert np.isfinite(angle)
    angle, _ = geometry.angle_and_axis(np.array([1.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0]))
    assert np.isfinite(angle)


def test_rotation_matrix_survives_a_degenerate_axis():
    """Two already-parallel vectors give a zero cross product; dividing by its
    norm produced a matrix of NaN."""
    matrix = geometry.rotation_matrix(np.zeros(3), 0.5)
    assert np.allclose(matrix, np.eye(3))


def test_triplet_search_is_deterministic_and_leaves_numpy_alone():
    """The search drew from the process-global numpy generator, so the same
    request gave a different orientation each run and two concurrent requests
    consumed each other's state."""
    source = _rotate(_REFERENCE_POINTS, (1.0, 0.4, 0.2), 15.0)
    np.random.seed(1234)
    before = np.random.get_state()[1][0]
    first = geometry.best_triplet(source, _REFERENCE_POINTS, False, 2500, 0)
    second = geometry.best_triplet(source, _REFERENCE_POINTS, False, 2500, 0)
    assert first == second
    assert np.random.get_state()[1][0] == before


# ---------------------------------------------------------------------------
# CBCT: discovery and pairing
# ---------------------------------------------------------------------------

def test_same_named_scans_in_different_folders_stay_apart(tmp_path):
    """`GetPatients` keyed on the base name, so two `scan.nii.gz` in different
    subfolders became one patient -- in the working dict AND in the flat output
    folder."""
    _cbct_case(tmp_path / "input" / "siteA", key="scan")
    _cbct_case(tmp_path / "input" / "siteB", key="scan")
    found = cbct_pipeline.discover(str(tmp_path / "input"))
    assert sorted(found) == [os.path.join("siteA", "scan"), os.path.join("siteB", "scan")]


def test_timepoints_are_two_patients(tmp_path):
    """`GetPatients` stripped `_T1`/`_T2`, collapsing two timepoints of one
    subject into a single patient: the second scan was dropped and both
    landmark sets were merged onto the survivor."""
    _cbct_case(tmp_path / "input", key="P1_T1")
    _cbct_case(tmp_path / "input", key="P1_T2")
    assert sorted(cbct_pipeline.discover(str(tmp_path / "input"))) == ["P1_T1", "P1_T2"]


def test_a_landmark_file_matching_no_scan_is_reported_not_silently_dropped(tmp_path):
    """The failure a real run hit: 1 patient found, 0 oriented, and a report
    saying "no landmark file (.mrk.json) alongside this scan" while the file
    was sitting right next to the scan.

    `discover` keys every file by `patient_stem` and then drops any group with
    no scan in it. A landmark file whose stem does not collapse onto the
    scan's -- `P1_CBCT.nii.gz` beside `P1.mrk.json` -- becomes its own
    scan-less group and disappears without a word: not in the report, not in
    the logs. The caller is then told the file is missing, which is the one
    thing it is not.
    """
    root = tmp_path / "input"
    _write_scan(root / "P1_CBCT.nii.gz")
    _write_markups(str(root / "P1.mrk.json"), _REFERENCE_POINTS)

    found = cbct_pipeline.discover(str(root))
    assert list(found) == ["P1_CBCT"]
    assert found["P1_CBCT"]["markups"] == []

    orphans = cbct_pipeline.orphan_markups(str(root))
    assert [os.path.basename(path) for paths in orphans.values() for path in paths] == [
        "P1.mrk.json"
    ]


def test_landmark_files_that_do_match_are_not_reported_as_orphans(tmp_path):
    _cbct_case(tmp_path / "input")
    assert cbct_pipeline.orphan_markups(str(tmp_path / "input")) == {}


def test_the_run_report_names_the_unmatched_landmark_files(tmp_path):
    """What the user actually needs on screen: the file is there, and this is
    the name it would have to have."""
    root = tmp_path / "input"
    _write_scan(root / "P1_CBCT.nii.gz")
    _write_markups(str(root / "P1.mrk.json"), _REFERENCE_POINTS)

    report = dispatch.orient(
        input_path=str(root),
        reference_path=_cbct_reference(tmp_path),
        modality=catalogs.MODALITY_CBCT,
        automation=catalogs.AUTOMATION_SEMI,
        cbct_landmarks=list(_REFERENCE_POINTS),
        output_dir=str(tmp_path / "out"),
    ).report

    assert report["summary"]["oriented"] == 0
    assert report["unmatched_markups"] == ["P1.mrk.json"]
    reason = report["cases"]["P1_CBCT"]["reason"]
    # The file is NAMED, and so is the rule that would have paired it.
    assert "P1.mrk.json" in reason
    assert "_scan" in reason and "_lm" in reason


def test_a_previous_run_is_not_re_ingested(tmp_path):
    """`patient1_Or.nii.gz` sorts BEFORE `patient1_scan.nii.gz`, so running
    twice on the same folder would orient the first run's output; and the first
    run's `patient1_lm_Or.mrk.json` would overwrite the caller's landmarks
    during the merge. AMASSS had the same class of bug."""
    root = tmp_path / "input"
    _cbct_case(root)
    _write_scan(root / "patient1_Or.nii.gz")
    _write_markups(root / "patient1_lm_Or.mrk.json", {"Ba": np.full(3, 99.0)})

    found = cbct_pipeline.discover(str(root), "Or")
    assert found["patient1"]["scan"].endswith("patient1_scan.nii.gz")
    assert [os.path.basename(path) for path in found["patient1"]["markups"]] == [
        "patient1_lm.mrk.json"
    ]


def test_a_folder_of_only_previous_outputs_still_works(tmp_path):
    """Skipping previous outputs outright would refuse a folder someone
    deliberately re-orients."""
    root = tmp_path / "input"
    _write_scan(root / "patient1_Or.nii.gz")
    _write_markups(root / "patient1_lm_Or.mrk.json", _REFERENCE_POINTS)
    found = cbct_pipeline.discover(str(root), "Or")
    assert found["patient1"]["scan"].endswith("patient1_Or.nii.gz")


def test_ios_previous_outputs_are_set_aside_too(tmp_path):
    root = tmp_path / "input"
    _ios_case(root, key="P1", jaws=("U",))
    _write_mesh(root / "P1_U_Seg_Or.vtk", _UPPER_CENTROIDS)
    found = ios_pipeline.discover(str(root), "Or")
    assert found["P1"]["Upper"]["surface"].endswith("P1_U_Seg.vtk")


def test_landmark_files_of_one_patient_are_merged_in_memory(tmp_path):
    """`MergeJson` merged a patient's per-group files by writing the result INTO
    the caller's input folder and deleting the sources."""
    root = tmp_path / "input"
    _write_scan(root / "P1_scan.nii.gz")
    _write_markups(root / "P1_lm_Pred_CB.mrk.json", {"Ba": np.zeros(3), "S": np.ones(3)})
    _write_markups(root / "P1_lm_Pred_U.mrk.json", {"N": np.full(3, 2.0)})

    found = cbct_pipeline.discover(str(root))
    merged = cbct_pipeline.load_landmarks(found["P1"]["markups"])
    assert sorted(merged) == ["Ba", "N", "S"]
    assert sorted(os.listdir(root)) == [
        "P1_lm_Pred_CB.mrk.json", "P1_lm_Pred_U.mrk.json", "P1_scan.nii.gz",
    ]


# ---------------------------------------------------------------------------
# CBCT: registration
# ---------------------------------------------------------------------------

def test_registration_needs_no_transform_file(tmp_path):
    """`SEMI_ASO_CBCT` read `data["tfm"]` unconditionally, but only the
    fully-automated chain ever produced one -- so every patient of a
    semi-automated run died on a KeyError caught 90 lines above."""
    source = _rotate(_REFERENCE_POINTS, (1.0, 0.3, 0.2), 12.0)
    registration = cbct_icp.register(
        source, _REFERENCE_POINTS, list(_REFERENCE_POINTS), pre_transform=None
    )
    assert registration.used == sorted(_REFERENCE_POINTS)
    assert registration.dropped == {}


def test_a_landmark_absent_from_the_reference_is_reported_not_fatal():
    """`GetDistDifference` indexed the reference's table with the INPUT's keys,
    so one extra landmark raised a KeyError that lost the whole patient."""
    source = dict(_rotate(_REFERENCE_POINTS, (1.0, 0.0, 0.0), 10.0))
    source["Extra"] = np.array([100.0, 100.0, 100.0])
    registration = cbct_icp.register(
        source, _REFERENCE_POINTS, list(source), pre_transform=None
    )
    assert registration.dropped["Extra"] == "not in the reference landmark file"
    assert "Ba" in registration.used


def test_too_few_landmarks_says_why():
    """`ICP()` returned (None, None, None) and the caller logged only 'ICP
    registration returned None'."""
    source = _rotate(_REFERENCE_POINTS, (1.0, 0.0, 0.0), 5.0)
    with pytest.raises(cbct_icp.RegistrationError) as raised:
        cbct_icp.register(source, _REFERENCE_POINTS, ["Ba", "S"], pre_transform=None)
    assert "2 usable landmark" in str(raised.value)


def test_registration_recovers_the_reference_frame():
    """The whole point: a rotated landmark set comes back onto the reference."""
    source = _rotate(_REFERENCE_POINTS, (0.3, 1.0, 0.2), 20.0)
    registration = cbct_icp.register(
        source, _REFERENCE_POINTS, list(_REFERENCE_POINTS), pre_transform=None
    )
    residual = geometry.mean_distance(registration.landmarks, _REFERENCE_POINTS)
    assert residual < 1.0, f"registration left {residual:.2f} mm of error"


def test_recentring_puts_the_volume_centre_on_the_origin(tmp_path):
    """All of what PRE_ASO_CBCT still does, once its learned orientation step
    was removed and its model_folder/SmallFOV/temp_folder arguments became
    dead."""
    path = _write_scan(tmp_path / "scan.nii.gz", size=(16, 16, 16), spacing=(0.5, 0.5, 0.5))
    image = sitk.ReadImage(path)
    original_center = np.array(
        image.TransformContinuousIndexToPhysicalPoint(
            (np.array(image.GetSize()) / 2.0).tolist()
        )
    )
    centered, translation = cbct_pipeline.recenter(image)
    new_center = np.array(
        centered.TransformContinuousIndexToPhysicalPoint(
            (np.array(centered.GetSize()) / 2.0).tolist()
        )
    )
    assert np.allclose(new_center, np.zeros(3), atol=1e-6)
    # The reported translation is what moved it there, which is what
    # center_landmarks applies to the points and what the .tfm composes back.
    assert np.allclose(np.array(translation.GetOffset()), -original_center, atol=1e-6)
    assert centered.GetSize() == image.GetSize()


def test_the_written_landmarks_overlay_the_written_volume(tmp_path):
    """The one thing that must be exact. The volume is moved by resampling and
    the landmarks by a matrix, through two different code paths -- if they
    disagree, the markups file opens in Slicer floating beside the scan it
    belongs to, and nothing in the report would say so."""
    path = _write_scan(tmp_path / "p1_scan.nii.gz", size=(24, 24, 24), spacing=(0.4, 0.4, 0.4))
    centered, pre = cbct_pipeline.prepare(path)
    source = cbct_pipeline.center_landmarks(
        _rotate(_REFERENCE_POINTS, (0.2, 1.0, 0.3), 17.0, offset=(12.0, -8.0, 5.0)), pre
    )
    registration = cbct_icp.register(source, _REFERENCE_POINTS, list(_REFERENCE_POINTS), pre)

    # resample_transform maps an OUTPUT point back to an INPUT point, so its
    # inverse is the forward move the voxels actually underwent.
    forward = registration.resample_transform.GetInverse()
    for name, point in source.items():
        assert np.allclose(
            forward.TransformPoint(point.tolist()), registration.landmarks[name], atol=1e-6
        ), name


def test_the_transform_file_maps_the_result_back_to_the_original(tmp_path):
    """The .tfm is what lets a clinician carry a measurement made on the
    oriented scan back onto the acquisition. Its direction is ORIENTED ->
    ORIGINAL, recentring included -- worth asserting rather than assuming, since
    getting it backwards is silent and the file still loads."""
    root = tmp_path / "input"
    original = _rotate(_REFERENCE_POINTS, (0.2, 1.0, 0.3), 17.0, offset=(12.0, -8.0, 5.0))
    _write_scan(root / "p1_scan.nii.gz", size=(24, 24, 24), spacing=(0.4, 0.4, 0.4))
    _write_markups(root / "p1_lm.mrk.json", original)

    run = dispatch.orient(
        input_path=str(root),
        reference_path=_cbct_reference(tmp_path),
        modality=catalogs.MODALITY_CBCT,
        automation=catalogs.AUTOMATION_SEMI,
        cbct_landmarks=list(_REFERENCE_POINTS),
        output_dir=str(tmp_path / "out"),
    )
    oriented = markups.load_landmarks(os.path.join(run.output_dir, "p1_lm_Or.mrk.json"))
    transform = sitk.ReadTransform(os.path.join(run.output_dir, "p1_Or_transform.tfm"))

    for name, point in oriented.items():
        assert np.allclose(
            transform.TransformPoint(point.tolist()), original[name], atol=1e-6
        ), name


def test_landmarks_follow_the_recentring(tmp_path):
    """Recentring the volume without moving the landmarks compares a centred
    scan against uncentred points -- the state the semi-automated CLI ran in,
    since it recentred nothing and then read a .tfm it never produced."""
    path = _write_scan(tmp_path / "scan.nii.gz")
    image = sitk.ReadImage(path)
    _, translation = cbct_pipeline.recenter(image)
    moved = cbct_pipeline.center_landmarks({"Ba": np.zeros(3)}, translation)
    assert np.allclose(moved["Ba"], np.array(translation.GetOffset()))


# ---------------------------------------------------------------------------
# CBCT: end to end
# ---------------------------------------------------------------------------

def test_semi_automated_cbct_end_to_end(tmp_path):
    """The mode that must work on day one, with no server-side model at all."""
    _cbct_case(tmp_path / "input")
    run = dispatch.orient(
        input_path=str(tmp_path / "input"),
        reference_path=_cbct_reference(tmp_path),
        modality=catalogs.MODALITY_CBCT,
        automation=catalogs.AUTOMATION_SEMI,
        cbct_landmarks=list(_REFERENCE_POINTS),
        output_dir=str(tmp_path / "out"),
    )
    entry = run.report["cases"]["patient1"]
    assert entry["status"] == "ok"
    assert set(entry["produced"]) == {
        "patient1_Or.nii.gz", "patient1_lm_Or.mrk.json", "patient1_Or_transform.tfm",
    }
    assert run.report["summary"] == {"cases": 1, "oriented": 1, "failed": 0}
    for name in entry["produced"]:
        assert os.path.getsize(os.path.join(run.output_dir, name)) > 0


def test_the_input_tree_is_untouched(tmp_path):
    """The original wrote a merged JSON into the input folder, deleted the
    sources, and converted DICOM into `<input>/NIFTI/` -- which a later run then
    re-discovered as input scans."""
    root = tmp_path / "input"
    _cbct_case(root)
    before = {
        name: os.path.getsize(os.path.join(root, name)) for name in sorted(os.listdir(root))
    }
    dispatch.orient(
        input_path=str(root),
        reference_path=_cbct_reference(tmp_path),
        modality=catalogs.MODALITY_CBCT,
        automation=catalogs.AUTOMATION_SEMI,
        cbct_landmarks=list(_REFERENCE_POINTS),
        output_dir=str(tmp_path / "out"),
    )
    after = {
        name: os.path.getsize(os.path.join(root, name)) for name in sorted(os.listdir(root))
    }
    assert after == before


@pytest.mark.parametrize(
    "given,expected", [(".nii", ".nii.gz"), (".nii.gz", ".nii.gz"), (".nrrd", ".nrrd")]
)
def test_output_keeps_the_input_format_and_is_compressed(tmp_path, given, expected):
    """A scan sent uncompressed must not come back as a 191 MB file per patient;
    and ITK has no ".nrrd.gz" writer at all, so NRRD keeps its own extension."""
    _cbct_case(tmp_path / "input", extension=given)
    run = dispatch.orient(
        input_path=str(tmp_path / "input"),
        reference_path=_cbct_reference(tmp_path),
        modality=catalogs.MODALITY_CBCT,
        automation=catalogs.AUTOMATION_SEMI,
        cbct_landmarks=list(_REFERENCE_POINTS),
        output_dir=str(tmp_path / "out"),
    )
    assert f"patient1_Or{expected}" in run.report["cases"]["patient1"]["produced"]


def test_a_patient_without_landmarks_fails_alone(tmp_path):
    """One bad case must not take the batch down: the original wrote a
    `<name>Error.txt` into the OUTPUT folder and carried on silently."""
    root = tmp_path / "input"
    _cbct_case(root, key="good")
    _write_scan(root / "orphan_scan.nii.gz")
    run = dispatch.orient(
        input_path=str(root),
        reference_path=_cbct_reference(tmp_path),
        modality=catalogs.MODALITY_CBCT,
        automation=catalogs.AUTOMATION_SEMI,
        cbct_landmarks=list(_REFERENCE_POINTS),
        output_dir=str(tmp_path / "out"),
    )
    assert run.report["cases"]["good"]["status"] == "ok"
    assert run.report["cases"]["orphan"]["status"] == "failed"
    assert "no landmark file" in run.report["cases"]["orphan"]["reason"]
    assert run.report["summary"] == {"cases": 2, "oriented": 1, "failed": 1}


def test_the_output_tree_mirrors_the_input(tmp_path):
    """Two patients in different folders must land in different folders."""
    _cbct_case(tmp_path / "input" / "siteA", key="scan")
    _cbct_case(tmp_path / "input" / "siteB", key="scan")
    run = dispatch.orient(
        input_path=str(tmp_path / "input"),
        reference_path=_cbct_reference(tmp_path),
        modality=catalogs.MODALITY_CBCT,
        automation=catalogs.AUTOMATION_SEMI,
        cbct_landmarks=list(_REFERENCE_POINTS),
        output_dir=str(tmp_path / "out"),
    )
    assert os.path.isfile(os.path.join(run.output_dir, "siteA", "scan_Or.nii.gz"))
    assert os.path.isfile(os.path.join(run.output_dir, "siteB", "scan_Or.nii.gz"))


def test_a_reference_of_landmarks_alone_is_enough(tmp_path):
    """`ExtractFilesFromFolder(..., gold=True)` took `scan_files[0]`, so a
    reference bundle holding only the landmarks it registers against died on an
    IndexError -- while the scan it demanded was read and never used."""
    landmarks = cbct_pipeline.load_reference(_cbct_reference(tmp_path))
    assert sorted(landmarks) == sorted(_REFERENCE_POINTS)


def test_the_run_is_reproducible(tmp_path):
    """Same input, same seed, same transform. Not true while the triplet search
    drew from the global numpy generator."""
    transforms = []
    for index in (0, 1):
        _cbct_case(tmp_path / f"input{index}")
        run = dispatch.orient(
            input_path=str(tmp_path / f"input{index}"),
            reference_path=_cbct_reference(tmp_path),
            modality=catalogs.MODALITY_CBCT,
            automation=catalogs.AUTOMATION_SEMI,
            cbct_landmarks=list(_REFERENCE_POINTS),
            output_dir=str(tmp_path / f"out{index}"),
        )
        with open(os.path.join(run.output_dir, "patient1_lm_Or.mrk.json")) as handle:
            transforms.append(handle.read())
    assert transforms[0] == transforms[1]


# ---------------------------------------------------------------------------
# CBCT: fully automated -- the supervisor seam
# ---------------------------------------------------------------------------

class FakeSup:
    """A supervisor, as a tool sees one: duck-typed, five members, no import.

    Records what it was asked for, and writes the landmark files the real tool
    would have written -- into the output directory it is handed, so the seam
    (write there, read back, merge per patient) is exercised rather than
    stubbed over.
    """

    def __init__(self, predictions: dict, tmp_path=None):
        self.predictions = predictions
        self.out = tmp_path
        self.tmp = tmp_path
        self.calls = []
        self.messages = []
        self.call_index = 0

    def run(self, tool, **params):
        # The input is captured HERE, not after the run: it lives in the
        # working directory, which is removed before `orient` returns.
        self.calls.append((tool, params))
        # Read into memory, not just listed: the files live in the working
        # directory, which is removed before `orient` returns.
        self.sent = [
            sitk.ReadImage(os.path.join(root, name))
            for root, _dirs, files in os.walk(str(params["input"]))
            for name in sorted(files)
        ]
        # The SUPERVISOR owns where a callee writes, exactly as the server owns
        # it for a request: the caller passes no `output_dir` and reads the path
        # back off the return value. Modelled here because a fake that let the
        # caller name the directory would pass whatever the caller did.
        self.call_index += 1
        output_dir = os.path.join(
            str(self.tmp), "sup", f"{self.call_index:02d}_{tool}", "output"
        )
        os.makedirs(output_dir, exist_ok=True)
        for key, landmarks in self.predictions.items():
            markups.write_landmarks(
                landmarks, os.path.join(output_dir, f"{key}_lm_Pred.mrk.json")
            )
        from pathlib import Path

        return Path(output_dir)

    def progress(self, fraction, message):
        self.messages.append((fraction, message))

    def log(self, message):
        self.messages.append((None, message))


def _predicted():
    return _rotate(_REFERENCE_POINTS, (0.2, 1.0, 0.3), 14.0)


def test_fully_automated_cbct_says_so_when_there_is_no_supervisor(tmp_path):
    """Nothing about the request is wrong -- there is just no way to reach the
    landmark tool. So the message names the way forward rather than an argument
    to change, and it is a class of its own so a caller can tell it apart."""
    with pytest.raises(SupervisorRequired) as raised:
        _run_aso(
            tmp_path, input=str(tmp_path), reference=str(tmp_path),
            modality="CBCT", automation="Fully-Automated",
            landmark_model="CBCT_landmark_model_ALI",
        )
    message = str(raised.value)
    assert dispatch.LANDMARK_TOOL in message
    assert "Semi-Automated" in message
    assert "landmarks" in message


def test_the_missing_supervisor_is_reported_before_the_missing_model(tmp_path):
    """With no way to run the landmark tool, naming a model cannot help -- so
    that is what the caller is told, whether or not they picked one."""
    with pytest.raises(SupervisorRequired):
        _run_aso(
            tmp_path, input=str(tmp_path), reference=str(tmp_path),
            modality="CBCT", automation="Fully-Automated",
        )


def test_fully_automated_cbct_asks_for_landmarks_not_for_weights(tmp_path):
    """ASO names no bundle. Which weights place the landmarks is the landmark
    tool's business, and it finds its own from its own data folder -- this used
    to compose a path into ALI's storage, which meant ASO holding a name for its
    neighbour's layout and a copy of its weights beside its own references."""
    sup = FakeSup({}, tmp_path)
    with pytest.raises(Exception):
        # Fails later, on the empty input -- what matters is that it got past
        # the argument check without a bundle.
        _run_aso(
            tmp_path, input=str(tmp_path), reference=str(tmp_path),
            modality="CBCT", automation="Fully-Automated", sup=sup,
        )
    for call in getattr(sup, "calls", []):
        assert "model" not in call.get("params", {}), call


def test_a_named_bundle_is_still_passed_on(tmp_path):
    """An override, not a requirement: a caller pinning a bundle is recording
    which weights ran, and is obeyed."""
    sup = FakeSup({}, tmp_path)
    with pytest.raises(Exception):
        _run_aso(
            tmp_path, input=str(tmp_path), reference=str(tmp_path),
            modality="CBCT", automation="Fully-Automated",
            landmark_model="MyBundle", sup=sup,
        )
    passed = [c for c in getattr(sup, "calls", []) if "model" in c.get("params", {})]
    assert not passed or passed[0]["params"]["model"] == "MyBundle"


def test_fully_automated_cbct_runs_through_the_supervisor(tmp_path):
    """The seam end to end: predictions replace the caller's markups and
    nothing else changes."""
    root = tmp_path / "input"
    _write_scan(root / "patient1_scan.nii.gz")
    sup = FakeSup({"patient1": _predicted()}, tmp_path)

    output_dir = _run_aso(
        tmp_path, input=str(root), reference=_cbct_reference(tmp_path),
        modality="CBCT", automation="Fully-Automated",
        landmark_model="CBCT_landmark_model_ALI",
        cbct_landmarks=list(_REFERENCE_POINTS), sup=sup,
    )
    report = _report(output_dir)
    assert report["cases"]["patient1"]["status"] == "ok"
    assert report["landmark_source"] == dispatch.LANDMARK_TOOL


def test_the_landmark_tool_is_asked_by_name_for_the_points_aso_needs(tmp_path):
    """By NAME, not by region: ASO's seven points straddle two of the landmark
    tool's regions, so asking by region would run 58 agents to use 7 -- and one
    agent is a full two-scale walk of the volume."""
    root = tmp_path / "input"
    _write_scan(root / "patient1_scan.nii.gz")
    sup = FakeSup({"patient1": _predicted()}, tmp_path)

    _run_aso(
        tmp_path, input=str(root), reference=_cbct_reference(tmp_path),
        modality="CBCT", automation="Fully-Automated",
        landmark_model="/data/ALI/models/Bundle",
        cbct_landmarks=list(_REFERENCE_POINTS), sup=sup,
    )

    assert len(sup.calls) == 1
    tool, params = sup.calls[0]
    # A string, not a dynamic attribute: a typo here is greppable, and the call
    # graph stays inspectable.
    assert tool == "ALI_CBCT"
    assert set(params["landmarks"]) == set(_REFERENCE_POINTS)
    assert params["model"] == "/data/ALI/models/Bundle"
    assert "cbct_regions" not in params


def test_the_landmark_tool_is_run_on_the_recentred_scans(tmp_path):
    """The ordering that made the supervisor necessary at all.

    The Slicer chain ran PRE_ASO_CBCT before ALI_CBCT, so the landmark tool has
    always seen centred volumes. Recentring is a pure metadata change and this
    *ought* to be reorderable -- but ALI takes the absolute value of the origin
    (issue #11), so it is kept exactly as it was rather than bet on.
    """
    root = tmp_path / "input"
    _write_scan(root / "patient1_scan.nii.gz")
    sup = FakeSup({"patient1": _predicted()}, tmp_path)

    _run_aso(
        tmp_path, input=str(root), reference=_cbct_reference(tmp_path),
        modality="CBCT", automation="Fully-Automated",
        landmark_model="Bundle", cbct_landmarks=list(_REFERENCE_POINTS), sup=sup,
    )

    _tool, params = sup.calls[0]
    assert len(sup.sent) == 1
    sent = sup.sent[0]
    centre = np.array(
        sent.TransformContinuousIndexToPhysicalPoint(
            (np.array(sent.GetSize()) / 2.0).tolist()
        )
    )
    # What it was handed is centred on the physical origin, not the original.
    assert np.allclose(centre, 0.0, atol=1e-6)
    assert str(params["input"]) != str(root)


def test_supplied_landmarks_need_no_supervisor(tmp_path):
    """The standalone case: run the landmark tool yourself, pass the folder.

    This is what makes the tool usable from a notebook with nothing but its own
    venv -- no repository checkout, no server, no supervisor.
    """
    root = tmp_path / "input"
    _write_scan(root / "patient1_scan.nii.gz")
    supplied = tmp_path / "landmarks"
    _write_markups(supplied / "patient1_lm_Pred.mrk.json", _predicted())

    output_dir = _run_aso(
        tmp_path, input=str(root), reference=_cbct_reference(tmp_path),
        modality="CBCT", automation="Fully-Automated",
        landmarks=str(supplied), cbct_landmarks=list(_REFERENCE_POINTS),
    )
    report = _report(output_dir)
    assert report["cases"]["patient1"]["status"] == "ok"
    assert report["landmark_source"] == "supplied"


def test_supplied_landmarks_win_over_the_supervisor(tmp_path):
    """Passing what you already have must not spend a GPU re-predicting it."""
    root = tmp_path / "input"
    _write_scan(root / "patient1_scan.nii.gz")
    supplied = tmp_path / "landmarks"
    _write_markups(supplied / "patient1_lm_Pred.mrk.json", _predicted())
    sup = FakeSup({"patient1": _predicted()}, tmp_path)

    _run_aso(
        tmp_path, input=str(root), reference=_cbct_reference(tmp_path),
        modality="CBCT", automation="Fully-Automated",
        landmarks=str(supplied), cbct_landmarks=list(_REFERENCE_POINTS), sup=sup,
    )
    assert sup.calls == []


def test_the_supervisor_is_never_imported(tmp_path):
    """It is duck-typed on purpose: importing a supervisor type would need a
    package shared with the server, which is the coupling the split removes."""
    source = open(dispatch.__file__).read()
    assert "import sup" not in source
    assert "supervisor import" not in source


# ---------------------------------------------------------------------------
# CBCT: DICOM input
# ---------------------------------------------------------------------------

def _write_dicom_series(directory, size=(8, 8, 6)):
    """A minimal but genuinely readable CT series."""
    os.makedirs(str(directory), exist_ok=True)
    array = np.zeros(size[::-1], dtype=np.int16)
    array[2:5, 2:6, 2:6] = 500
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((0.5, 0.5, 1.0))

    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    series_uid = "1.2.826.0.1.3680043.2.1125.1234567890"
    for index in range(image.GetDepth()):
        slice_image = image[:, :, index]
        position = "\\".join(
            str(value) for value in image.TransformIndexToPhysicalPoint((0, 0, index))
        )
        for tag, value in (
            ("0008|0060", "CT"), ("0020|000e", series_uid), ("0020|0032", position),
            ("0020|0013", str(index)), ("0028|0030", "0.5\\0.5"), ("0018|0050", "1.0"),
            ("0020|0037", "1\\0\\0\\0\\1\\0"),
        ):
            slice_image.SetMetaData(tag, value)
        writer.SetFileName(os.path.join(str(directory), f"slice{index:03d}.dcm"))
        writer.Execute(slice_image)
    return str(directory)


def test_dicom_conversion_never_writes_into_the_input(tmp_path):
    """`convertdicom2nifti` wrote `<input>/NIFTI/` -- into the caller's own
    data, which a later run then re-discovered as input scans."""
    root = tmp_path / "input"
    _write_dicom_series(root / "patientA")
    _write_markups(root / "patientA_lm.mrk.json", _REFERENCE_POINTS)
    before = sorted(os.listdir(root))

    converted = dicom_module().convert_tree(str(root), str(tmp_path / "nifti"))
    assert sorted(os.listdir(root)) == before
    assert os.path.isfile(os.path.join(converted, "patientA.nii.gz"))
    # Markups travel with the scans, which is what lets Semi-Automated take DICOM.
    assert os.path.isfile(os.path.join(converted, "patientA_lm.mrk.json"))


def test_nested_dicom_exports_keep_their_tree(tmp_path):
    """The original listed one level of subfolders, so a nested export was
    invisible -- and two patients with the same folder name under different
    parents would have collided in its flat NIFTI/ output."""
    root = tmp_path / "input"
    _write_dicom_series(root / "siteA" / "scan")
    _write_dicom_series(root / "siteB" / "scan")
    converted = dicom_module().convert_tree(str(root), str(tmp_path / "nifti"))
    assert os.path.isfile(os.path.join(converted, "siteA", "scan.nii.gz"))
    assert os.path.isfile(os.path.join(converted, "siteB", "scan.nii.gz"))


def test_input_without_a_dicom_series_says_so(tmp_path):
    root = tmp_path / "input"
    _write_scan(root / "patient1.nii.gz")
    with pytest.raises(RuntimeError) as raised:
        dicom_module().convert_tree(str(root), str(tmp_path / "nifti"))
    assert "dicom_input" in str(raised.value)


def test_semi_automated_cbct_from_dicom_end_to_end(tmp_path):
    """The whole DICOM path, through main()'s own argument."""
    root = tmp_path / "input"
    _write_dicom_series(root / "patient1")
    _write_markups(
        root / "patient1_lm.mrk.json",
        _rotate(_REFERENCE_POINTS, (0.2, 1.0, 0.3), 15.0, offset=(6.0, -4.0, 3.0)),
    )
    output_dir = _run_aso(tmp_path, 
        input=str(root),
        modality="CBCT",
        automation="Semi-Automated",
        reference=_cbct_reference(tmp_path),
        cbct_landmarks=list(_REFERENCE_POINTS),
        dicom_input=True,
    )
    assert _report(output_dir)["cases"]["patient1"]["status"] == "ok"
    assert os.path.isfile(os.path.join(output_dir, "patient1_Or.nii.gz"))


def dicom_module():
    """Imported through a function so the DICOM tests state the seam explicitly."""
    from sadt_aso.cbct import dicom

    return dicom


# ---------------------------------------------------------------------------
# IOS
# ---------------------------------------------------------------------------

def test_a_file_that_does_not_name_its_jaw_is_refused(tmp_path):
    """`UpperOrLower` DEFAULTED TO LOWER, so a maxillary scan named
    `patient1.vtk` was registered against the mandibular reference and returned
    as a success."""
    with pytest.raises(ios_pipeline.JawError):
        ios_pipeline.patient_and_jaw("patient1.vtk")
    assert ios_pipeline.patient_and_jaw("P1_U_Seg.vtk") == ("P1", "Upper")
    assert ios_pipeline.patient_and_jaw("P1_Lower.vtk") == ("P1", "Lower")


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("Upper_gold.vtk", ("gold", "Upper")),
        ("Lower_gold.vtk", ("gold", "Lower")),
        ("Upper_gold.json", ("gold", "Upper")),
    ],
)
def test_the_published_reference_bundle_is_readable(filename, expected):
    """The IOS reference published as HUTIN1/ASO v1.0.0 Gold_file.zip names the
    jaw FIRST -- `Upper_gold.vtk`. Requiring an identifier before the jaw token
    rejected the entire bundle with "no mesh whose name says which jaw it is",
    which would have made Fully-Automated IOS unusable with the only reference
    anyone ships. Verified against the real archive."""
    assert ios_pipeline.patient_and_jaw(filename) == expected


@pytest.mark.parametrize(
    "filename,expected",
    [
        # What ALI and Crown_Seg actually write beside a mesh.
        ("Upper_new_9_Upper_O_Pred.json", ("new_9", "Upper")),
        ("Lower_new_9_Lower_O_Pred.json", ("new_9", "Lower")),
        # The mesh they were predicted from: the two must agree, or the pair
        # lands under two patients and neither has what it needs.
        ("Upper_new_9.vtk", ("new_9", "Upper")),
        ("Lower_new_9.vtk", ("new_9", "Lower")),
        # A second jaw token ends the identifier; one alone does not.
        ("Upper_gold.vtk", ("gold", "Upper")),
        ("Upper_P1_Lower_Seg.vtk", ("P1", "Upper")),
    ],
)
def test_a_prediction_suffix_is_decoration_not_part_of_the_patient(filename, expected):
    """`Upper_new_9_Upper_O_Pred.json` is the landmark file of
    `Upper_new_9.vtk`, and the two have to land on the same patient key.

    They did not: with the jaw token first, the identifier ran to the end of
    the name, so the mesh was `new_9` and its landmarks `new_9_Upper_O_Pred`.
    Semi-Automated IOS then failed with "no landmark file for this jaw" on
    `DATA/ASO/testfiles/IOS_SemiAuto`, the dataset this repository's own
    manifest ships.
    """
    assert ios_pipeline.patient_and_jaw(filename) == expected


@pytest.mark.parametrize(
    "filename,expected",
    [
        # This tool's own output, fed back in -- which the fallback in
        # _discover explicitly allows ("used only when a patient has nothing
        # else"). The suffix must not become part of the patient.
        ("Upper_new_9_Or.vtk", ("new_9", "Upper")),
        ("Lower_new_9_Or.vtk", ("new_9", "Lower")),
        ("Upper_new_9_Upper_O_Pred_Or.mrk.json", ("new_9", "Upper")),
        # No suffix given: nothing to strip, "Or" is just a token.
        ("Upper_Or_9.vtk", ("Or_9", "Upper")),
    ],
)
def test_the_output_suffix_is_decoration_too(filename, expected):
    """`Upper_new_9_Or.vtk` gave `new_9_Or` while its landmark file gave
    `new_9`, so re-running on an output folder split one patient in two and
    failed with "no landmark file for this jaw" -- the same symptom as the
    prediction suffix, one layer out."""
    suffix = "Or" if filename != "Upper_Or_9.vtk" else ""
    assert ios_pipeline.patient_and_jaw(filename, suffix) == expected


def test_a_previous_run_is_discovered_as_one_patient(tmp_path):
    """The round trip: everything a run wrote, read back, is one patient with
    both a surface and a markups file for each jaw."""
    root = tmp_path / "previous"
    root.mkdir()
    for name in ("Upper_new_9_Or.vtk", "Lower_new_9_Or.vtk"):
        _write_mesh(root / name, _UPPER_CENTROIDS, array_name="PredictedID")
    for name in ("Upper_new_9_Upper_O_Pred_Or.mrk.json",
                 "Lower_new_9_Lower_O_Pred_Or.mrk.json"):
        (root / name).write_text('{"markups": [{"controlPoints": []}]}')

    found = ios_pipeline.discover(str(root), "Or")

    assert list(found) == ["new_9"], f"expected one patient, got {sorted(found)}"
    for jaw in ("Upper", "Lower"):
        assert found["new_9"][jaw]["surface"], f"{jaw}: no surface"
        assert found["new_9"][jaw]["markups"], f"{jaw}: no markups"


def test_two_patients_whose_names_are_prefixes_stay_apart():
    """The reason the substring pairing was replaced by an exact stem: upstream
    compared with `in`, so patient `1` matched patient `10`. Ending the
    identifier at a second jaw token must not bring that back."""
    assert ios_pipeline.patient_and_jaw("1_U.vtk") == ("1", "Upper")
    assert ios_pipeline.patient_and_jaw("10_U.vtk") == ("10", "Upper")
    assert ios_pipeline.patient_and_jaw("1_U_Upper_O_Pred.json") == ("1", "Upper")
    assert ios_pipeline.patient_and_jaw("10_U_Upper_O_Pred.json") == ("10", "Upper")


def test_a_gold_bundle_pairs_both_jaws_under_one_case(tmp_path):
    """Both files of the published bundle have to land on the same patient key,
    or load_reference would see two half-cases instead of one reference."""
    root = tmp_path / "gold"
    _write_mesh(root / "Upper_gold.vtk", _UPPER_CENTROIDS, array_name="PredictedID")
    _write_mesh(root / "Lower_gold.vtk", _LOWER_CENTROIDS, array_name="PredictedID")
    found = ios_pipeline.discover(str(root))
    assert list(found) == ["gold"]
    assert sorted(found["gold"]) == ["Lower", "Upper"]

    reference = ios_pipeline.load_reference(str(root), need_surfaces=True)
    assert reference["Upper"]["surface"].endswith("Upper_gold.vtk")
    assert reference["Lower"]["surface"].endswith("Lower_gold.vtk")


def test_patients_are_paired_by_exact_stem(tmp_path):
    """`Files_vtk_json.organise` paired with `vtk_name in json_name`, so patient
    `1` matched patient `10` -- and padded its list with a literal sentinel
    string to keep a loop non-empty."""
    root = tmp_path / "input"
    _ios_case(root, key="P1", with_markups=True, jaws=("U",))
    _ios_case(root, key="P10", with_markups=True, jaws=("U",))
    found = ios_pipeline.discover(str(root))
    assert sorted(found) == ["P1", "P10"]
    assert found["P1"]["Upper"]["markups"].endswith("P1_U_lm.mrk.json")


@pytest.mark.parametrize("array_name", ["Universal_ID", "PredictedID", "UniversalID"])
def test_every_accepted_label_array_name_works(tmp_path, array_name):
    """Slicer's tools have used three names over time and all three are in the
    wild."""
    _write_mesh(tmp_path / "P1_U_Seg.vtk", _UPPER_CENTROIDS, array_name=array_name)
    mesh = surfaces.read_surface(str(tmp_path / "P1_U_Seg.vtk"))
    assert surfaces.label_array_name(mesh) == array_name


def test_fully_automated_ios_end_to_end(tmp_path):
    """Tooth centroids in, both jaws oriented, one transform each."""
    _ios_case(tmp_path / "input")
    run = dispatch.orient(
        input_path=str(tmp_path / "input"),
        reference_path=_ios_reference(tmp_path),
        modality=catalogs.MODALITY_IOS,
        automation=catalogs.AUTOMATION_FULLY,
        ios_teeth=catalogs.DEFAULT_TEETH,
        ios_jaws=["Upper", "Lower"],
        output_dir=str(tmp_path / "out"),
    )
    entry = run.report["cases"]["P1"]
    assert entry["status"] == "ok"
    assert entry["jaws"]["Upper"]["status"] == "ok"
    assert entry["jaws"]["Lower"]["status"] == "ok"
    # Per jaw: the original wrote `<patient>_SegOr.tfm` for both, so the second
    # silently overwrote the first.
    assert "P1_Upper_Or.tfm" in entry["produced"]
    assert "P1_Lower_Or.tfm" in entry["produced"]


def test_semi_automated_ios_end_to_end(tmp_path):
    """Landmarks in, meshes AND landmarks come back oriented."""
    _ios_case(tmp_path / "input", with_markups=True)
    run = dispatch.orient(
        input_path=str(tmp_path / "input"),
        reference_path=_ios_reference(tmp_path, with_surfaces=False, with_markups=True),
        modality=catalogs.MODALITY_IOS,
        automation=catalogs.AUTOMATION_SEMI,
        ios_teeth=catalogs.DEFAULT_TEETH,
        ios_landmark_types=["O"],
        ios_jaws=["Upper", "Lower"],
        output_dir=str(tmp_path / "out"),
    )
    entry = run.report["cases"]["P1"]
    assert entry["status"] == "ok"
    assert "P1_U_Seg_Or.vtk" in entry["produced"]
    assert "P1_U_lm_Or.mrk.json" in entry["produced"]


def test_unsegmented_meshes_are_refused_with_the_array_names(tmp_path):
    """A 422 must say what is missing and where to go instead."""
    mesh = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    for offset in range(4):
        points.InsertNextPoint(offset, offset, 0)
    mesh.SetPoints(points)
    surfaces.write_surface(mesh, str(tmp_path / "input" / "P1_U_Seg.vtk"))

    with pytest.raises(ToolInputError) as raised:
        dispatch.orient(
            input_path=str(tmp_path / "input"),
            reference_path=_ios_reference(tmp_path),
            modality=catalogs.MODALITY_IOS,
            automation=catalogs.AUTOMATION_FULLY,
            ios_teeth=catalogs.DEFAULT_TEETH,
            ios_jaws=["Upper"],
            output_dir=str(tmp_path / "out"),
        )
    for name in markups.LABEL_ARRAY_NAMES:
        assert name in str(raised.value)


def test_a_mixed_batch_processes_what_it_can(tmp_path):
    """'None of your meshes is segmented' is a 422; 'one of your forty was bad'
    is a report entry. Returning nothing for the second is the failure the
    original made routine."""
    root = tmp_path / "input"
    _ios_case(root, key="good", jaws=("U",))
    bad = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    for offset in range(4):
        points.InsertNextPoint(offset, 0, 0)
    bad.SetPoints(points)
    surfaces.write_surface(bad, str(root / "bad_U_Seg.vtk"))

    run = dispatch.orient(
        input_path=str(root),
        reference_path=_ios_reference(tmp_path),
        modality=catalogs.MODALITY_IOS,
        automation=catalogs.AUTOMATION_FULLY,
        ios_teeth=catalogs.DEFAULT_TEETH,
        ios_jaws=["Upper"],
        output_dir=str(tmp_path / "out"),
    )
    assert run.report["cases"]["good"]["status"] == "ok"
    assert run.report["cases"]["bad"]["status"] == "failed"
    assert "tooth labels" in run.report["cases"]["bad"]["jaws"]["Upper"]["reason"]


def test_occlusion_moves_both_jaws_with_one_transform(tmp_path):
    """The point of the mode: two meshes in occlusion stay in occlusion."""
    _ios_case(tmp_path / "input")
    run = dispatch.orient(
        input_path=str(tmp_path / "input"),
        reference_path=_ios_reference(tmp_path),
        modality=catalogs.MODALITY_IOS,
        automation=catalogs.AUTOMATION_FULLY,
        ios_teeth=catalogs.DEFAULT_TEETH,
        ios_jaws=["Upper", "Lower"],
        ios_occlusion=catalogs.OCCLUSION_UPPER_DRIVES,
        output_dir=str(tmp_path / "out"),
    )
    entry = run.report["cases"]["P1"]
    assert entry["jaws"]["Lower"]["registered_on"] == "the Upper jaw's transform"
    upper = sitk.ReadTransform(os.path.join(run.output_dir, "P1_Upper_Or.tfm"))
    lower = sitk.ReadTransform(os.path.join(run.output_dir, "P1_Lower_Or.tfm"))
    assert np.allclose(upper.GetParameters(), lower.GetParameters())


def test_meshes_are_written_binary(tmp_path):
    """vtkPolyDataWriter defaults to ASCII: bigger, ~130x slower to parse, and
    LESS accurate (it prints about six significant digits)."""
    path = _write_mesh(tmp_path / "P1_U_Seg.vtk", _UPPER_CENTROIDS)
    with open(path, "rb") as handle:
        assert b"BINARY" in handle.read(256)


def test_a_missing_tooth_fails_that_jaw_with_its_name(tmp_path):
    """`SameNumberPoint` silently subsampled the longer point set with the
    global numpy generator and re-keyed both by index, so a patient missing one
    tooth was registered against a random correspondence."""
    mesh = surfaces.read_surface(
        _write_mesh(tmp_path / "P1_U_Seg.vtk", _UPPER_CENTROIDS)
    )
    with pytest.raises(ios_icp.RegistrationError) as raised:
        ios_icp.mean_teeth(mesh, (3, 9, 14, 16), "Universal_ID")
    assert "tooth 16" in str(raised.value)


# ---------------------------------------------------------------------------
# Cross-argument rules
# ---------------------------------------------------------------------------

def test_fewer_than_three_cbct_landmarks_is_refused(tmp_path):
    with pytest.raises(ToolInputError) as raised:
        _run_aso(tmp_path, 
            input=str(tmp_path), modality="CBCT", automation="Semi-Automated",
            reference=str(tmp_path), cbct_landmarks={"Ba": True, "S": True},
        )
    assert "at least 3 landmarks" in str(raised.value)


def test_fully_automated_ios_needs_three_or_four_teeth_per_jaw(tmp_path):
    with pytest.raises(ToolInputError) as raised:
        _run_aso(tmp_path, 
            input=str(tmp_path), modality="IOS", automation="Fully-Automated",
            reference=str(tmp_path),
            ios_teeth={"UR6": True, "UL6": True},
            ios_jaws={"Upper": True, "Lower": False},
        )
    assert "3 or 4 teeth" in str(raised.value)


def test_a_driving_jaw_must_be_selected(tmp_path):
    """The original carried an `occlusion` boolean and a `jaw` string that could
    contradict each other; one argument cannot."""
    with pytest.raises(ToolInputError) as raised:
        _run_aso(tmp_path, 
            input=str(tmp_path), modality="IOS", automation="Fully-Automated",
            reference=str(tmp_path),
            ios_jaws=["Lower"],
            ios_occlusion=catalogs.OCCLUSION_UPPER_DRIVES,
        )
    assert "not selected in 'ios_jaws'" in str(raised.value)


def test_output_suffix_cannot_be_a_path(tmp_path):
    """It becomes part of every output file name."""
    with pytest.raises(ToolInputError):
        _run_aso(tmp_path, 
            input=str(tmp_path), modality="CBCT", automation="Semi-Automated",
            reference=str(tmp_path), output_suffix="../escape",
        )


def test_a_selection_the_reference_cannot_support_is_refused_once(tmp_path):
    """The two published references carry DISJOINT landmark sets: Frankfurt
    Horizontal has Ba/S/N/RPo/LPo/ROr/LOr (the schema's defaults), Occlusal has
    ANS/IF/PNS/UL6O/UR1O/UR6O. Picking the second one and leaving the defaults
    alone would drop every landmark and fail all forty patients separately, for
    one wrong choice made in one place."""
    occlusal = os.path.join(str(tmp_path), "occlusal")
    _write_markups(
        os.path.join(occlusal, "UP01_Or.mrk.json"),
        {name: np.array([float(index), 1.0, 2.0]) for index, name in
         enumerate(("ANS", "IF", "PNS", "UL6O", "UR1O", "UR6O"))},
    )
    _cbct_case(tmp_path / "input")

    with pytest.raises(ToolInputError) as raised:
        dispatch.orient(
            input_path=str(tmp_path / "input"),
            reference_path=occlusal,
            modality=catalogs.MODALITY_CBCT,
            automation=catalogs.AUTOMATION_SEMI,
            cbct_landmarks=list(catalogs.DEFAULT_CBCT_LANDMARKS),
            output_dir=str(tmp_path / "out"),
        )
    message = str(raised.value)
    # It names what the reference actually offers, so the fix is one step away.
    for name in ("ANS", "IF", "PNS", "UL6O", "UR1O", "UR6O"):
        assert name in message


def test_the_matching_selection_runs_against_the_same_reference(tmp_path):
    """The other half of the rule: selecting the reference's own landmarks
    works, so the check cannot be blocking a legitimate request."""
    points = {
        "ANS": np.array([0.0, 30.0, 5.0]),
        "IF": np.array([0.0, 20.0, -5.0]),
        "PNS": np.array([0.0, -10.0, 0.0]),
        "UL6O": np.array([20.0, 5.0, -10.0]),
        "UR1O": np.array([0.0, 28.0, -8.0]),
        "UR6O": np.array([-20.0, 5.0, -10.0]),
    }
    occlusal = os.path.join(str(tmp_path), "occlusal")
    _write_markups(os.path.join(occlusal, "UP01_Or.mrk.json"), points)
    _cbct_case(tmp_path / "input", points=points)

    run = dispatch.orient(
        input_path=str(tmp_path / "input"),
        reference_path=occlusal,
        modality=catalogs.MODALITY_CBCT,
        automation=catalogs.AUTOMATION_SEMI,
        cbct_landmarks=list(points),
        output_dir=str(tmp_path / "out"),
    )
    assert run.report["cases"]["patient1"]["status"] == "ok"
    assert run.report["reference_landmarks"] == sorted(points)


def test_an_empty_input_says_what_was_expected(tmp_path):
    (tmp_path / "input").mkdir()
    with pytest.raises(ToolInputError) as raised:
        dispatch.orient(
            input_path=str(tmp_path / "input"),
            reference_path=_cbct_reference(tmp_path),
            modality=catalogs.MODALITY_CBCT,
            automation=catalogs.AUTOMATION_SEMI,
            output_dir=str(tmp_path / "out"),
        )
    assert ".nii.gz" in str(raised.value)


# ---------------------------------------------------------------------------
# main() / the archive
# ---------------------------------------------------------------------------

def test_main_returns_a_directory_with_the_report_and_no_intermediates(tmp_path):
    """main.py zips whatever comes back and names the archive after it, so the
    working copies must be gone by then."""
    _cbct_case(tmp_path / "input")
    output_dir = _run_aso(tmp_path, 
        input=str(tmp_path / "input"),
        modality="CBCT",
        automation="Semi-Automated",
        reference=_cbct_reference(tmp_path),
        cbct_landmarks=list(_REFERENCE_POINTS),
        output_suffix="Or",
    )
    assert os.path.isfile(os.path.join(output_dir, dispatch.REPORT_NAME))
    # A surviving working directory means a run crashed, so it has to be gone
    # whatever happened -- and it is INSIDE output_dir, which the caller keeps.
    assert not os.path.exists(os.path.join(output_dir, dispatch.WORK_DIRNAME))


def test_an_archive_is_not_unpacked_here(tmp_path):
    """No tool handles `.zip` any more: the server unpacks before `run()` is
    called, with the bomb cap and the single-root strip that used to live in
    `_as_directory`. A tool that still tried would be applying neither."""
    import zipfile

    _cbct_case(tmp_path / "cohort")
    archive = str(tmp_path / "cohort.zip")
    with zipfile.ZipFile(archive, "w") as zf:
        for name in os.listdir(tmp_path / "cohort"):
            zf.write(os.path.join(tmp_path / "cohort", name), os.path.join("cohort", name))

    # Linked into a directory of its own and then found to hold no scan, which
    # is exactly what a stray file of any other kind would do.
    with pytest.raises(ToolInputError, match="No CBCT scan found"):
        dispatch.orient(
            input_path=archive,
            reference_path=_cbct_reference(tmp_path),
            modality=catalogs.MODALITY_CBCT,
            automation=catalogs.AUTOMATION_SEMI,
            cbct_landmarks=list(_REFERENCE_POINTS),
            output_dir=str(tmp_path / "out"),
        )


def test_a_single_uploaded_file_does_not_drag_in_its_neighbours(tmp_path):
    """main.py streams every upload of a request into ONE work directory, so
    treating the file's parent as the input root would make the reference
    archive part of the input."""
    work_dir = tmp_path / "work"
    _write_scan(work_dir / "input.nii.gz")
    (work_dir / "reference.zip").write_bytes(b"not a real archive")

    root = dispatch._as_directory(
        str(work_dir / "input.nii.gz"), str(tmp_path / "scratch" / "input_extracted")
    )
    assert sorted(os.listdir(root)) == ["input.nii.gz"]


def test_a_tools_own_run_report_is_not_read_as_landmarks(tmp_path):
    """`is_markups_file` accepts a bare `.json`, because some markups files are
    written that way -- so the landmark tool's `run_report.json`, which sits
    beside its results, was probed on every fully-automated run and skipped with
    a warning that reads like data loss."""
    results = tmp_path / "ali_out"
    _write_markups(results / "patient1_lm_Pred.mrk.json", _REFERENCE_POINTS)
    (results / "run_report.json").write_text('{"tool": "ALI", "summary": {}}')

    collected = dispatch._collect(str(results))

    assert set(collected) == {"patient1"}
    assert set(collected["patient1"]) == set(_REFERENCE_POINTS)


def test_arguments_naming_hosted_weights_end_in_model_or_reference():
    """Weights and reference bundles must be NAMED, never uploaded.

    Whatever serves this tool decides which `Path` arguments it hosts from the
    argument's NAME: `model`, `*_model` and `*_reference` are picked from the
    data it already holds, and every other path is something the caller may
    send. That is a safety property, not a convenience -- a clinician must not
    be asked to upload model weights from a laptop.

    `landmark_model` was `landmark_models` and missed the rule by one letter,
    which would have put a file picker in front of a 4.7 GB bundle. Hence this
    test rather than a comment.
    """
    import inspect
    import typing

    hints = typing.get_type_hints(run)
    hosted = {"reference", "landmark_model"}
    paths = {
        name for name, parameter in inspect.signature(run).parameters.items()
        if hints.get(name) is Path and name != "output_dir"
    }

    def is_hosted(name):
        return name == "model" or name.endswith(("_model", "_reference")) or name == "reference"

    assert hosted <= paths, "an argument this test names has disappeared: {}".format(
        hosted - paths
    )
    for name in hosted:
        assert is_hosted(name), name
    # And nothing else claims to be hosted by accident.
    assert {name for name in paths if is_hosted(name)} == hosted


# ---------------------------------------------------------------------------
# Progress -- what a client watching a forty-patient run is shown
# ---------------------------------------------------------------------------

def _events(path) -> list:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def test_the_cbct_phases_report_where_they_have_got_to(tmp_path, monkeypatch):
    """One event per patient per phase, and the fraction only ever rises.

    Three loops share the bar -- recentring, then registration -- and each is
    given the slice of the run it occupies. Without that the second phase sends
    the bar back to zero, which reads as a run starting over.
    """
    events_file = tmp_path / "events.jsonl"
    monkeypatch.setenv("SADT_PROGRESS_FILE", str(events_file))
    _cbct_case(tmp_path / "input", key="patient1")
    _cbct_case(tmp_path / "input", key="patient2")

    dispatch.orient(
        input_path=str(tmp_path / "input"),
        reference_path=_cbct_reference(tmp_path),
        modality=catalogs.MODALITY_CBCT,
        automation=catalogs.AUTOMATION_SEMI,
        cbct_landmarks=list(_REFERENCE_POINTS),
        output_dir=str(tmp_path / "out"),
    )

    events = _events(events_file)
    assert [event["message"] for event in events] == [
        "centring scan 1 of 2", "centring scan 2 of 2",
        "orienting patient 1 of 2", "orienting patient 2 of 2",
    ]
    fractions = [event["fraction"] for event in events]
    assert fractions == sorted(fractions)
    # No patient's name, no file name: a progress message is stored on the
    # server and shown, and either one is patient metadata.
    assert not any("patient1" in event["message"] for event in events)


def test_the_ios_batch_reports_one_event_per_patient(tmp_path, monkeypatch):
    events_file = tmp_path / "events.jsonl"
    monkeypatch.setenv("SADT_PROGRESS_FILE", str(events_file))
    _ios_case(tmp_path / "input", key="patient1")

    dispatch.orient(
        input_path=str(tmp_path / "input"),
        reference_path=_ios_reference(tmp_path),
        modality=catalogs.MODALITY_IOS,
        automation=catalogs.AUTOMATION_FULLY,
        output_dir=str(tmp_path / "out"),
    )

    assert [event["message"] for event in _events(events_file)] == [
        "orienting patient 1 of 1",
    ]


def test_the_landmark_waypoint_reaches_the_supervisor(tmp_path):
    """The one thing `FakeSup.messages` exists for, and nothing asserted on it.

    The waypoint is what tells a watcher the run has left ASO and is inside the
    landmark tool -- the longest single step of a fully-automated run, and the
    one a bar frozen at 20% would otherwise be unexplained by.
    """
    root = tmp_path / "input"
    _write_scan(root / "patient1_scan.nii.gz")
    sup = FakeSup({"patient1": _predicted()}, tmp_path)

    _run_aso(
        tmp_path, input=str(root), reference=_cbct_reference(tmp_path),
        modality="CBCT", automation="Fully-Automated",
        landmark_model="Bundle", cbct_landmarks=list(_REFERENCE_POINTS), sup=sup,
    )

    assert (0.2, f"predicting landmarks with {dispatch.LANDMARK_TOOL}") in sup.messages


# ---------------------------------------------------------------------------
# What reaches the log: a position, never a patient
# ---------------------------------------------------------------------------

def test_an_unreadable_markups_file_is_logged_without_its_name(tmp_path, caplog):
    """The rule the progress messages follow, applied to the log.

    A tool's stderr is captured to a file in the job directory, and on a FAILED
    run the server copies its tail into its own persistent log, so a name
    written here outlives the run. The unobvious half is that the exception's
    MESSAGE cannot travel either: `markups.load_landmarks` quotes the file's
    own name in it, so passing `exc` through would put the name straight back.
    """
    bad = tmp_path / "Smith_John_T1_lm.mrk.json"
    bad.write_text(json.dumps({"markups": []}))

    with caplog.at_level(logging.INFO, logger="sadt_aso.cbct.pipeline"):
        assert cbct_pipeline.load_landmarks([str(bad)]) == {}

    messages = [record.getMessage() for record in caplog.records]
    assert messages == ["Skipping markups file 1 of 1: ValueError"], messages


def test_a_landmark_file_the_landmark_tool_wrote_is_skipped_without_its_name(
    tmp_path, caplog
):
    """Same rule on the collection side: these files are named after the
    caller's scans, so their names are the caller's too."""
    (tmp_path / "Smith_John_T1_Pred.mrk.json").write_text(json.dumps({"markups": []}))

    with caplog.at_level(logging.INFO, logger="sadt_aso.dispatch"):
        assert dispatch._collect(str(tmp_path)) == {}

    messages = [record.getMessage() for record in caplog.records]
    assert messages == [
        "Skipping a landmark file the landmark tool wrote: ValueError"
    ], messages


def test_meshes_with_no_jaw_in_the_name_are_counted_not_named(tmp_path, caplog):
    """How many, and what to do about it -- which is the whole diagnosis, the
    reason being the same for every one of them. The JawError quotes the file
    it came from, so the messages themselves are never what is logged."""
    (tmp_path / "Smith_John.vtk").write_bytes(b"")
    (tmp_path / "Jones_Mary.vtk").write_bytes(b"")

    with caplog.at_level(logging.INFO, logger="sadt_aso.ios.pipeline"):
        assert ios_pipeline.discover(str(tmp_path), "Or") == {}

    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 1, messages
    assert messages[0].startswith("2 file(s) skipped: nothing in the name says")
    assert not any("Smith_John" in m or "Jones_Mary" in m for m in messages), messages


def test_a_dicom_tree_is_recognised_without_being_declared():
    """The panel used to ask "Input is DICOM". A clinician cannot tell either:
    DICOM slices routinely carry no extension, so the question was one they had
    to guess at, and guessing wrong produced a run that failed for a reason
    nobody could see."""
    import os
    from sadt_aso.cbct import dicom

    testfiles = "/home/luciacev/code/VISOR-serve/DATA/ASO/testfiles"
    if not os.path.isdir(os.path.join(testfiles, "CBCT_FullyAuto_DCM")):
        import pytest
        pytest.skip("the ASO test files are not staged on this machine")

    assert dicom.holds_a_series(os.path.join(testfiles, "CBCT_FullyAuto_DCM"))
    assert dicom.holds_a_series(os.path.join(testfiles, "CBCT_SemiAuto_DCM"))
    # And the NIfTI forms of the same cohorts are not mistaken for one.
    assert not dicom.holds_a_series(os.path.join(testfiles, "CBCT_FullyAuto"))
    assert not dicom.holds_a_series(os.path.join(testfiles, "CBCT_SemiAuto"))


def test_an_empty_folder_is_not_a_series():
    """GDCM raises on a directory it cannot scan, which is "no series here",
    not a failure of the run."""
    import tempfile
    from sadt_aso.cbct import dicom

    assert not dicom.holds_a_series(tempfile.mkdtemp())


def test_the_reference_is_chosen_by_what_the_selection_needs():
    """No choice here for a clinician to make. The two published CBCT bundles
    carry DISJOINT landmark sets, so the selection already says which target
    frame applies; asking again could only produce a pairing that fails every
    patient separately."""
    import os
    from sadt_aso import catalogs, dispatch

    models = "/home/luciacev/code/VISOR-serve/DATA/ASO/models"
    if not os.path.isdir(os.path.join(models, "IOS_Gold_file")):
        import pytest
        pytest.skip("the ASO reference bundles are not staged on this machine")

    frankfurt = ["Ba", "S", "N", "RPo", "LPo", "ROr", "LOr"]
    occlusal = ["ANS", "IF", "PNS", "UL6O", "UR1O", "UR6O"]

    assert "Frankfurt" in dispatch.choose_reference(models, catalogs.MODALITY_CBCT, frankfurt)
    assert "Occlusal" in dispatch.choose_reference(models, catalogs.MODALITY_CBCT, occlusal)
    assert "IOS" in dispatch.choose_reference(models, catalogs.MODALITY_IOS, [])


def test_a_selection_no_reference_supports_is_refused_by_name():
    """Mixing the two frames names what each one offers, rather than orienting
    onto one of them quietly."""
    import os
    from sadt_aso import catalogs, dispatch
    from sadt_aso.errors import ToolInputError
    import pytest

    models = "/home/luciacev/code/VISOR-serve/DATA/ASO/models"
    if not os.path.isdir(os.path.join(models, "IOS_Gold_file")):
        pytest.skip("the ASO reference bundles are not staged on this machine")

    with pytest.raises(ToolInputError) as caught:
        dispatch.choose_reference(models, catalogs.MODALITY_CBCT, ["Ba", "ANS"])
    assert "Frankfurt" in str(caught.value) and "Occlusal" in str(caught.value)


def test_the_models_folder_is_not_mistaken_for_a_bundle():
    """Asked on the TOP LEVEL only: a bundle keeps its markups or its meshes
    right there, while the models folder keeps only directories. Asked
    recursively the folder would look like a bundle, because a child is one."""
    import os
    from sadt_aso import dispatch

    models = "/home/luciacev/code/VISOR-serve/DATA/ASO/models"
    if not os.path.isdir(os.path.join(models, "IOS_Gold_file")):
        import pytest
        pytest.skip("the ASO reference bundles are not staged on this machine")

    assert not dispatch._is_reference_bundle(models)
    assert dispatch._is_reference_bundle(os.path.join(models, "IOS_Gold_file"))
    # ALI's landmark weights live in the same folder and are not a reference.
    assert not dispatch._is_reference_bundle(os.path.join(models, "CBCT_landmark_models"))



def test_a_cohort_with_some_landmarks_and_some_without_does_both(tmp_path):
    """The reason the mode became a derivation. One folder, two kinds of
    patient: the one carrying landmarks registers on them, the one without is
    predicted. Declared as a single mode, half the cohort was always wrong."""
    root = tmp_path / "input"
    _write_scan(root / "has_lm_scan.nii.gz")
    _write_markups(str(root / "has_lm_lm.mrk.json"), _REFERENCE_POINTS)
    _write_scan(root / "no_lm_scan.nii.gz")

    sup = FakeSup({}, tmp_path)
    try:
        _run_aso(tmp_path, input=str(root), reference=_cbct_reference(tmp_path),
                 modality="CBCT", cbct_landmarks=list(_REFERENCE_POINTS), sup=sup)
    except Exception:
        pass
    # The landmark tool was asked about the cohort, not about nobody and not
    # about the patient that already had points.
    assert getattr(sup, "calls", []), "the scan without landmarks was not predicted"


def test_a_patient_with_landmarks_is_never_predicted_for(tmp_path):
    """Re-predicting over points a clinician placed, or a previous run wrote,
    is the surprise this removes."""
    root = tmp_path / "input"
    _write_scan(root / "p1_scan.nii.gz")
    _write_markups(str(root / "p1_lm.mrk.json"), _REFERENCE_POINTS)

    sup = FakeSup({}, tmp_path)
    _run_aso(tmp_path, input=str(root), reference=_cbct_reference(tmp_path),
             modality="CBCT", cbct_landmarks=list(_REFERENCE_POINTS), sup=sup)

    assert not getattr(sup, "calls", []), "landmarks already on disk were predicted again"


def test_written_landmarks_are_drawn_when_slicer_opens_them(tmp_path):
    """The markups DISPLAY node must be on, not just each control point.

    `false` on the display node switches the whole node off: Slicer loads the
    file, builds the node, lists it in the Markups module -- and draws nothing,
    with no error to explain it. Both original CLIs wrote `false`, and it went
    unnoticed inside the old Slicer module because that module loaded the nodes
    itself and could switch them back on. A returned archive has no such panel:
    the file is opened as it was written.

    Pinned here because nothing pinned it, which is exactly why it shipped. ALI
    carries the same rule in its own writer.
    """
    path = markups.write_landmarks(
        {"Ba": (1.0, 2.0, 3.0), "N": (4.0, 5.0, 6.0)},
        str(tmp_path / "P1_lm_Or.mrk.json"),
    )
    node = json.loads(Path(path).read_text())["markups"][0]

    assert node["display"]["visibility"] is True, (
        "the display node is off, so nothing is drawn however visible the points are"
    )
    assert all(point["visibility"] is True for point in node["controlPoints"])


def test_landmarks_rewritten_from_a_callers_file_are_drawn_too(tmp_path):
    """`rewrite_landmarks` keeps the caller's own display settings -- their
    colours, their glyph sizes -- and that is right for every field but one.

    `visibility` on the display node decides whether anything is drawn at all.
    Every file this tool wrote before that was fixed carries `false`, so a
    caller feeding one back in got an invisible result and no way to tell it
    from a run that placed nothing.
    """
    template = tmp_path / "given.mrk.json"
    template.write_text(json.dumps({
        "markups": [{
            "type": "Fiducial",
            "display": {"visibility": False, "color": [1.0, 0.0, 0.0]},
            "controlPoints": [
                {"label": "Ba", "position": [0.0, 0.0, 0.0], "visibility": True},
            ],
        }]
    }))

    path = markups.rewrite_landmarks(
        {"Ba": (1.0, 2.0, 3.0)}, str(template), str(tmp_path / "out.mrk.json"))
    node = json.loads(Path(path).read_text())["markups"][0]

    assert node["display"]["visibility"] is True
    assert node["display"]["color"] == [1.0, 0.0, 0.0], (
        "the caller's own colour is theirs to keep"
    )
