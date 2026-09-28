"""AREG_IOS's unit tests: no GPU, no weights, no network.

Split out of the single `tools/AREG/tests/test_run.py` AREG had before it
became three tools; see AREG_CBCT/tests/test_run.py for why none of them ran.

The patch network is stubbed (it needs a checkpoint and pytorch3d); everything
around it runs for real.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk
import vtk

from sadt_areg_ios import butterfly, icp, mgl, orientation, postprocess, surfaces
from sadt_areg_ios import butterfly as butterfly_module
from sadt_areg_ios import dispatch, landmarks as landmark_files, run, tools
from sadt_areg_ios import pipeline as ios_pipeline
from sadt_areg_common import catalogs, pairing
from sadt_areg_common.errors import SupervisorRequired, ToolInputError


class FakeSup:
    """A supervisor, as a tool sees one. Records what it was asked for.

    `outputs` maps a tool name to a callable taking the parameters it was sent
    and returning the directory it "produced", so a test can plant results
    without any of the real tools existing.
    """

    def __init__(self, tmp_path, outputs=None):
        self.out = Path(tmp_path) / "out"
        self.tmp = Path(tmp_path) / "tmp"
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.outputs = outputs or {}
        self.calls = []
        self.messages = []

    def run(self, tool, **params):
        self.calls.append((tool, params))
        maker = self.outputs.get(tool)
        if maker is None:
            raise AssertionError(f"nothing planted for {tool!r} in this test")
        return Path(maker(params))

    def progress(self, fraction, message):
        self.messages.append((fraction, message))

    def log(self, message):
        self.messages.append((None, message))


def _phantom(size=48, seed=0, spacing=0.8, origin=(-140.0, -90.0, 60.0)):
    """A textured volume with an origin far from zero.

    Far from zero on purpose: that is the condition under which elastix's
    centre of rotation matters, and a phantom centred on the origin would let
    the bug this suite pins pass unnoticed.
    """
    rng = np.random.default_rng(seed)
    volume = rng.random((size,) * 3).astype(np.float32) * 120
    zz, yy, xx = np.meshgrid(*[np.arange(size)] * 3, indexing="ij")
    half = size // 2
    volume += 1400 * (
        ((zz - half) ** 2 / 180 + (yy - half + 2) ** 2 / 140 + (xx - half - 2) ** 2 / 160) < 1
    )
    volume += 900 * (
        ((zz - half + 12) ** 2 / 40 + (yy - half - 10) ** 2 / 35 + (xx - half + 12) ** 2 / 30) < 1
    )
    image = sitk.GetImageFromArray(volume)
    image.SetSpacing((spacing,) * 3)
    image.SetOrigin(origin)
    return image


def _moved(image, rotation=(0.04, -0.025, 0.03), translation=(1.2, -1.6, 0.9)):
    """`image` displaced by a known rigid transform, and that transform."""
    truth = sitk.Euler3DTransform()
    size = np.array(image.GetSize()) / 2.0
    truth.SetCenter(image.TransformContinuousIndexToPhysicalPoint(size.tolist()))
    truth.SetRotation(*rotation)
    truth.SetTranslation(translation)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image)
    resampler.SetTransform(truth.GetInverse())
    resampler.SetInterpolator(sitk.sitkLinear)
    return resampler.Execute(image), truth


def _write(image, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sitk.WriteImage(image, path, useCompression=True)
    return path


def _full_mask(image):
    mask = sitk.GetImageFromArray(np.ones(sitk.GetArrayViewFromImage(image).shape, np.uint8))
    mask.CopyInformation(image)
    return mask


def _grid_mesh(rows=12, columns=12, spacing=1.0):
    """A flat triangulated grid, the smallest thing with a real adjacency."""
    points = vtk.vtkPoints()
    for row in range(rows):
        for column in range(columns):
            points.InsertNextPoint(column * spacing, row * spacing, 0.0)

    triangles = vtk.vtkCellArray()
    for row in range(rows - 1):
        for column in range(columns - 1):
            a = row * columns + column
            for corners in ((a, a + 1, a + columns), (a + 1, a + columns + 1, a + columns)):
                triangle = vtk.vtkTriangle()
                for index, corner in enumerate(corners):
                    triangle.GetPointIds().SetId(index, corner)
                triangles.InsertNextCell(triangle)

    mesh = vtk.vtkPolyData()
    mesh.SetPoints(points)
    mesh.SetPolys(triangles)
    return mesh


class TestCheckpointLookup:
    def test_the_checkpoint_is_found_inside_the_bundle_folder(self, tmp_path):
        """A `server_selectable` name resolves to the hosted ENTRY, and the
        published AREG bundle is a folder -- so what reaches the predictor is a
        directory, not the .ckpt the network loads."""
        from sadt_areg_ios import butterfly

        bundle = tmp_path / "AREG_model"
        (bundle / "nested").mkdir(parents=True)
        (bundle / "nested" / "patch.ckpt").write_text("x")
        assert butterfly.find_checkpoint(str(bundle)).endswith("patch.ckpt")

    def test_a_bundle_with_no_checkpoint_names_the_setup_script(self, tmp_path):
        from sadt_areg_ios import butterfly

        (tmp_path / "empty").mkdir()
        with pytest.raises(ToolInputError, match="setup-models.sh"):
            butterfly.find_checkpoint(str(tmp_path / "empty"))

    def test_several_checkpoints_are_a_422_naming_them(self, tmp_path):
        """Which weights registered a patient must never be a surprise."""
        from sadt_areg_ios import butterfly

        bundle = tmp_path / "two"
        bundle.mkdir()
        (bundle / "a.ckpt").write_text("x")
        (bundle / "b.ckpt").write_text("x")
        with pytest.raises(ToolInputError, match="a.ckpt, b.ckpt"):
            butterfly.find_checkpoint(str(bundle))


class TestJaws:
    def test_a_mesh_that_does_not_say_its_jaw_is_not_a_lower_arch(self):
        """FIX: `Sort` split on `isLowerUpper(file, "Upper")` and treated
        everything else as LOWER, so a maxillary mesh named `patient1.vtk` was
        registered against the mandibular timepoint and returned as a
        success."""
        assert surfaces.jaw_of("patient1.vtk") is None
        assert surfaces.jaw_of("P1_T1_Upper.vtk") == catalogs.JAW_UPPER
        assert surfaces.jaw_of("P1_T1_L.vtk") == catalogs.JAW_LOWER

    def test_the_jaw_token_is_a_whole_token(self):
        """FIX: the vocabulary was matched as substrings and included the bare
        `_U` and `U_`, so a patient identifier like `P_U12` was an upper arch
        whatever the file held."""
        assert surfaces.jaw_of("P_U12_T1.vtk") is None
        assert surfaces.jaw_of("Mdx_T1.vtk") is None


class TestIOSPairing:
    def _cohort(self, tmp_path, names, jaw=catalogs.JAW_UPPER, carry_other=True):
        for timepoint, files in names.items():
            for file_name in files:
                surfaces.write_surface(_grid_mesh(), str(tmp_path / timepoint / file_name))
        return ios_pipeline.pair(
            str(tmp_path / "T1"), str(tmp_path / "T2"), "Reg",
            registered_jaw=jaw, carry_other=carry_other,
        )

    def test_both_arches_of_a_subject_are_paired(self, tmp_path):
        matched = self._cohort(
            tmp_path,
            {
                "T1": ["P1_T1_Upper.vtk", "P1_T1_Lower.vtk"],
                "T2": ["P1_T2_Upper.vtk", "P1_T2_Lower.vtk"],
            },
        )
        assert list(matched.matched) == ["P1"]
        assert sorted(matched.matched["P1"]) == ["Lower", "Upper"]

    def test_a_subject_with_no_upper_arch_is_named_not_dropped(self, tmp_path):
        matched = self._cohort(
            tmp_path, {"T1": ["P1_T1_Lower.vtk"], "T2": ["P1_T2_Lower.vtk"]}
        )
        assert matched.matched == {}
        assert matched.report()["patients_without_a_upper_arch"] == ["P1"]

    def test_a_mesh_naming_no_jaw_is_reported(self, tmp_path):
        matched = self._cohort(
            tmp_path, {"T1": ["mystery.vtk", "P1_T1_U.vtk"], "T2": ["P1_T2_U.vtk"]}
        )
        assert matched.report()["meshes_without_a_jaw_in_their_name"] == ["mystery.vtk"]
        assert list(matched.matched) == ["P1"]

    def test_a_folder_of_mandibles_alone_pairs_for_the_mgl_mode(self, tmp_path):
        """FIX: `Sort` only kept a lower pair when the matching UPPER pair
        existed, since the palatal registration always starts from the maxilla.
        A study that only scanned mandibles paired to nothing -- which is
        exactly what upstream added `SortLower` for."""
        matched = self._cohort(
            tmp_path,
            {"T1": ["P1_T1_Lower.vtk"], "T2": ["P1_T2_Lower.vtk"]},
            jaw=catalogs.JAW_LOWER,
            carry_other=False,
        )
        assert list(matched.matched) == ["P1"]
        assert list(matched.matched["P1"]) == ["Lower"]

    def test_the_mgl_mode_leaves_the_upper_arches_alone(self, tmp_path):
        """Upstream registers the mandibles and does not touch the maxillae:
        the MG model covers the mandible only, and a maxilla is not rigidly
        attached to it."""
        matched = self._cohort(
            tmp_path,
            {
                "T1": ["P1_T1_Upper.vtk", "P1_T1_Lower.vtk"],
                "T2": ["P1_T2_Upper.vtk", "P1_T2_Lower.vtk"],
            },
            jaw=catalogs.JAW_LOWER,
            carry_other=False,
        )
        assert list(matched.matched["P1"]) == ["Lower"]


class TestPostProcess:
    def test_a_small_island_is_absorbed_and_a_large_one_is_kept(self):
        mesh = _grid_mesh(rows=20, columns=20)
        adjacency = postprocess.Adjacency(mesh)
        labels = np.zeros(mesh.GetNumberOfPoints(), np.int64)
        labels[0] = 1                       # a speck
        labels[np.arange(200, 400)] = 1     # a patch of 200

        postprocess.remove_islands(labels, adjacency, 1, min_count=50)
        assert labels[0] == 0
        assert labels[np.arange(200, 400)].sum() == 200

    def test_dilate_then_erode_closes_a_pinhole(self):
        mesh = _grid_mesh(rows=20, columns=20)
        adjacency = postprocess.Adjacency(mesh)
        labels = np.zeros(mesh.GetNumberOfPoints(), np.int64)
        block = [row * 20 + column for row in range(5, 15) for column in range(5, 15)]
        labels[block] = 1
        labels[10 * 20 + 10] = 0            # the pinhole

        postprocess.dilate(labels, adjacency, 1, iterations=2)
        postprocess.erode(labels, adjacency, 1, iterations=2)
        assert labels[10 * 20 + 10] == 1

    def test_a_non_triangle_mesh_is_triangulated_before_its_faces_are_read(self):
        """FIX: `GetPolys().GetData().reshape(-1, 4)` does not fail on a quad
        mesh, it silently reads the wrong point indices."""
        quads = vtk.vtkCellArray()
        quad = vtk.vtkQuad()
        for index in range(4):
            quad.GetPointIds().SetId(index, index)
        quads.InsertNextCell(quad)
        points = vtk.vtkPoints()
        for corner in ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)):
            points.InsertNextPoint(*corner)
        mesh = vtk.vtkPolyData()
        mesh.SetPoints(points)
        mesh.SetPolys(quads)

        assert postprocess.faces_of(postprocess.triangulate(mesh)).shape == (2, 3)


class TestOrientation:
    def test_two_anti_parallel_vectors_do_not_produce_a_nan(self):
        """FIX: `np.arccos` was clamped at +1 only, so a dot product rounding
        just past -1 gave NaN, which then propagated through the rotation
        matrix into every vertex."""
        assert orientation._angle_between(np.array([0.0, 0.0, -1.0]), np.array([0.0, 0.0, 1.0])) \
            == pytest.approx(np.pi)

    def test_two_parallel_vectors_do_not_divide_by_zero(self):
        """FIX: `RotationMatrix` normalised its axis without checking it, so
        the cross product of two already-parallel vectors -- length zero --
        divided by zero and returned NaNs."""
        assert np.allclose(orientation.rotation_matrix(np.zeros(3), 0.0), np.eye(3))

    def test_an_unlabelled_mesh_says_so_rather_than_raising_a_keyerror(self):
        with pytest.raises(orientation.OrientationError, match="no tooth-label array"):
            orientation.tooth_centroids(_grid_mesh())

    def test_a_missing_tooth_is_named(self):
        from vtk.util.numpy_support import numpy_to_vtk

        mesh = _grid_mesh()
        labels = np.full(mesh.GetNumberOfPoints(), 3, np.int64)
        array = numpy_to_vtk(labels, deep=True)
        array.SetName("Universal_ID")
        mesh.GetPointData().AddArray(array)
        mesh.GetPointData().SetActiveScalars("Universal_ID")

        with pytest.raises(orientation.OrientationError, match="8, 9, 14"):
            orientation.tooth_centroids(mesh)


class TestMGLPatch:
    """The lower arch's patch: a band around the mucogingival line, built from
    ALI's 13 MG landmarks. No network, so all of it runs here for real."""

    @staticmethod
    def _labelled_grid(rows=24, columns=24, spacing=1.0, tooth_rows=()):
        """A flat grid, optionally with some rows labelled as lower crowns."""
        from vtk.util.numpy_support import numpy_to_vtk

        mesh = _grid_mesh(rows=rows, columns=columns, spacing=spacing)
        labels = np.zeros(mesh.GetNumberOfPoints(), np.int64)
        for row in tooth_rows:
            labels[row * columns: (row + 1) * columns] = 19  # a lower molar
        array = numpy_to_vtk(np.ascontiguousarray(labels), deep=True)
        array.SetName("Universal_ID")
        mesh.GetPointData().AddArray(array)
        return mesh

    @staticmethod
    def _landmarks_along_row(row, columns, spacing=1.0, count=13):
        """MG landmarks sitting on one row of the grid, in arch order."""
        positions = np.linspace(0, columns - 1, count)
        return {
            name: np.array([position * spacing, row * spacing, 0.0])
            for name, position in zip(mgl.MGL_ORDER, positions)
        }

    def test_the_band_is_grown_along_the_surface_not_through_space(self):
        """The whole reason the walk is geodesic: a straight-line radius of a
        few millimetres reaches the lingual surface wherever the ridge is
        thinner than that, and a buccal patch that leaks across registers on
        the wrong side of the arch."""
        mesh = self._labelled_grid()
        patched, note = mgl.build_patch(
            mesh, self._landmarks_along_row(12, 24), height=2.0
        )
        from vtk.util.numpy_support import vtk_to_numpy

        inside = vtk_to_numpy(patched.GetPointData().GetArray(mgl.MGL_ARRAY_NAME)).astype(bool)
        rows = np.flatnonzero(inside) // 24
        # 2 mm at 1 mm spacing reaches two rows either side, and no further.
        assert rows.min() == 10 and rows.max() == 14
        assert note is None

    def test_a_taller_band_reaches_further(self):
        mesh = self._labelled_grid()
        counts = []
        for height in (1.0, 3.0, 5.0):
            from vtk.util.numpy_support import vtk_to_numpy

            patched, _note = mgl.build_patch(
                self._labelled_grid(), self._landmarks_along_row(12, 24), height=height
            )
            counts.append(
                int(vtk_to_numpy(patched.GetPointData().GetArray(mgl.MGL_ARRAY_NAME)).sum())
            )
        assert counts[0] < counts[1] < counts[2]

    def test_height_zero_registers_on_the_landmarks_alone(self):
        """Upstream's control case: no band and no curve either, so the ICP runs
        on the 13 snapped points only. It is how you measure what the surface
        around the line buys over the points that carry it."""
        from vtk.util.numpy_support import vtk_to_numpy

        patched, _note = mgl.build_patch(
            self._labelled_grid(), self._landmarks_along_row(12, 24), height=0
        )
        inside = vtk_to_numpy(patched.GetPointData().GetArray(mgl.MGL_ARRAY_NAME))
        assert 0 < int(inside.sum()) <= 13

    def test_the_band_is_kept_off_the_crowns(self):
        """The crowns are what MOVES between the two timepoints, so a patch
        overlapping them would register the change instead of measuring it."""
        from vtk.util.numpy_support import vtk_to_numpy

        patched, _note = mgl.build_patch(
            self._labelled_grid(tooth_rows=(13, 14)),
            self._landmarks_along_row(12, 24),
            height=3.0,
        )
        inside = vtk_to_numpy(patched.GetPointData().GetArray(mgl.MGL_ARRAY_NAME)).astype(bool)
        labels = vtk_to_numpy(patched.GetPointData().GetArray("Universal_ID"))
        assert not (inside & (labels == 19)).any()
        assert inside.any()

    def test_an_unsegmented_mesh_keeps_its_whole_band(self):
        """All-False rather than an error: losing the patch entirely would be a
        worse answer than a band that may touch a crown."""
        from vtk.util.numpy_support import vtk_to_numpy

        patched, _note = mgl.build_patch(
            _grid_mesh(rows=24, columns=24), self._landmarks_along_row(12, 24), height=2.0
        )
        assert vtk_to_numpy(patched.GetPointData().GetArray(mgl.MGL_ARRAY_NAME)).sum() > 0

    def test_missing_landmarks_are_tolerated_and_reported(self):
        """ALI does not always place all 13. Three are enough for a curve, and
        which ones were absent reaches the run report rather than a log line."""
        landmarks = self._landmarks_along_row(12, 24)
        for name in ("LL6MG", "LR6MG", "LL5MG"):
            del landmarks[name]
        _patched, note = mgl.build_patch(self._labelled_grid(), landmarks, height=2.0)
        assert note and "LL6MG" in note

    def test_fewer_than_three_landmarks_is_refused(self):
        landmarks = {"LL6MG": np.zeros(3), "LL5MG": np.ones(3)}
        with pytest.raises(mgl.PatchError, match="fewer than 3"):
            mgl.build_patch(self._labelled_grid(), landmarks, height=2.0)

    def test_the_legacy_suffixless_names_are_accepted(self):
        """Predictions made before the MG suffix was added carry LL6, not
        LL6MG."""
        landmarks = {
            name[:-2]: position
            for name, position in self._landmarks_along_row(12, 24).items()
        }
        _patched, _note = mgl.build_patch(self._labelled_grid(), landmarks, height=2.0)

    def test_the_patch_is_not_named_after_the_palate(self):
        """A mandible must never carry an array called "Butterfly": the ICP is
        pointed at an array by name, and two different arches sharing one would
        make a mis-selected mode silently register the wrong thing."""
        assert mgl.MGL_ARRAY_NAME != butterfly_module.PATCH_ARRAY_NAME

    def test_a_patch_that_lands_nowhere_says_so(self):
        """Landmarks from another patient snap onto the mesh regardless -- what
        catches it is the band coming out empty once the crowns are dropped."""
        mesh = self._labelled_grid(tooth_rows=range(24))  # every vertex is a crown
        with pytest.raises(mgl.PatchError, match="came out empty"):
            mgl.build_patch(mesh, self._landmarks_along_row(12, 24), height=2.0)


class TestMGLLandmarkMatching:
    @staticmethod
    def _write(path, names):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        document = {
            "markups": [
                {
                    "controlPoints": [
                        {"label": name, "position": [float(i), 0.0, 0.0]}
                        for i, name in enumerate(names)
                    ]
                }
            ]
        }
        with open(path, "w") as handle:
            json.dump(document, handle)
        return path

    _TOKENS = set(catalogs.JAW_TOKENS)

    def _index(self, root):
        return landmark_files.index(str(root), also_drop=self._TOKENS)

    def test_a_landmark_file_finds_the_scan_it_belongs_to(self, tmp_path):
        self._write(str(tmp_path / "P1_T1_Lower_MG_Pred.json"), mgl.MGL_ORDER)
        found = landmark_files.for_scan(
            self._index(tmp_path), "somewhere/P1_T1_Lower.vtk", also_drop=self._TOKENS
        )
        assert found.endswith("P1_T1_Lower_MG_Pred.json")

    def test_both_timepoints_share_one_folder_and_keep_their_own_file(self, tmp_path):
        """The layout upstream produces -- `lm_T1` and `lm_T2` point at ONE
        directory -- and the bug an end-to-end run caught: keying the landmarks
        by PATIENT collapsed a subject's two files into one ambiguous entry, so
        every MGL run failed with 'two landmark files match'. A patient has one
        key and two scans; the landmarks are per scan."""
        self._write(str(tmp_path / "P1_T1_Lower_MG_Pred.json"), mgl.MGL_ORDER)
        self._write(str(tmp_path / "P1_T2_Lower_MG_Pred.json"), mgl.MGL_ORDER)
        indexed = self._index(tmp_path)

        for timepoint in ("T1", "T2"):
            found = landmark_files.for_scan(
                indexed, f"P1_{timepoint}_Lower.vtk", also_drop=self._TOKENS
            )
            assert os.path.basename(found) == f"P1_{timepoint}_Lower_MG_Pred.json"

    def test_patient_1_does_not_borrow_patient_10s_landmarks(self, tmp_path):
        """FIX: `FindLandmarkFile` fell back to any json whose name merely
        CONTAINS the scan's stem, and took sorted(...)[0] with a warning. With
        P1's own file missing, P1 would take P10's -- and register a mandible
        against another patient's mucogingival line while reporting success."""
        self._write(str(tmp_path / "P10_T1_Lower_MG_Pred.json"), mgl.MGL_ORDER)
        with pytest.raises(landmark_files.LandmarkError, match="no landmark file"):
            landmark_files.for_scan(
                self._index(tmp_path), "P1_T1_Lower.vtk", also_drop=self._TOKENS
            )

    def test_two_files_for_one_scan_is_an_error_not_a_coin_toss(self, tmp_path):
        self._write(str(tmp_path / "a" / "P1_T1_Lower_MG_Pred.json"), mgl.MGL_ORDER)
        self._write(str(tmp_path / "b" / "P1_T1_Lower_MG_Pred.json"), mgl.MGL_ORDER)
        with pytest.raises(landmark_files.LandmarkError, match="landmark files match"):
            landmark_files.for_scan(
                self._index(tmp_path), "P1_T1_Lower.vtk", also_drop=self._TOKENS
            )

    def test_a_landmark_folder_laid_out_differently_still_matches(self, tmp_path):
        """The scans may be in per-site subfolders and the landmarks flat."""
        self._write(str(tmp_path / "P1_T1_Lower_MG_Pred.json"), mgl.MGL_ORDER)
        found = landmark_files.for_scan(
            self._index(tmp_path), "siteA/P1_T1_Lower.vtk", also_drop=self._TOKENS
        )
        assert found.endswith("P1_T1_Lower_MG_Pred.json")

    @pytest.mark.parametrize(
        "landmark_name",
        [
            "H10_T1_L_MG_edited.mrk.json",     # what a hand-corrected file is called
            "H10_T1_L_MG_corrected.json",
            "H10_T1_L_MG_Pred_v2.json",
            "H10_T1_L_Seg_Lower_MG_Pred.json",  # what ALI writes
            "H10_T1_L_MG_JGP.mrk.json",         # someone's initials
        ],
    )
    def test_a_landmark_file_named_after_its_scan_finds_it(self, tmp_path, landmark_name):
        """FIX, found on real data: the rule was a BLACKLIST of words to strip
        (mg, pred, lm), so anything else a person appends -- `_edited`,
        `_corrected`, `_v2`, their initials -- reduced to a different key and
        the file was reported missing while sitting right there.

        A blacklist can never cover what people actually type. The scan's
        tokens being a PREFIX of the landmark file's is the rule people are
        following when they name these.
        """
        self._write(str(tmp_path / landmark_name), mgl.MGL_ORDER)
        found = landmark_files.for_scan(
            self._index(tmp_path), "H10_T1_L.vtk", also_drop=self._TOKENS
        )
        assert os.path.basename(found) == landmark_name

    def test_the_prefix_is_matched_on_whole_tokens(self, tmp_path):
        """Which is what keeps the looser rule safe. `P1` must not match
        `P10_...` -- upstream's substring fallback did exactly that, handing
        one patient another's mucogingival line and reporting success."""
        self._write(str(tmp_path / "P10_T1_L_MG_edited.json"), mgl.MGL_ORDER)
        with pytest.raises(landmark_files.LandmarkError, match="no landmark file"):
            landmark_files.for_scan(
                self._index(tmp_path), "P1_T1_L.vtk", also_drop=self._TOKENS
            )

    def test_two_files_both_extending_the_scan_name_is_an_error(self, tmp_path):
        """Which of `_edited` and `_Pred` is this scan's? Nothing says, so
        neither is picked."""
        self._write(str(tmp_path / "H10_T1_L_MG_edited.json"), mgl.MGL_ORDER)
        self._write(str(tmp_path / "H10_T1_L_MG_Pred_v2.json"), mgl.MGL_ORDER)
        with pytest.raises(landmark_files.LandmarkError, match="could belong to"):
            landmark_files.for_scan(
                self._index(tmp_path), "H10_T1_L.vtk", also_drop=self._TOKENS
            )

    def test_the_timepoints_stay_apart_under_the_prefix_rule(self, tmp_path):
        """`H10_T1` is not a prefix of `H10_T2_...`, so the looser matching
        cannot undo what the scan key exists for."""
        for timepoint in ("T1", "T2"):
            self._write(str(tmp_path / f"H10_{timepoint}_L_MG_edited.json"), mgl.MGL_ORDER)
        indexed = self._index(tmp_path)
        for timepoint in ("T1", "T2"):
            found = landmark_files.for_scan(
                indexed, f"H10_{timepoint}_L.vtk", also_drop=self._TOKENS
            )
            assert os.path.basename(found) == f"H10_{timepoint}_L_MG_edited.json"

    def test_a_file_that_is_not_markups_says_so(self, tmp_path):
        path = str(tmp_path / "P1.json")
        with open(path, "w") as handle:
            json.dump({"something": "else"}, handle)
        with pytest.raises(landmark_files.LandmarkError, match="not a Slicer markups file"):
            landmark_files.load(path)


class TestIOSTransform:
    def test_the_written_transform_resamples_rather_than_moves(self, tmp_path):
        """A .tfm maps a point of the result back to where to sample it, which
        is the inverse of the matrix that moves the mesh. Same convention as
        the CBCT half, and asserted for the same reason: getting it backwards
        is silent."""
        matrix = np.eye(4)
        matrix[:3, 3] = [3.0, -2.0, 1.0]
        path = icp.write_transform(matrix, str(tmp_path / "t.tfm"))
        assert np.allclose(sitk.ReadTransform(path).TransformPoint((0.0, 0.0, 0.0)),
                           (-3.0, 2.0, -1.0))

    def test_a_prior_transform_is_composed_in(self, tmp_path):
        """FIX (of a fix that never ran the way it reads): the automated modes
        orient both timepoints first, so a transform relating AREG's own inputs
        refers to meshes the caller never had. Composing gives one file mapping
        the ORIGINAL mesh to the registered result."""
        prior_move = np.eye(4)
        prior_move[:3, 3] = [10.0, 0.0, 0.0]
        prior_path = icp.write_transform(prior_move, str(tmp_path / "prior.tfm"))

        areg = np.eye(4)
        areg[:3, 3] = [0.0, 5.0, 0.0]
        composed = icp.write_transform(areg, str(tmp_path / "c.tfm"), prior_path=prior_path)

        # original --(+10, 0, 0)--> oriented --(0, +5, 0)--> registered, so the
        # resampling transform of the whole chain is (-10, -5, 0).
        assert np.allclose(
            sitk.ReadTransform(composed).TransformPoint((0.0, 0.0, 0.0)), (-10.0, -5.0, 0.0)
        )

    def test_the_offset_is_read_rather_than_the_translation(self, tmp_path):
        """FIX: `read_matrix` built its 4x4 from `GetTranslation()`, which is
        only the whole offset when the transform's centre is the origin. The
        same defect as elastix's centre of rotation, one file over."""
        transform = sitk.AffineTransform(3)
        transform.SetCenter((50.0, 0.0, 0.0))
        transform.SetMatrix([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0])  # 90 deg about z
        transform.SetTranslation((1.0, 2.0, 3.0))
        path = str(tmp_path / "centred.tfm")
        sitk.WriteTransform(transform, path)

        matrix = icp.matrix_of(sitk.ReadTransform(path))
        # The 4x4 has to reproduce what the transform does to a point; reading
        # GetTranslation() into the last column does not.
        probe = np.array([7.0, -3.0, 2.0])
        assert np.allclose(
            matrix @ np.append(probe, 1.0), np.append(transform.TransformPoint(probe), 1.0)
        )
        assert not np.allclose(matrix[:3, 3], transform.GetTranslation())


class TestPatchCloud:
    def test_a_patch_selecting_nothing_says_so(self):
        from sadt_areg_ios import butterfly
        from vtk.util.numpy_support import numpy_to_vtk

        mesh = _grid_mesh()
        array = numpy_to_vtk(np.zeros(mesh.GetNumberOfPoints(), np.int64), deep=True)
        array.SetName(butterfly.PATCH_ARRAY_NAME)
        mesh.GetPointData().AddArray(array)
        mesh.GetPointData().SetActiveScalars(butterfly.PATCH_ARRAY_NAME)

        with pytest.raises(surfaces.SurfaceError, match="selected no point"):
            butterfly.patch_cloud(mesh)

    def test_the_cloud_holds_exactly_the_patch_points(self):
        from sadt_areg_ios import butterfly
        from vtk.util.numpy_support import numpy_to_vtk

        mesh = _grid_mesh()
        labels = np.zeros(mesh.GetNumberOfPoints(), np.int64)
        labels[:20] = 1
        array = numpy_to_vtk(labels, deep=True)
        array.SetName(butterfly.PATCH_ARRAY_NAME)
        mesh.GetPointData().AddArray(array)
        mesh.GetPointData().SetActiveScalars(butterfly.PATCH_ARRAY_NAME)

        cloud = butterfly.patch_cloud(mesh)
        assert cloud.GetNumberOfPoints() == 20
        assert cloud.GetNumberOfCells() == 20


class _StubPainter:
    """The patch, painted deterministically instead of predicted.

    Everything else on the IOS path -- reading, pairing, the ICP, the transform
    composition, the writing -- runs for real.
    """

    def __init__(self, array_name):
        self.array_name = array_name

    def __call__(self, surface, key):
        from vtk.util.numpy_support import numpy_to_vtk

        points = surfaces.points_of(surface)
        labels = (points[:, 0] < points[:, 0].mean()).astype(np.int64)
        array = numpy_to_vtk(np.ascontiguousarray(labels), deep=True)
        array.SetName(self.array_name)
        surface.GetPointData().AddArray(array)
        surface.GetPointData().SetActiveScalars(self.array_name)
        return surface, None


class TestIOSRun:
    @staticmethod
    def _cohort(tmp_path, jaws=("Upper", "Lower")):
        for timepoint in ("T1", "T2"):
            for jaw in jaws:
                surfaces.write_surface(
                    _grid_mesh(), str(tmp_path / timepoint / f"P1_{timepoint}_{jaw}.vtk")
                )

    def test_the_lower_arch_is_moved_by_the_upper_arch_transform(self, tmp_path):
        """The occlusion the two arches were captured in has to survive the
        registration, which is the whole reason the palate is what is
        registered on."""
        self._cohort(tmp_path)
        matched = ios_pipeline.pair(
            str(tmp_path / "T1"), str(tmp_path / "T2"), "Reg",
            registered_jaw=catalogs.JAW_UPPER, carry_other=True,
        )
        entry = ios_pipeline.register_patient(
            jaws=matched.matched["P1"],
            painter=_StubPainter("Butterfly"),
            registered_jaw=catalogs.JAW_UPPER,
            output_dir=str(tmp_path / "out"),
            relative_key="P1",
            suffix="Reg",
        )
        assert entry["status"] == "ok"
        assert entry["jaws"] == ["Lower", "Upper"]
        assert entry["registered_jaw"] == "Upper"
        # Four meshes plus one transform.
        assert len(entry["outputs"]) == 5
        assert any(name.endswith("_transform.tfm") for name in entry["outputs"])

    def test_the_mgl_run_writes_the_mandibles_only(self, tmp_path):
        """The MG model covers the mandible, so upstream registers the lower
        arches and leaves the maxillae untouched -- two meshes out, not four."""
        self._cohort(tmp_path)
        matched = ios_pipeline.pair(
            str(tmp_path / "T1"), str(tmp_path / "T2"), "Reg",
            registered_jaw=catalogs.JAW_LOWER, carry_other=False,
        )
        entry = ios_pipeline.register_patient(
            jaws=matched.matched["P1"],
            painter=_StubPainter("Bottom_MGL"),
            registered_jaw=catalogs.JAW_LOWER,
            output_dir=str(tmp_path / "out"),
            relative_key="P1",
            suffix="Reg",
        )
        assert entry["registered_on"] == "Bottom_MGL"
        assert entry["jaws"] == ["Lower"]
        assert len(entry["outputs"]) == 3  # two meshes plus the transform
        assert all("Upper" not in name for name in entry["outputs"])


# Moved from AREG_IOSCBCT's suite: this package is the one that makes
# the call, so this is where a change to it should fail.
def test_the_landmark_request_asks_for_mucogingival_alone(tmp_path):
    """Alone, because it is the one network ALI leaves off by default and
    the crown networks would each cost another pass over every mesh."""
    planted = tmp_path / "mg"
    planted.mkdir()
    sup = FakeSup(tmp_path, {"ALI_IOS": lambda params: planted})

    tools.predict_mucogingival(sup, str(tmp_path / "meshes"))

    tool, params = sup.calls[0]
    assert tool == "ALI_IOS"
    assert params["networks"] == ["Mucogingival"]
    # Not named, so ALI picks the bundle matching the input itself.
    assert "model" not in params
    # NOT sent. This test and the caller beside it agreed with each other and
    # not with ALI, which is why the suite stayed green while the call could
    # only ever have failed: ALI fixed its marker at `Pred` and refuses the
    # argument now. A test that pins a contract the callee has dropped is
    # worse than no test -- it reports health.
    assert "prediction_ID" not in params


# Moved from the single AREG suite when AREG became three tools. These drive
# `main()` with t1/t2, which is this tool's signature, not the orchestrator's;
# they were sitting in AREG_IOSCBCT's file only because the three used to share
# one. The three that crossed `modality` with `automation` are not ported: the
# split replaced that argument with the choice of which tool you call.

class TestArgumentRules:
    def _main(self, tmp_path=None, **overrides):
            """Drive `main()` WITH a supervisor unless a test says otherwise.

            Every server run has one, and these tests are about the argument rules
            rather than about reaching the other tools -- without a supervisor the
            structural refusal fires first and hides the rule under test. The
            absent-supervisor case is `TestTheToolsItDrives`.
            """
            arguments = {
                "automation": catalogs.AUTOMATION_SEMI,
                "t1": "/nonexistent/t1",
                "t2": "/nonexistent/t2",
                "sup": FakeSup(tmp_path or "/tmp/areg-rules"),
            }
            arguments.update(overrides)
            return dispatch.main(**arguments)

    def test_the_mucogingival_patch_predicts_its_own_landmarks(self):
            """Sending nothing is the ORDINARY case: the landmarks are predicted by
            the landmark tool. Reaching the file system is the pass condition --
            the rules let the request through."""
            with pytest.raises(Exception) as raised:
                self._main(
                    automation=catalogs.AUTOMATION_SEMI,
                    ios_patch=catalogs.PATCH_MGL,
                )
            assert not isinstance(raised.value, ToolInputError), str(raised.value)

    def test_without_a_way_to_reach_the_landmark_tool_it_asks_for_the_landmarks(self):
            """A deployment may legitimately not carry ALI, and must then say which
            field to fill rather than fail from somewhere inside another tool."""
            with pytest.raises(SupervisorRequired, match="mgl_landmarks"):
                self._main(
                    sup=None,
                    automation=catalogs.AUTOMATION_SEMI,
                    ios_patch=catalogs.PATCH_MGL,
                )

    def test_the_mucogingival_patch_runs_without_a_registration_model(self):
            """Reaching the file system is the pass condition: the rules let it
            through, which is what proves the palatal checkpoint is not required."""
            with pytest.raises(Exception) as raised:
                self._main(
                    automation=catalogs.AUTOMATION_SEMI,
                    ios_patch=catalogs.PATCH_MGL,
                    mgl_landmarks="/nonexistent/landmarks",
                )
            assert "registration_model" not in str(raised.value)
            assert "checkpoint" not in str(raised.value)

    def test_a_negative_patch_height_is_refused(self):
            with pytest.raises(ToolInputError, match="cannot be negative"):
                self._main(
                    automation=catalogs.AUTOMATION_SEMI,
                    ios_patch=catalogs.PATCH_MGL,
                    mgl_landmarks="/nonexistent/landmarks",
                    mgl_patch_height=-1.0,
                )


# ---------------------------------------------------------------------------
# Progress -- the cohort loop, and the channel it does NOT use
# ---------------------------------------------------------------------------

def _events(path) -> list:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def test_the_cohort_loop_reports_one_event_per_subject(tmp_path, monkeypatch):
    """One event per subject, the fraction rising, and no name in any message.

    The subject key is built from the file names the caller sent, so it is
    patient metadata: what travels is the position in the batch.
    """
    events_file = tmp_path / "events.jsonl"
    monkeypatch.setenv("SADT_PROGRESS_FILE", str(events_file))
    for subject in ("P1", "P2"):
        for timepoint in ("T1", "T2"):
            surfaces.write_surface(
                _grid_mesh(),
                str(tmp_path / timepoint / f"{subject}_{timepoint}_Lower.vtk"),
            )
    landmark_root = tmp_path / "mgl"
    landmark_root.mkdir()
    # The band is painted deterministically instead of walked from landmarks;
    # the loop around it, which is what reports, runs for real.
    monkeypatch.setattr(
        ios_pipeline, "MGLPainter", lambda root, height: _StubPainter("Bottom_MGL")
    )

    dispatch.register(
        t1_path=str(tmp_path / "T1"),
        t2_path=str(tmp_path / "T2"),
        automation=catalogs.AUTOMATION_SEMI,
        ios_patch=catalogs.PATCH_MGL,
        mgl_landmarks_path=str(landmark_root),
        output_dir=str(tmp_path / "out"),
    )

    events = _events(events_file)
    assert [event["message"] for event in events] == ["subject 1 of 2", "subject 2 of 2"]
    assert [event["fraction"] for event in events] == [0.0, 0.5]
    assert not any("P1" in event["message"] for event in events)


def test_progress_does_not_travel_through_the_supervisor(tmp_path, monkeypatch):
    """`FakeSup.messages` records everything a tool sends through a supervisor.

    This tool sends none of its progress that way, and that is the design: only
    four of the fifteen tools take a supervisor at all, so the channel has to be
    one the other eleven can use. The environment variable is that channel.
    """
    events_file = tmp_path / "events.jsonl"
    monkeypatch.setenv("SADT_PROGRESS_FILE", str(events_file))
    for timepoint in ("T1", "T2"):
        surfaces.write_surface(
            _grid_mesh(), str(tmp_path / timepoint / f"P1_{timepoint}_Lower.vtk")
        )
    (tmp_path / "mgl").mkdir()
    monkeypatch.setattr(
        ios_pipeline, "MGLPainter", lambda root, height: _StubPainter("Bottom_MGL")
    )
    sup = FakeSup(tmp_path)

    dispatch.register(
        t1_path=str(tmp_path / "T1"),
        t2_path=str(tmp_path / "T2"),
        automation=catalogs.AUTOMATION_SEMI,
        ios_patch=catalogs.PATCH_MGL,
        mgl_landmarks_path=str(tmp_path / "mgl"),
        output_dir=str(tmp_path / "out"),
        sup=sup,
    )

    assert sup.messages == []
    assert [event["message"] for event in _events(events_file)] == ["subject 1 of 1"]


# ---------------------------------------------------------------------------
# The bundles this deployment fills in
# ---------------------------------------------------------------------------

class TestTheModelFieldsThisEngineFillsItself:
    """`models/` holds every AREG bundle together, around sixty checkpoints.

    An unset `registration_model` arrived as that WHOLE folder, and the run died
    on "has to name an entry holding exactly one" -- after it had segmented and
    oriented both timepoints, so the failure cost the whole prediction. Neither
    of these is a clinical choice, so neither is asked for any more.
    """

    def _data_root(self, tmp_path, *names):
        for name in names:
            (tmp_path / "AREG" / "models" / name).mkdir(parents=True)
        return str(tmp_path)

    def test_the_registration_checkpoint_is_found_under_the_data_root(self, tmp_path):
        from sadt_areg_ios import dispatch

        root = self._data_root(tmp_path, "AREG_model")
        assert dispatch._own_bundle(root, dispatch._REGISTRATION_BUNDLE) == str(
            tmp_path / "AREG" / "models" / "AREG_model")

    def test_the_orientation_reference_is_found_the_same_way(self, tmp_path):
        from sadt_areg_ios import dispatch

        root = self._data_root(tmp_path, "IOS_Gold_files")
        assert dispatch._own_bundle(root, dispatch._ORIENTATION_REFERENCE) == str(
            tmp_path / "AREG" / "models" / "IOS_Gold_files")

    def test_a_deployment_publishing_neither_gets_an_empty_string(self, tmp_path):
        """Empty, not a path that does not exist: the checks downstream say what
        is missing, and they can only do that if they are handed nothing."""
        from sadt_areg_ios import dispatch

        root = self._data_root(tmp_path)
        assert dispatch._own_bundle(root, dispatch._REGISTRATION_BUNDLE) == ""
        assert dispatch._own_bundle(None, dispatch._REGISTRATION_BUNDLE) == ""

    def test_what_the_caller_named_wins_over_what_the_deployment_publishes(self, tmp_path):
        """A deployment that ships a bundle must not override a caller who
        deliberately sent another one."""
        from sadt_areg_ios import dispatch

        root = self._data_root(tmp_path, "AREG_model")
        captured = {}

        def spy(*args, **kwargs):
            captured.update(kwargs)
            raise RuntimeError("stop here")

        original = dispatch.register
        dispatch.register = spy
        try:
            dispatch.main(
                automation="Semi-Automated", t1=str(tmp_path), t2=str(tmp_path),
                registration_model="/somewhere/mine", output_dir=str(tmp_path),
                data_root=root,
            )
        except RuntimeError:
            pass
        finally:
            dispatch.register = original
        assert captured.get("registration_model") == "/somewhere/mine"


class TestTheModelFieldsAreNotOffered:
    """Hidden, for the same reason the other two AREG engines hide theirs: a
    field a clinician cannot answer correctly is a field that produces wrong
    runs."""

    @pytest.mark.parametrize(
        "name", ["reference", "registration_model", "mgl_model", "crown_model"])
    def test_each_model_field_is_hidden(self, name):
        from sadt_areg_ios.layout import LAYOUT

        assert LAYOUT[name].get("hidden") is True, name
