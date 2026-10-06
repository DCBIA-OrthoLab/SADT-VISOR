"""End-to-end tests for AMASSS.

No GPU and no real nnUNet models are needed for most of it:
`nnunet_runner.predict_folder` is monkeypatched with a stub that writes
synthetic masks, so everything around the inference itself -- input discovery,
output filtering, model resolution, format conversion, label merging, file
naming, the report -- is exercised for real.

`test_real_models_*` is marked `gpu` and `models`: it runs the shipped bundle on
a real CBCT when `SADT_AMASSS_MODELS` and `SADT_AMASSS_SCAN` point at them, and
is skipped otherwise. CI skips it; run it by hand before opening a PR.
"""

import json
import logging
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from sadt_amasss import run
from sadt_amasss import catalog, nnunet_runner, pipeline, vtk_export
from sadt_amasss.errors import ToolInputError


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _write_scan(path, size=(8, 8, 8), value=100):
    array = np.full(size[::-1], value, dtype=np.int16)
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((0.5, 0.5, 0.5))
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    sitk.WriteImage(image, str(path))
    return str(path)


def _make_model_bundle(root, codes):
    """A bundle laid out exactly as find_model_folder expects."""
    for code in codes:
        plans = root / code / "Dataset001_X" / "nnUNetTrainer__nnUNetPlans__3d_fullres"
        (plans / "fold_0").mkdir(parents=True)
        (plans / "fold_0" / nnunet_runner.CHECKPOINT_NAME).write_bytes(b"fake checkpoint")
    return str(root)


def segmentation_files(report):
    return [path for scan in report["cases"] for path in scan.get("segmentations", [])]


@pytest.fixture
def stub_predictor(monkeypatch):
    """Replace nnUNet inference with a deterministic synthetic mask writer."""

    def fake_predict_folder(model_folder, input_dir, output_dir, device, **kwargs):
        os.makedirs(output_dir, exist_ok=True)
        for name in sorted(os.listdir(input_dir)):
            if not name.endswith("_0000.nii.gz"):
                continue
            case_id = name[: -len("_0000.nii.gz")]
            reference = sitk.ReadImage(os.path.join(input_dir, name))
            array = np.zeros(sitk.GetArrayFromImage(reference).shape, dtype=np.uint8)
            array[2:5, 2:5, 2:5] = 1
            mask = sitk.GetImageFromArray(array)
            mask.CopyInformation(reference)
            sitk.WriteImage(mask, os.path.join(output_dir, f"{case_id}.nii.gz"))

    monkeypatch.setattr(nnunet_runner, "predict_folder", fake_predict_folder)
    monkeypatch.setattr(nnunet_runner, "resolve_device", lambda requested: "cpu")


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------

def test_run_returns_the_output_directory_it_was_given(tmp_path, stub_predictor):
    _write_scan(tmp_path / "input" / "patient01.nii.gz")
    bundle = _make_model_bundle(tmp_path / "bundle", ["MAND", "MAX"])

    output = run(
        scans=tmp_path / "input",
        model=Path(bundle),
        output_dir=tmp_path / "out",
        structures=["MAND", "MAX"],
        merge=["MERGED"],
    )

    assert output == tmp_path / "out"
    assert (output / "AMASSS_report.json").is_file()
    assert (output / "patient01_Pred_SegOut" / "patient01_Pred_MERGED.nii.gz").is_file()
    with open(output / "AMASSS_report.json") as handle:
        assert json.load(handle)["summary"]["processed"] == 1


def test_run_writes_nothing_outside_the_output_directory(tmp_path, stub_predictor):
    _write_scan(tmp_path / "input" / "patient01.nii.gz")
    bundle = _make_model_bundle(tmp_path / "bundle", ["MAND"])
    before = sorted(p for p in tmp_path.rglob("*") if p.is_file())

    run(scans=tmp_path / "input", model=Path(bundle), output_dir=tmp_path / "out")

    after = sorted(
        path
        for path in tmp_path.rglob("*")
        if path.is_file() and not path.is_relative_to(tmp_path / "out")
    )
    assert after == before


def test_the_bulky_intermediates_do_not_survive_the_run(tmp_path, stub_predictor):
    """One predicted volume per scan and per structure, inside what gets shipped."""
    _write_scan(tmp_path / "input" / "patient01.nii.gz")
    bundle = _make_model_bundle(tmp_path / "bundle", ["MAND", "MAX"])

    output = run(scans=tmp_path / "input", model=Path(bundle), output_dir=tmp_path / "out")

    assert not (output / pipeline.WORK_DIRNAME).exists()
    assert sorted(p.name for p in output.iterdir()) == [
        "AMASSS_report.json",
        "patient01_Pred_SegOut",
    ]


def test_run_accepts_a_single_scan_as_readily_as_a_folder(tmp_path, stub_predictor):
    scan = _write_scan(tmp_path / "input" / "patient01.nii.gz")
    bundle = _make_model_bundle(tmp_path / "bundle", ["MAND"])

    output = run(scans=Path(scan), model=Path(bundle), output_dir=tmp_path / "out")

    assert (output / "patient01_Pred_SegOut" / "patient01_Pred_MAND.nii.gz").is_file()


def test_run_accepts_the_old_display_names(tmp_path, stub_predictor):
    """A client still sending 'Mandible' rather than MAND keeps working."""
    _write_scan(tmp_path / "input" / "patient01.nii.gz")
    bundle = _make_model_bundle(tmp_path / "bundle", ["MAND"])

    output = run(
        scans=tmp_path / "input",
        model=Path(bundle),
        output_dir=tmp_path / "out",
        structures=["Mandible"],
        merge=["Separated segmentation files"],
    )

    assert (output / "patient01_Pred_SegOut" / "patient01_Pred_MAND.nii.gz").is_file()


# ---------------------------------------------------------------------------
# split_scan_extension / is_previous_output
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "filename, expected",
    [
        ("scan.nii.gz", ("scan", ".nii.gz")),
        ("scan.nii", ("scan", ".nii")),
        ("scan.nrrd", ("scan", ".nrrd")),
        ("scan.gipl.gz", ("scan", ".gipl.gz")),
        ("a.b.nii.gz", ("a.b", ".nii.gz")),
    ],
)
def test_split_scan_extension(filename, expected):
    assert pipeline.split_scan_extension(filename) == expected


@pytest.mark.parametrize(
    "filename, expected",
    [
        ("patient01.nii.gz", False),
        ("patient01_Pred_MAND.nii.gz", True),
        ("patient01_Pred_MERGED.nii.gz", True),
        ("patient01_Pred_CBMASK.nii.gz", True),
        # A legitimate scan whose name merely contains MASK is NOT an output:
        # the original CLI's blunt `'MASK' not in f` filter dropped it.
        ("MASKED_patient.nii.gz", False),
    ],
)
def test_is_previous_output(filename, expected):
    assert pipeline.is_previous_output(filename, "Pred") is expected


def test_discover_scans_excludes_previous_outputs(tmp_path):
    """Regression: running twice on the same folder must not re-ingest run 1."""
    _write_scan(tmp_path / "input" / "patient01.nii.gz")
    _write_scan(tmp_path / "input" / "patient01_Pred_MAND.nii.gz")
    _write_scan(tmp_path / "input" / "patient01_Pred_MERGED.nii.gz")

    scans = pipeline.discover_scans(str(tmp_path / "input"), "Pred")

    assert [os.path.basename(path) for path in scans] == ["patient01.nii.gz"]


def test_discover_scans_is_recursive(tmp_path):
    """The Slicer UI always counted recursively; the CLI did not. It does now."""
    _write_scan(tmp_path / "input" / "a.nii.gz")
    _write_scan(tmp_path / "input" / "nested" / "deeper" / "b.nii.gz")

    scans = pipeline.discover_scans(str(tmp_path / "input"), "Pred")

    assert sorted(os.path.basename(path) for path in scans) == ["a.nii.gz", "b.nii.gz"]


def test_discover_scans_raises_instead_of_exiting(tmp_path):
    """The original called sys.exit(1), which no caller can act on."""
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        pipeline.discover_scans(str(tmp_path / "empty"), "Pred")


# ---------------------------------------------------------------------------
# model resolution
# ---------------------------------------------------------------------------

def test_resolve_models_reports_missing_structures(tmp_path):
    bundle = _make_model_bundle(tmp_path / "bundle", ["MAND"])

    available, missing = pipeline.resolve_models(bundle, ("MAND", "MAX"))

    assert list(available) == ["MAND"]
    assert missing == ["MAX"]


def test_resolve_models_ignores_model_without_checkpoint(tmp_path):
    """A half-copied bundle degrades to 'unavailable', it does not crash later."""
    plans = tmp_path / "bundle" / "MAND" / "D1" / "t__nnUNetPlans__3d_fullres"
    plans.mkdir(parents=True)
    _make_model_bundle(tmp_path / "bundle", ["MAX"])

    available, missing = pipeline.resolve_models(str(tmp_path / "bundle"), ("MAND", "MAX"))

    assert list(available) == ["MAX"]
    assert missing == ["MAND"]


def test_resolve_models_raises_when_nothing_found(tmp_path):
    (tmp_path / "bundle").mkdir()
    with pytest.raises(nnunet_runner.ModelNotFoundError):
        pipeline.resolve_models(str(tmp_path / "bundle"), ("MAND",))


def test_resolve_models_descends_into_single_wrapper_folder(tmp_path):
    """A copy of 'AMASSS_Models/' rather than of its contents still resolves."""
    _make_model_bundle(tmp_path / "bundle" / "AMASSS_Models", ["MAND"])

    available, _missing = pipeline.resolve_models(str(tmp_path / "bundle"), ("MAND",))

    assert list(available) == ["MAND"]


# ---------------------------------------------------------------------------
# format conversion
# ---------------------------------------------------------------------------

def test_nrrd_input_is_really_converted(tmp_path):
    """The original renamed NRRD to .nii.gz without converting it."""
    source = _write_scan(tmp_path / "scan.nrrd")
    destination = str(tmp_path / "p_000_0000.nii.gz")

    pipeline._convert_to_nifti(source, destination)

    reread = sitk.ReadImage(destination)
    assert reread.GetSize() == (8, 8, 8)
    # Really NIfTI, not an NRRD wearing a .nii.gz name.
    with open(destination, "rb") as handle:
        assert handle.read(2) == b"\x1f\x8b"  # gzip magic
    assert "NRRD" not in sitk.ReadImage(destination).GetMetaDataKeys()


def test_converted_input_keeps_its_voxel_type(tmp_path):
    """Casting to float32 doubled the bytes gzipped per scan and bought nothing."""
    source = _write_scan(tmp_path / "scan.nrrd")
    destination = str(tmp_path / "p_000_0000.nii.gz")

    pipeline._convert_to_nifti(source, destination)

    assert sitk.ReadImage(destination).GetPixelID() == sitk.ReadImage(source).GetPixelID()


# ---------------------------------------------------------------------------
# full pipeline
# ---------------------------------------------------------------------------

def test_segment_batch_merged_and_separate(tmp_path, stub_predictor):
    _write_scan(tmp_path / "input" / "patient01.nii.gz")
    _write_scan(tmp_path / "input" / "patient02.nii.gz")
    bundle = _make_model_bundle(tmp_path / "bundle", ["MAND", "MAX"])

    report = pipeline.segment(
        input_path=str(tmp_path / "input"),
        model_path=bundle,
        output_dir=str(tmp_path / "out"),
        structures=("MAND", "MAX"),
        merge=("MERGED", "SEPARATE"),
        prediction_ID="Pred",
    )

    assert report["summary"] == {"total": 2, "processed": 2, "failed": 0, "partial": 0}
    assert sorted(os.path.basename(p) for p in segmentation_files(report)) == [
        "patient01_Pred_MAND.nii.gz",
        "patient01_Pred_MAX.nii.gz",
        "patient01_Pred_MERGED.nii.gz",
        "patient02_Pred_MAND.nii.gz",
        "patient02_Pred_MAX.nii.gz",
        "patient02_Pred_MERGED.nii.gz",
    ]


def test_an_uncompressed_input_still_produces_compressed_masks(tmp_path, stub_predictor):
    """Regression: the output extension used to mirror the input's, so a scan
    sent as a plain .nii produced one UNCOMPRESSED mask per structure -- 191 MB
    each on a real CBCT, 1.75 GB for a nine-structure run. Label volumes gzip ~100x."""
    _write_scan(tmp_path / "input" / "patient01.nii")
    bundle = _make_model_bundle(tmp_path / "bundle", ["MAND", "MAX"])

    report = pipeline.segment(
        input_path=str(tmp_path / "input"),
        model_path=bundle,
        output_dir=str(tmp_path / "out"),
        structures=("MAND", "MAX"),
        merge=("MERGED", "SEPARATE"),
    )

    assert sorted(os.path.basename(p) for p in segmentation_files(report)) == [
        "patient01_Pred_MAND.nii.gz",
        "patient01_Pred_MAX.nii.gz",
        "patient01_Pred_MERGED.nii.gz",
    ]
    # Written compressed, not merely named .gz.
    for path in segmentation_files(report):
        with open(path, "rb") as handle:
            assert handle.read(2) == b"\x1f\x8b", f"{path} is not gzip data"


def test_a_nrrd_input_keeps_its_format_and_is_compressed(tmp_path, stub_predictor):
    """Compression must not cost the user their chosen format -- and NRRD
    compresses inside the file, so it keeps its own extension (ITK has no
    ".nrrd.gz" writer; asking for one used to fail the whole run)."""
    _write_scan(tmp_path / "input" / "patient01.nrrd")
    bundle = _make_model_bundle(tmp_path / "bundle", ["MAND"])

    report = pipeline.segment(
        input_path=str(tmp_path / "input"),
        model_path=bundle,
        output_dir=str(tmp_path / "out"),
        structures=("MAND",),
        merge=("SEPARATE",),
    )

    produced = segmentation_files(report)[0]
    assert os.path.basename(produced) == "patient01_Pred_MAND.nrrd"
    # Still a readable NRRD carrying the right labels, not just a renamed file.
    assert set(np.unique(sitk.GetArrayFromImage(sitk.ReadImage(produced))).tolist()) <= {0, 1}
    with open(produced, "rb") as handle:
        assert b"encoding: gzip" in handle.read(512), "NRRD written uncompressed"


def test_every_scan_extension_maps_to_a_writable_output_extension(tmp_path):
    """compressed_extension must only ever name a spelling ITK can write --
    the ".nrrd.gz" mapping that shipped first could not be written at all."""
    image = sitk.GetImageFromArray(np.zeros((4, 4, 4), dtype=np.int16))
    for extension in pipeline.SCAN_EXTENSIONS:
        mapped = pipeline.compressed_extension(extension)
        # Writing is the real check: ITK accepts or refuses an extension, and
        # asserting against a hardcoded list would just restate the table.
        sitk.WriteImage(image, str(tmp_path / f"x{mapped}"), useCompression=True)


def test_merged_volume_uses_the_documented_labels(tmp_path, stub_predictor):
    _write_scan(tmp_path / "input" / "patient01.nii.gz")
    bundle = _make_model_bundle(tmp_path / "bundle", ["MAND", "CB"])

    report = pipeline.segment(
        input_path=str(tmp_path / "input"),
        model_path=bundle,
        output_dir=str(tmp_path / "out"),
        structures=("MAND", "CB"),
        merge=("MERGED",),
    )

    merged = next(p for p in segmentation_files(report) if p.endswith("_MERGED.nii.gz"))
    values = set(np.unique(sitk.GetArrayFromImage(sitk.ReadImage(merged))).tolist())
    # The stub gives both structures the same voxels, so MAND (painted last
    # per MERGING_ORDER) wins -- which is exactly the documented behavior.
    assert values == {0, catalog.LABELS["MAND"]}


def test_single_structure_is_written_separately_even_in_merged_mode(tmp_path, stub_predictor):
    _write_scan(tmp_path / "input" / "patient01.nii.gz")
    bundle = _make_model_bundle(tmp_path / "bundle", ["MAND"])

    report = pipeline.segment(
        input_path=str(tmp_path / "input"),
        model_path=bundle,
        output_dir=str(tmp_path / "out"),
        structures=("MAND",),
        merge=("MERGED",),
    )

    assert [os.path.basename(p) for p in segmentation_files(report)] == [
        "patient01_Pred_MAND.nii.gz"
    ]


def test_report_lists_structures_without_a_model(tmp_path, stub_predictor):
    """Regression: a structure with no model used to vanish silently."""
    _write_scan(tmp_path / "input" / "patient01.nii.gz")
    bundle = _make_model_bundle(tmp_path / "bundle", ["MAND"])

    report = pipeline.segment(
        input_path=str(tmp_path / "input"),
        model_path=bundle,
        output_dir=str(tmp_path / "out"),
        structures=("MAND", "MAX", "SKIN"),
    )

    assert report["structures_without_model"] == ["MAX", "SKIN"]
    assert report["predicted_structures"] == ["MAND"]


def test_report_records_the_settings_that_move_the_masks(tmp_path, stub_predictor):
    """GPU resampling and the step size both change the output, so a mask is
    only reproducible next to the values that produced it."""
    _write_scan(tmp_path / "input" / "patient01.nii.gz")
    bundle = _make_model_bundle(tmp_path / "bundle", ["MAND"])

    report = pipeline.segment(
        input_path=str(tmp_path / "input"),
        model_path=bundle,
        output_dir=str(tmp_path / "out"),
        structures=("MAND",),
    )

    # resolve_device is stubbed to "cpu", where the GPU path cannot apply.
    assert report["gpu_resampling"] is False
    assert report["tile_step_size"] == 0.5
    # No surfaces requested -> nothing to report.
    assert report["surface_decimation"] is None


# ---------------------------------------------------------------------------
# argument validation
# ---------------------------------------------------------------------------

def test_segment_rejects_unknown_structure(tmp_path):
    _write_scan(tmp_path / "input" / "patient01.nii.gz")
    with pytest.raises(ToolInputError, match="Unknown structure"):
        pipeline.segment(
            input_path=str(tmp_path / "input"),
            model_path=str(tmp_path),
            output_dir=str(tmp_path / "out"),
            structures=("RC",),
        )


def test_segment_rejects_an_unknown_merge_mode(tmp_path):
    """An unrecognised merge mode must fail before inference, not surface as a
    run that "succeeds" while writing zero segmentation files."""
    _write_scan(tmp_path / "input" / "patient01.nii.gz")
    with pytest.raises(ToolInputError, match="Unknown merge mode"):
        pipeline.segment(
            input_path=str(tmp_path / "input"),
            model_path=str(tmp_path),
            output_dir=str(tmp_path / "out"),
            structures=("MAND",),
            merge=("SEPARATED",),  # plausible typo for SEPARATE
        )


def test_segment_rejects_an_empty_selection(tmp_path):
    """`structures=[]` is a valid list, so the schema cannot catch it."""
    _write_scan(tmp_path / "input" / "patient01.nii.gz")
    with pytest.raises(ToolInputError, match="at least one structure"):
        pipeline.segment(
            input_path=str(tmp_path / "input"),
            model_path=str(tmp_path),
            output_dir=str(tmp_path / "out"),
            structures=(),
        )


@pytest.mark.parametrize(
    "sent, expected",
    [
        # One bare string -- never to be iterated as characters. `merge` shipped
        # as a "choice" once and _codes_from split it into chars, so no mode
        # matched and a full run came back holding nothing but the report.
        ("Separated segmentation files", ("SEPARATE",)),
        ("SEPARATE", ("SEPARATE",)),
        # Display names inside a list are translated like codes are.
        (["One merged segmentation file", "SEPARATE"], ("MERGED", "SEPARATE")),
        # The old base.Selection shape is still accepted.
        ({"One merged segmentation file": True, "Separated segmentation files": True},
         ("MERGED", "SEPARATE")),
    ],
)
def test_merge_modes_accepts_every_legitimate_shape(sent, expected):
    assert catalog.merge_modes(sent) == expected


def test_omitted_optional_merge_falls_back_to_the_default():
    assert catalog.merge_modes(None) == catalog.DEFAULT_MERGE_MODES


def test_structures_with_no_shipped_model_are_not_offered():
    """Offering a structure with no model is worse than not offering it."""
    assert not {"TEETH", "RC", "MCAN"} & set(catalog.STRUCTURE_CODES)
    assert catalog.structure_codes(["Mandible", "Maxilla"]) == ("MAND", "MAX")


# ---------------------------------------------------------------------------
# GPU resampling
# ---------------------------------------------------------------------------

class _FakeConfigurationManager:
    """Stands in for nnUNet's, reproducing the one detail that can bite.

    The real `resampling_fn_data` / `resampling_fn_probabilities` are
    `@property @lru_cache`, so a value read before the swap outlives it unless
    the cache is cleared. Declaring them the same way here means a version of
    `_enable_gpu_resampling` that forgot `cache_clear()` fails the test below
    instead of passing it.
    """

    def __init__(self, configuration):
        self.configuration = configuration

    @property
    @lru_cache(maxsize=1)  # noqa: B019 - deliberately mirrors nnUNet's own shape
    def resampling_fn_data(self):
        return self.configuration["resampling_fn_data"]

    @property
    @lru_cache(maxsize=1)  # noqa: B019
    def resampling_fn_probabilities(self):
        return self.configuration["resampling_fn_probabilities"]


def _fake_predictor(**overrides):
    configuration = {
        "resampling_fn_data": "resample_data_or_seg_to_shape",
        "resampling_fn_data_kwargs": {"is_seg": False, "order": 3},
        "resampling_fn_probabilities": "resample_data_or_seg_to_shape",
        "resampling_fn_probabilities_kwargs": {"is_seg": False, "order": 1},
    }
    configuration.update(overrides)

    class _Predictor:
        pass

    predictor = _Predictor()
    predictor.configuration_manager = _FakeConfigurationManager(configuration)
    return predictor


def test_gpu_resampling_is_skipped_on_cpu():
    """No CUDA, nothing to move: the scipy resamplers must stay untouched."""
    predictor = _fake_predictor()
    assert nnunet_runner._enable_gpu_resampling(predictor, "cpu") is False
    assert (
        predictor.configuration_manager.configuration["resampling_fn_data"]
        == "resample_data_or_seg_to_shape"
    )


def test_gpu_resampling_leaves_a_non_default_resampler_alone():
    """A bundle pinning its own resampler configured its geometry on purpose."""
    predictor = _fake_predictor(resampling_fn_data="no_resampling")
    assert nnunet_runner._enable_gpu_resampling(predictor, "cuda") is False
    assert predictor.configuration_manager.configuration["resampling_fn_data"] == "no_resampling"


def test_gpu_resampling_redirects_both_ends_and_drops_the_memoized_value():
    """Both resamplers move to the GPU, and the cached property does not survive."""
    predictor = _fake_predictor()
    manager = predictor.configuration_manager

    # Read one through the cache first: this is what the swap has to invalidate.
    assert manager.resampling_fn_data == "resample_data_or_seg_to_shape"

    assert nnunet_runner._enable_gpu_resampling(predictor, "cuda") is True

    for key in ("resampling_fn_data", "resampling_fn_probabilities"):
        assert manager.configuration[key] == "resample_torch_fornnunet"
        kwargs = manager.configuration[f"{key}_kwargs"]
        assert kwargs["mode"] == "linear"
        assert str(kwargs["device"]).startswith("cuda")
        # The scipy-only spline order must be gone, not merely overridden.
        assert "order" not in kwargs

    assert manager.resampling_fn_data == "resample_torch_fornnunet"


# ---------------------------------------------------------------------------
# surfaces
# ---------------------------------------------------------------------------

def test_surfaces_are_written_as_binary_vtk(tmp_path):
    """ASCII was the default, and it was both the bulkiest and the lossiest.

    A nine-structure run shipped 1386MB of surfaces against 6.4MB of actual
    segmentation, purely because every coordinate was a decimal string -- and
    printing them to ~6 significant digits moved vertices by up to 5e-05mm.
    """
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    volume = np.zeros((30, 30, 30), dtype=np.uint8)
    volume[10:20, 10:20, 10:20] = 1
    reference = sitk.GetImageFromArray(volume)
    reference.SetSpacing((0.4, 0.4, 0.4))

    output = str(tmp_path / "surface.vtk")
    mesh = vtk_export._mesh_from_mask(volume, reference, 5, (216, 101, 79))
    vtk_export._write(mesh, output)

    with open(output, "rb") as handle:
        assert b"BINARY" in handle.read(200), "legacy VTK header must not say ASCII"

    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(output)
    reader.Update()
    written = reader.GetOutput()

    # Exact round trip, which is what ASCII could not give.
    assert np.array_equal(
        vtk_to_numpy(mesh.GetPoints().GetData()),
        vtk_to_numpy(written.GetPoints().GetData()),
    )
    assert np.array_equal(
        vtk_to_numpy(mesh.GetCellData().GetScalars()),
        vtk_to_numpy(written.GetCellData().GetScalars()),
    )


def test_the_mask_reaches_vtk_without_touching_the_disk(tmp_path):
    """It used to be written out as a `.nrrd` and read back by vtkNrrdReader:
    96 MB out and 96 MB in per mask, eighteen masks per scan on a full run, to
    move a buffer that was already in memory. The scratch file also had a fixed
    name once, and was never removed."""
    volume = np.zeros((20, 20, 20), dtype=np.uint8)
    volume[5:15, 5:15, 5:15] = 1
    reference = sitk.GetImageFromArray(volume)

    mesh = vtk_export._mesh_from_mask(volume, reference, 3, (1, 2, 3))

    assert mesh.GetNumberOfCells() > 0
    assert list(tmp_path.iterdir()) == []


def test_the_contour_is_the_same_surface_the_discrete_filter_gave(tmp_path):
    """`vtkFlyingEdges3D` at 0.5 replaced `vtkDiscreteMarchingCubes`, and the
    brief was that no clinical output moves. On a 0/1 mask the 0.5 isosurface
    crosses every edge at its midpoint, which is where the discrete filter put
    its vertices -- so the two are equal, not merely close. Verified on a real
    mandible of 589 934 triangles; pinned here on something a test can afford.
    """
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    volume = np.zeros((30, 30, 30), dtype=np.uint8)
    zz, yy, xx = np.ogrid[:30, :30, :30]
    volume[((zz - 15) ** 2 + (yy - 15) ** 2 + (xx - 15) ** 2) < 100] = 1
    reference = sitk.GetImageFromArray(volume)
    reference.SetSpacing((0.4, 0.4, 0.4))
    image = vtk_export._image_from_mask(volume, reference)

    discrete = vtk.vtkDiscreteMarchingCubes()
    discrete.SetInputData(image)
    discrete.GenerateValues(1, 1, 1)
    discrete.Update()
    flying = vtk.vtkFlyingEdges3D()
    flying.SetInputData(image)
    flying.SetValue(0, 0.5)
    flying.Update()

    def triangles(poly):
        points = vtk_to_numpy(poly.GetPoints().GetData())
        cells = vtk_to_numpy(poly.GetPolys().GetData()).reshape(-1, 4)[:, 1:]
        # Vertices sorted inside each triangle and triangles sorted between
        # them: the same surface emitted in a different order is the same
        # surface, and that is what has to hold.
        flat = np.sort(points[cells].round(6), axis=1).reshape(-1, 9)
        return flat[np.lexsort(flat.T[::-1])]

    assert discrete.GetOutput().GetNumberOfCells() == flying.GetOutput().GetNumberOfCells()
    assert np.array_equal(triangles(discrete.GetOutput()), triangles(flying.GetOutput()))


def test_decimation_reduces_triangles_and_zero_disables_it(tmp_path):
    """Raw marching cubes on a CBCT grid yields meshes nothing downstream can
    open (1.6M triangles for a cranial base); decimation is what makes the
    .vtk usable, and 0 must still give the untouched mesh back."""
    volume = np.zeros((40, 40, 40), dtype=np.uint8)
    zz, yy, xx = np.ogrid[:40, :40, :40]
    volume[((zz - 20) ** 2 + (yy - 20) ** 2 + (xx - 20) ** 2) < 225] = 1
    reference = sitk.GetImageFromArray(volume)
    reference.SetSpacing((0.4, 0.4, 0.4))

    raw = vtk_export._mesh_from_mask(volume, reference, 5, (1, 2, 3), 0)
    reduced = vtk_export._mesh_from_mask(volume, reference, 5, (1, 2, 3), 90)

    assert raw.GetNumberOfCells() > 0
    assert reduced.GetNumberOfCells() < raw.GetNumberOfCells() / 2

    # The colour array is per-cell, so it has to be built AFTER decimating --
    # sized to the mesh that is actually written, not the one before it.
    assert reduced.GetCellData().GetScalars().GetNumberOfTuples() == reduced.GetNumberOfCells()
    assert raw.GetCellData().GetScalars().GetNumberOfTuples() == raw.GetNumberOfCells()


def test_surfaces_are_produced_alongside_the_segmentations(tmp_path, stub_predictor):
    _write_scan(tmp_path / "input" / "patient01.nii.gz", size=(24, 24, 24))
    bundle = _make_model_bundle(tmp_path / "bundle", ["MAND"])

    report = pipeline.segment(
        input_path=str(tmp_path / "input"),
        model_path=bundle,
        output_dir=str(tmp_path / "out"),
        structures=("MAND",),
        merge=("SEPARATE",),
        generate_surface=True,
    )

    surfaces = [path for scan in report["cases"] for path in scan["surfaces"]]
    assert [os.path.basename(p) for p in surfaces] == ["patient01_Pred_MAND.vtk"]
    assert os.path.getsize(surfaces[0]) > 0
    assert report["surface_decimation"] == 90


# ---------------------------------------------------------------------------
# the real models
# ---------------------------------------------------------------------------

REAL_MODELS = os.environ.get("SADT_AMASSS_MODELS")
REAL_SCAN = os.environ.get("SADT_AMASSS_SCAN")


@pytest.mark.gpu
@pytest.mark.models
@pytest.mark.skipif(
    not (REAL_MODELS and REAL_SCAN),
    reason="set SADT_AMASSS_MODELS and SADT_AMASSS_SCAN (see tests/data/README.md)",
)
def test_real_models_segment_a_real_scan(tmp_path):
    """The shipped bundle on a real CBCT, on the GPU.

    Masks are compared against the pre-port implementation separately (see
    README, "Validated against"); what this asserts is that the real bundle,
    the real scan geometry and the GPU resampling path survive the repackaging.
    """
    output = run(
        scans=Path(REAL_SCAN),
        model=Path(REAL_MODELS),
        output_dir=tmp_path / "out",
        structures=["MAND", "MAX", "CB"],
        merge=["MERGED", "SEPARATE"],
    )

    with open(output / "AMASSS_report.json") as handle:
        report = json.load(handle)

    assert report["summary"] == {"total": 1, "processed": 1, "failed": 0, "partial": 0}
    assert report["predicted_structures"] == ["MAND", "MAX", "CB"]
    assert report["device"].startswith("cuda")
    assert report["gpu_resampling"] is True

    merged = next((output).rglob("*_MERGED.nii.gz"))
    labels = set(np.unique(sitk.GetArrayFromImage(sitk.ReadImage(str(merged)))).tolist())
    assert labels == {0, catalog.LABELS["MAND"], catalog.LABELS["MAX"], catalog.LABELS["CB"]}


# ---------------------------------------------------------------------------
# the schema's published options
# ---------------------------------------------------------------------------

def _choices(argument):
    """The options run()'s annotation publishes for one argument."""
    import typing

    hint = typing.get_type_hints(run)[argument]
    inner = typing.get_args(hint)[0] if typing.get_origin(hint) is list else hint
    return list(typing.get_args(inner))


def test_published_structure_options_match_the_catalog():
    """`Literal` takes literals only, so it cannot be built from the catalog.

    That leaves two declarations of one set -- the exact drift the contract
    exists to prevent -- so this is what keeps them honest. A structure added
    to catalog.STRUCTURE_CODES and not to run() would be unselectable from the
    client; the reverse would publish an option the tool rejects.
    """
    assert _choices("structures") == list(catalog.STRUCTURE_CODES)


def test_published_merge_options_match_the_catalog():
    assert _choices("merge") == list(catalog.MERGE_MODES)


def test_every_published_structure_option_is_accepted_by_the_tool():
    """Published, not enforced: the runner calls run(**params) from JSON."""
    for code in _choices("structures"):
        assert catalog.structure_codes([code]) == (code,)


# ---------------------------------------------------------------------------
# Progress -- three phases, and only the middle one is long
# ---------------------------------------------------------------------------

def test_progress_counts_structures_because_that_is_what_the_run_loops_over(
    tmp_path, stub_predictor, monkeypatch
):
    """Per STRUCTURE, not per scan, and that is the honest unit.

    One nnUNet call covers the whole cohort per structure, so there is no
    per-scan position to report inside it. Interpolating one would put a number
    on the bar that nothing in the run measured. What the two ends of the run
    can count -- reading each scan, writing each scan's outputs -- is counted.
    """
    events_file = tmp_path / "events.jsonl"
    monkeypatch.setenv("SADT_PROGRESS_FILE", str(events_file))
    # The masks counted between completions are pinned by the test below; the
    # watcher is kept out of this one so its exact sequence cannot depend on
    # how fast the stub writes.
    monkeypatch.setattr(pipeline, "WATCH_INTERVAL", 3600.0)
    _write_scan(tmp_path / "input" / "patient01.nii.gz")
    _write_scan(tmp_path / "input" / "patient02.nii.gz")
    bundle = _make_model_bundle(tmp_path / "bundle", ["MAND", "MAX"])

    pipeline.segment(
        input_path=str(tmp_path / "input"),
        model_path=bundle,
        output_dir=str(tmp_path / "out"),
        structures=("MAND", "MAX"),
        merge=("SEPARATE",),
        prediction_ID="Pred",
    )

    events = [json.loads(line) for line in
              Path(events_file).read_text().splitlines()
              if line and json.loads(line).get("kind") != "log"]
    assert [event["message"] for event in events] == [
        "reading scan 1 of 2", "reading scan 2 of 2",
        "predicting 2 structure(s)",
        "structure 1 of 2", "structure 2 of 2",
        "writing scan 1 of 2", "writing scan 2 of 2",
    ]
    fractions = [event["fraction"] for event in events]
    assert fractions == sorted(fractions), "the bar must not restart per phase"
    assert max(fractions) < 1.0, "the run is not finished until the server says so"
    assert not any("patient" in event["message"] for event in events)


def test_the_bar_moves_inside_a_structure_as_its_masks_are_written(
    tmp_path, monkeypatch
):
    """One nnUNet call per structure covers the whole cohort and says nothing
    on the way, but it writes each case's mask as soon as that case is done.
    Counting those masks across every structure is a count of (structure,
    scan) pairs finished -- never an estimate -- so the bar moves through each
    structure instead of sitting still for its whole length, and never runs
    ahead of, or back behind, the completions reported beside it.
    """
    import time

    events_file = tmp_path / "events.jsonl"
    monkeypatch.setenv("SADT_PROGRESS_FILE", str(events_file))
    monkeypatch.setattr(pipeline, "WATCH_INTERVAL", 0.01)
    monkeypatch.setattr(nnunet_runner, "resolve_device", lambda requested: "cpu")
    total_masks = 4
    written = []

    def one_case_at_a_time(model_folder, input_dir, output_dir, device, **kwargs):
        os.makedirs(output_dir, exist_ok=True)
        for name in sorted(os.listdir(input_dir)):
            if not name.endswith("_0000.nii.gz"):
                continue
            case_id = name[: -len("_0000.nii.gz")]
            reference = sitk.ReadImage(os.path.join(input_dir, name))
            array = np.zeros(sitk.GetArrayFromImage(reference).shape, dtype=np.uint8)
            array[2:5, 2:5, 2:5] = 1
            mask = sitk.GetImageFromArray(array)
            mask.CopyInformation(reference)
            sitk.WriteImage(mask, os.path.join(output_dir, f"{case_id}.nii.gz"))
            written.append(case_id)
            if len(written) == total_masks:
                continue
            # Wait for the watcher to have seen this mask, so the sequence
            # below does not depend on how fast this machine is.
            wanted = f"mask {len(written) + 1} of {total_masks}"
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and wanted not in events_file.read_text():
                time.sleep(0.01)

    monkeypatch.setattr(nnunet_runner, "predict_folder", one_case_at_a_time)
    _write_scan(tmp_path / "input" / "Smith_John.nii.gz")
    _write_scan(tmp_path / "input" / "Jones_Mary.nii.gz")
    bundle = _make_model_bundle(tmp_path / "bundle", ["MAND", "MAX"])

    pipeline.segment(
        input_path=str(tmp_path / "input"),
        model_path=bundle,
        output_dir=str(tmp_path / "out"),
        structures=("MAND", "MAX"),
        merge=("SEPARATE",),
        device="cpu",
    )

    records = [json.loads(line) for line in events_file.read_text().splitlines() if line]
    events = [record for record in records if record.get("kind") != "log"]
    assert [event["message"] for event in events] == [
        "reading scan 1 of 2", "reading scan 2 of 2",
        "predicting 2 structure(s)",
        "mask 2 of 4", "mask 3 of 4", "structure 1 of 2",
        "mask 4 of 4", "structure 2 of 2",
        "writing scan 1 of 2", "writing scan 2 of 2",
    ]
    fractions = [event["fraction"] for event in events]
    assert fractions == sorted(fractions), "the two counts must not fight"
    assert all(0.1 <= event["fraction"] < 0.9 for event in events
               if event["message"].startswith(("mask", "structure")))
    assert [record for record in records if record.get("kind") == "log"] == []
    text = events_file.read_text()
    assert "Smith" not in text and "Jones" not in text


def test_a_structure_with_no_model_is_logged_to_the_clinician(
    tmp_path, stub_predictor, monkeypatch
):
    """Asked for and not delivered: the report lists it, and the panel says so
    at the time, by the structure's code -- which names anatomy, not a patient."""
    events_file = tmp_path / "events.jsonl"
    monkeypatch.setenv("SADT_PROGRESS_FILE", str(events_file))
    _write_scan(tmp_path / "input" / "patient01.nii.gz")
    bundle = _make_model_bundle(tmp_path / "bundle", ["MAND"])

    report = pipeline.segment(
        input_path=str(tmp_path / "input"),
        model_path=bundle,
        output_dir=str(tmp_path / "out"),
        structures=("MAND", "MAX"),
        merge=("SEPARATE",),
    )

    assert report["structures_without_model"] == ["MAX"]
    logs = [json.loads(line) for line in events_file.read_text().splitlines()
            if line and json.loads(line).get("kind") == "log"]
    assert [(log["level"], log["audience"], log["message"]) for log in logs] == [
        ("warning", "user", "no model is installed for MAX; it was not segmented"),
    ]


# ---------------------------------------------------------------------------
# What reaches the log: a position, never a patient
# ---------------------------------------------------------------------------

def test_a_scan_that_cannot_be_read_is_logged_by_position(
    tmp_path, stub_predictor, caplog
):
    """The rule the progress messages follow, applied to the log.

    A tool's stderr is captured to a file in the job directory, and on a FAILED
    run the server copies its tail into its own persistent log -- so a scan
    name written on this path outlives the run and its job directory. The
    per-scan report still names the input; that goes back to whoever sent it.

    Asserted on the composed message: SimpleITK's own exception can still name
    the file it could not open, and that is a separate exposure.
    """
    _write_scan(tmp_path / "input" / "Smith_John_T1.nii.gz")
    (tmp_path / "input" / "Jones_Mary_T2.nii.gz").write_bytes(b"not a volume")
    # A previous run's output, in the folder the caller sent back unchanged.
    (tmp_path / "input" / "Smith_John_T1_Pred_MAND.nii.gz").write_bytes(b"")
    bundle = _make_model_bundle(tmp_path / "bundle", ["MAND"])

    with caplog.at_level(logging.INFO, logger="sadt_amasss.pipeline"):
        report = pipeline.segment(
            input_path=str(tmp_path / "input"),
            model_path=bundle,
            output_dir=str(tmp_path / "out"),
            structures=("MAND",),
            merge=("SEPARATE",),
            prediction_ID="Pred",
        )

    messages = [record.getMessage() for record in caplog.records]
    assert "Skipping 1 file(s) that look like a previous AMASSS output" in messages
    assert any(m.startswith("scan ") and " of 2: reading failed (" in m
               for m in messages), messages
    assert not any("Smith_John" in m or "Jones_Mary" in m for m in messages), messages
    assert [scan["input"] for scan in report["cases"] if scan["status"] == "failed"] \
        == ["Jones_Mary_T2.nii.gz"], "the report still names it"


def test_a_missing_mask_is_logged_by_position_and_marks_the_scan_partial(
    tmp_path, monkeypatch, caplog
):
    """The case id `p_001` reaches the operator as `<id>`; the position does
    not. And a scan written without one of its structures says so in the
    report, not only in a log line."""

    def predict_max_for_the_first_case_only(model_folder, input_dir, output_dir,
                                            device, **kwargs):
        os.makedirs(output_dir, exist_ok=True)
        names = sorted(n for n in os.listdir(input_dir) if n.endswith("_0000.nii.gz"))
        if "MAX" in model_folder:
            names = names[:1]
        for name in names:
            reference = sitk.ReadImage(os.path.join(input_dir, name))
            array = np.zeros(sitk.GetArrayFromImage(reference).shape, dtype=np.uint8)
            array[2:5, 2:5, 2:5] = 1
            mask = sitk.GetImageFromArray(array)
            mask.CopyInformation(reference)
            sitk.WriteImage(mask, os.path.join(
                output_dir, name[: -len("_0000.nii.gz")] + ".nii.gz"))

    monkeypatch.setattr(nnunet_runner, "predict_folder", predict_max_for_the_first_case_only)
    monkeypatch.setattr(nnunet_runner, "resolve_device", lambda requested: "cpu")
    _write_scan(tmp_path / "input" / "Adams_Ann_T1.nii.gz")
    _write_scan(tmp_path / "input" / "Zulu_Zoe_T2.nii.gz")
    bundle = _make_model_bundle(tmp_path / "bundle", ["MAND", "MAX"])

    with caplog.at_level(logging.INFO, logger="sadt_amasss"):
        report = pipeline.segment(
            input_path=str(tmp_path / "input"),
            model_path=bundle,
            output_dir=str(tmp_path / "out"),
            structures=("MAND", "MAX"),
            merge=("SEPARATE",),
            prediction_ID="Pred",
        )

    messages = [record.getMessage() for record in caplog.records]
    assert "scan 2 of 2 has no MAX mask; written without it" in messages, messages
    assert not any("p_00" in m for m in messages), messages
    assert not any("Zulu_Zoe" in m or "Adams_Ann" in m for m in messages), messages
    assert report["summary"] == {"total": 2, "processed": 2, "failed": 0, "partial": 1}
    assert report["cases"][1]["partial"] is True
    assert report["cases"][1]["structures_missing"] == ["MAX"]
    assert "partial" not in report["cases"][0]
    summary = [r for r in caplog.records if r.getMessage().startswith("2 of 2 scans segmented")]
    assert [r.levelno for r in summary] == [logging.WARNING], messages


def test_a_scan_with_no_mask_at_all_is_logged_with_position_and_class(
    tmp_path, monkeypatch, caplog
):
    def predict_only_the_first_case(model_folder, input_dir, output_dir, device, **kwargs):
        os.makedirs(output_dir, exist_ok=True)
        name = sorted(n for n in os.listdir(input_dir) if n.endswith("_0000.nii.gz"))[0]
        reference = sitk.ReadImage(os.path.join(input_dir, name))
        array = np.zeros(sitk.GetArrayFromImage(reference).shape, dtype=np.uint8)
        array[2:5, 2:5, 2:5] = 1
        mask = sitk.GetImageFromArray(array)
        mask.CopyInformation(reference)
        sitk.WriteImage(mask, os.path.join(output_dir, name[: -len("_0000.nii.gz")] + ".nii.gz"))

    monkeypatch.setattr(nnunet_runner, "predict_folder", predict_only_the_first_case)
    monkeypatch.setattr(nnunet_runner, "resolve_device", lambda requested: "cpu")
    _write_scan(tmp_path / "input" / "Adams_Ann_T1.nii.gz")
    _write_scan(tmp_path / "input" / "Zulu_Zoe_T2.nii.gz")
    bundle = _make_model_bundle(tmp_path / "bundle", ["MAND"])

    with caplog.at_level(logging.INFO, logger="sadt_amasss"):
        report = pipeline.segment(
            input_path=str(tmp_path / "input"),
            model_path=bundle,
            output_dir=str(tmp_path / "out"),
            structures=("MAND",),
            merge=("SEPARATE",),
            prediction_ID="Pred",
        )

    messages = [record.getMessage() for record in caplog.records]
    assert ("scan 2 of 2: writing outputs failed (RuntimeError: nnUNet wrote no mask "
            "for this scan for any of 1 structure(s))") in messages, messages
    assert not any("Zulu_Zoe" in m or "Adams_Ann" in m for m in messages), messages
    assert "Zulu_Zoe" not in report["cases"][1]["error"]
    summary = [r for r in caplog.records if r.getMessage().startswith("1 of 2 scans segmented, 1 failed")]
    assert [r.levelno for r in summary] == [logging.WARNING], messages


def test_no_scan_segmented_raises_a_runtime_error_with_the_commonest_cause(
    tmp_path, monkeypatch
):
    """Every scan readable, none written: the server's fault, so not an input
    error -- and the message leads with the count and the cause."""

    def predict_nothing(model_folder, input_dir, output_dir, device, **kwargs):
        os.makedirs(output_dir, exist_ok=True)

    monkeypatch.setattr(nnunet_runner, "predict_folder", predict_nothing)
    monkeypatch.setattr(nnunet_runner, "resolve_device", lambda requested: "cpu")
    _write_scan(tmp_path / "input" / "a.nii.gz")
    _write_scan(tmp_path / "input" / "b.nii.gz")
    bundle = _make_model_bundle(tmp_path / "bundle", ["MAND"])

    with pytest.raises(RuntimeError) as caught:
        pipeline.segment(
            input_path=str(tmp_path / "input"), model_path=bundle,
            output_dir=str(tmp_path / "out"), structures=("MAND",), merge=("SEPARATE",),
        )

    assert not isinstance(caught.value, ValueError)
    assert str(caught.value) == (
        "0 of 2 scans segmented; most common failure: RuntimeError: nnUNet wrote no "
        "mask for this scan for any of 1 structure(s) (2 of 2)"
    )


def test_no_structure_predicted_raises_with_the_commonest_cause(
    tmp_path, monkeypatch, caplog
):
    def boom(model_folder, input_dir, output_dir, device, **kwargs):
        raise RuntimeError("nnUNet prediction failed on cpu (MemoryError: out of RAM)")

    monkeypatch.setattr(nnunet_runner, "predict_folder", boom)
    monkeypatch.setattr(nnunet_runner, "resolve_device", lambda requested: "cpu")
    _write_scan(tmp_path / "input" / "a.nii.gz")
    bundle = _make_model_bundle(tmp_path / "bundle", ["MAND", "MAX"])

    with caplog.at_level(logging.INFO, logger="sadt_amasss"), \
            pytest.raises(RuntimeError) as caught:
        pipeline.segment(
            input_path=str(tmp_path / "input"), model_path=bundle,
            output_dir=str(tmp_path / "out"), structures=("MAND", "MAX"),
            merge=("SEPARATE",),
        )

    assert str(caught.value).startswith(
        "0 of 2 structures predicted; most common failure: RuntimeError: nnUNet "
        "prediction failed on cpu (MemoryError: out of RAM) (2 of 2)"
    )
    messages = [record.getMessage() for record in caplog.records]
    assert any(m.startswith("structure 1 of 2 (")
               and "prediction failed (RuntimeError: nnUNet prediction" in m
               for m in messages), messages


def test_no_readable_scan_stays_an_input_error_with_the_commonest_cause(tmp_path):
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "a.nii.gz").write_bytes(b"not a volume")
    (tmp_path / "input" / "b.nii.gz").write_bytes(b"not a volume")
    bundle = _make_model_bundle(tmp_path / "bundle", ["MAND"])

    with pytest.raises(ToolInputError, match=r"^0 of 2 scans could be read .*\(2 of 2\)$"):
        pipeline.segment(
            input_path=str(tmp_path / "input"), model_path=bundle,
            output_dir=str(tmp_path / "out"), structures=("MAND",), merge=("SEPARATE",),
            device="cpu",
        )


def test_input_errors_name_the_argument_and_no_path(tmp_path):
    with pytest.raises(FileNotFoundError) as missing:
        pipeline.discover_scans(str(tmp_path / "nowhere"), "Pred")
    assert str(missing.value) == "'scans' path does not exist"

    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError) as empty:
        pipeline.discover_scans(str(tmp_path / "empty"), "Pred")
    assert "'scans' folder" in str(empty.value)
    assert str(tmp_path) not in str(empty.value)
    # No dotted suffix: the redaction would read ".nii.gz" as a file name.
    assert ".nii" not in str(empty.value) and "nii, nrrd, gipl" in str(empty.value)

    with pytest.raises(FileNotFoundError) as model:
        pipeline.resolve_models(str(tmp_path / "nowhere"), ("MAND",))
    assert str(model.value) == "'model' path is not a directory"


def test_no_model_says_why_in_words_per_structure(tmp_path, caplog):
    bundle = tmp_path / "bundle"
    (bundle / "MAX" / "D1" / "t__nnUNetPlans__3d_fullres").mkdir(parents=True)
    (bundle / "CB").mkdir()

    with caplog.at_level(logging.INFO, logger="sadt_amasss"), \
            pytest.raises(nnunet_runner.ModelNotFoundError) as caught:
        pipeline.resolve_models(str(bundle), ("MAND", "MAX", "CB"))

    message = str(caught.value)
    assert message.startswith(
        "No usable nnUNet model for any requested structure (MAND: folder missing; "
        "MAX: no fold-0 final checkpoint under the 3d_fullres plans folder; "
        "CB: no 3d_fullres plans folder)"
    )
    assert "/" not in message and str(bundle) not in message
    messages = [record.getMessage() for record in caplog.records]
    assert "structure MAND has no usable model (folder missing); it is skipped" in messages


def test_predict_folder_runs_in_process_on_the_cpu_and_names_the_failure(
    tmp_path, monkeypatch
):
    """`predict_from_files` on the CPU fans out to worker processes whose
    death reads only "Background workers died"; the sequential path raises
    the real error in this process."""
    calls = []

    class FakePredictor:
        configuration_manager = None

        def initialize_from_trained_model_folder(self, *args, **kwargs):
            pass

        def predict_from_files(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError("worker-process path used")

        def predict_from_files_sequential(self, *args, **kwargs):
            calls.append(kwargs)
            raise MemoryError("Unable to allocate 4.00 GiB\nfor an array")

    monkeypatch.setattr(nnunet_runner, "_build_predictor", lambda *a, **k: FakePredictor())

    with pytest.raises(RuntimeError) as caught:
        nnunet_runner.predict_folder("model", str(tmp_path), str(tmp_path / "out"), "cpu",
                                     tile_step_size=0.5, gpu_resampling=True)

    assert len(calls) == 1
    assert str(caught.value) == "nnUNet prediction failed on cpu (MemoryError: for an array)"
    assert isinstance(caught.value.__cause__, MemoryError)


def test_the_card_with_scipy_resamplers_keeps_the_workers_and_names_their_death(
    tmp_path, monkeypatch
):
    """There the workers overlap resampling with inference, so they stay; a
    dead one is then reported with its usual cause, since its own error went
    to a stderr nobody reads."""

    class FakePredictor:
        def initialize_from_trained_model_folder(self, *args, **kwargs):
            pass

        def predict_from_files(self, *args, **kwargs):
            raise RuntimeError("Background workers died. Look for the error message further up!")

    monkeypatch.setattr(nnunet_runner, "_build_predictor", lambda *a, **k: FakePredictor())

    with pytest.raises(RuntimeError) as caught:
        nnunet_runner.predict_folder("model", str(tmp_path), str(tmp_path / "out"), "cuda",
                                     tile_step_size=0.5, gpu_resampling=False)

    assert str(caught.value).startswith(
        "nnUNet prediction failed on cuda (RuntimeError: an nnUNet worker process died, "
        "most often for lack of RAM")


def test_a_failed_surface_write_raises_instead_of_passing_silently(tmp_path):
    reference = sitk.Image(10, 10, 10, sitk.sitkInt16)
    mask = np.zeros((10, 10, 10), dtype=np.uint8)
    mask[3:7, 3:7, 3:7] = 1

    with pytest.raises(RuntimeError, match="^writing a surface failed"):
        vtk_export.write_separate_surface(
            mask, reference, "MAND", catalog.LABEL_COLORS, catalog.LABELS,
            smoothing=0, output_path=str(tmp_path / "missing" / "x.vtk"),
        )


def test_an_empty_surface_is_warned_about_with_the_scan_position(tmp_path, caplog):
    reference = sitk.Image(10, 10, 10, sitk.sitkInt16)
    mask = np.zeros((10, 10, 10), dtype=np.uint8)

    with caplog.at_level(logging.INFO, logger="sadt_amasss"):
        vtk_export.write_separate_surfaces(
            {"MAND": mask}, reference, catalog.LABEL_COLORS, catalog.LABELS,
            smoothing=0, decimation=0, output_dir=str(tmp_path),
            name_of=lambda code: f"{code}.vtk", where="scan 3 of 4: ",
        )

    messages = [record.getMessage() for record in caplog.records]
    assert "scan 3 of 4: surface for MAND is empty (0 triangles); written anyway" in messages


# ---------------------------------------------------------------------------
# Structures side by side
#
# One nnUNet call is a third inference and two thirds preprocessing, so a run
# left the card idle 78 % of its own length. Structures are the honest axis for
# that: separate models over the same read-only input folder, each writing to a
# directory of its own.
# ---------------------------------------------------------------------------

class _Supervisor:
    """The one member of the supervisor this tool uses."""

    def __init__(self, grant):
        self.grant = grant
        self.asked = []

    def channels(self, wanted=0):
        self.asked.append(wanted)
        return self.grant


def test_it_asks_for_one_channel_per_structure_it_was_told_to_produce():
    sup = _Supervisor(grant=5)
    assert pipeline._channels_for(sup, 5) == 5
    assert sup.asked == [5]


def test_a_narrower_grant_is_what_it_takes():
    """The whole bargain: a busy server narrows the run instead of refusing
    it."""
    assert pipeline._channels_for(_Supervisor(grant=2), 5) == 2


def test_a_grant_wider_than_the_work_is_capped_at_the_work():
    assert pipeline._channels_for(_Supervisor(grant=8), 3) == 3


def test_without_a_supervisor_it_predicts_one_at_a_time():
    """Which is how this tool is run from a CLI and from its own tests --
    nothing has reserved anything, and opening five nnUNet predictors on an
    unknown card is a way to be killed rather than a way to be fast."""
    assert pipeline._channels_for(None, 5) == 1


def test_a_supervisor_that_raises_costs_the_width_and_not_the_run():
    class _Broken:
        def channels(self, wanted=0):
            raise RuntimeError("no budget service")

    assert pipeline._channels_for(_Broken(), 5) == 1


def _overlap_recording_predictor(monkeypatch, stub_predictor_fn):
    """Wrap the stub so it records how many predictions were ever in flight."""
    import threading

    state = {"running": 0, "peak": 0}
    lock = threading.Lock()
    started = threading.Barrier(2, timeout=5)

    def recording(model_folder, input_dir, output_dir, device, **kwargs):
        with lock:
            state["running"] += 1
            state["peak"] = max(state["peak"], state["running"])
        try:
            # Both structures have to be inside at once for the barrier to
            # clear, so this FAILS rather than passing by luck on a machine
            # that happened to schedule them one after the other.
            started.wait()
        except threading.BrokenBarrierError:
            pass
        stub_predictor_fn(model_folder, input_dir, output_dir, device, **kwargs)
        with lock:
            state["running"] -= 1

    monkeypatch.setattr(nnunet_runner, "predict_folder", recording)
    return state


def test_two_structures_are_predicted_at_once_when_the_machine_pays_for_two(
    tmp_path, stub_predictor, monkeypatch
):
    state = _overlap_recording_predictor(
        monkeypatch, nnunet_runner.predict_folder)
    _write_scan(tmp_path / "input" / "patient01.nii.gz")
    bundle = _make_model_bundle(tmp_path / "bundle", ["MAND", "MAX"])

    pipeline.segment(
        input_path=str(tmp_path / "input"),
        model_path=bundle,
        output_dir=str(tmp_path / "out"),
        structures=("MAND", "MAX"),
        merge=("SEPARATE",),
        device="cpu",
        sup=_Supervisor(grant=2),
    )

    assert state["peak"] == 2, "the structures ran one after the other"


def test_with_no_supervisor_they_do_not_overlap(tmp_path, stub_predictor, monkeypatch):
    """The floor is one, so the CLI and the tests behave exactly as before."""
    import threading

    state = {"running": 0, "peak": 0}
    lock = threading.Lock()
    inner = nnunet_runner.predict_folder

    def recording(model_folder, input_dir, output_dir, device, **kwargs):
        with lock:
            state["running"] += 1
            state["peak"] = max(state["peak"], state["running"])
        inner(model_folder, input_dir, output_dir, device, **kwargs)
        with lock:
            state["running"] -= 1

    monkeypatch.setattr(nnunet_runner, "predict_folder", recording)
    _write_scan(tmp_path / "input" / "patient01.nii.gz")
    bundle = _make_model_bundle(tmp_path / "bundle", ["MAND", "MAX"])

    pipeline.segment(
        input_path=str(tmp_path / "input"),
        model_path=bundle,
        output_dir=str(tmp_path / "out"),
        structures=("MAND", "MAX"),
        merge=("SEPARATE",),
        device="cpu",
    )

    assert state["peak"] == 1


def test_one_structure_failing_still_loses_only_that_one_in_a_pool(
    tmp_path, stub_predictor, monkeypatch
):
    inner = nnunet_runner.predict_folder

    def fail_MAX(model_folder, input_dir, output_dir, device, **kwargs):
        if os.path.basename(os.path.normpath(model_folder)) == "MAX" or "MAX" in model_folder:
            raise RuntimeError("no model")
        inner(model_folder, input_dir, output_dir, device, **kwargs)

    monkeypatch.setattr(nnunet_runner, "predict_folder", fail_MAX)
    _write_scan(tmp_path / "input" / "patient01.nii.gz")
    bundle = _make_model_bundle(tmp_path / "bundle", ["MAND", "MAX"])

    report = pipeline.segment(
        input_path=str(tmp_path / "input"),
        model_path=bundle,
        output_dir=str(tmp_path / "out"),
        structures=("MAND", "MAX"),
        merge=("SEPARATE",),
        device="cpu",
        sup=_Supervisor(grant=2),
    )

    assert any("MAND" in path for path in segmentation_files(report))
    assert not any("MAX" in path for path in segmentation_files(report))


def test_the_bar_counts_finished_structures_not_started_ones(
    tmp_path, stub_predictor, monkeypatch
):
    """With a pool every structure starts at once, so a bar driven off starts
    jumps to full and then sits there for the length of the run."""
    events_file = tmp_path / "events.jsonl"
    monkeypatch.setenv("SADT_PROGRESS_FILE", str(events_file))
    _write_scan(tmp_path / "input" / "patient01.nii.gz")
    bundle = _make_model_bundle(tmp_path / "bundle", ["MAND", "MAX"])

    pipeline.segment(
        input_path=str(tmp_path / "input"),
        model_path=bundle,
        output_dir=str(tmp_path / "out"),
        structures=("MAND", "MAX"),
        merge=("SEPARATE",),
        device="cpu",
        sup=_Supervisor(grant=2),
    )

    messages = [json.loads(line)["message"] for line in
                events_file.read_text().splitlines()]
    structures = [m for m in messages if m.startswith("structure ")]
    assert structures == ["structure 1 of 2", "structure 2 of 2"], structures


def test_cudnn_autotuning_is_turned_back_off_after_the_predictor_is_built():
    """`nnUNetPredictor.__init__` turns it ON, and autotuning picks a
    convolution by TIMING candidates -- so the algorithm, and with it the
    rounding, depends on how busy the card was. Seven runs of one untouched
    scan produced three different masks, 205-265 voxels apart. With structures
    now predicted side by side that stops being a rare coincidence.
    """
    import torch

    torch.backends.cudnn.benchmark = True

    nnunet_runner._build_predictor("cpu", tile_step_size=0.5)

    assert torch.backends.cudnn.benchmark is False


def test_a_caller_named_number_is_a_ceiling_on_the_ask_not_a_floor_over_it():
    """Admission reserved against what it granted, so a tool spreading wider
    than its own share would spend memory nobody set aside."""
    sup = _Supervisor(grant=8)

    assert pipeline._channels_for(sup, 5, declared=2) == 2
    assert sup.asked == [2], "it asked for more than the caller allowed"


def test_without_a_supervisor_a_named_number_is_the_one_that_decides():
    """A CLI, a test: nothing has reserved anything, so the only number left
    is the one the caller typed."""
    assert pipeline._channels_for(None, 5, declared=3) == 3
    assert pipeline._channels_for(None, 2, declared=9) == 2
    assert pipeline._channels_for(None, 5, declared=0) == 1


def test_the_width_is_declared_only_where_it_is_actually_in_force(
    tmp_path, stub_predictor, monkeypatch
):
    """A MEASUREMENT, not the permission. The server keeps the NARROWEST width
    it was told and divides the run's peak by it; a record with no width is
    ignored rather than read as one. So the phase that has a width declares it
    and the serial phases either side stay silent -- and the peak, which is one
    nnUNet model per channel, lands inside the phase that declared it.

    Told nothing, the whole peak is priced as a single channel: every run after
    this one is reserved for five channels' worth and then granted fewer.
    """
    events_file = tmp_path / "events.jsonl"
    monkeypatch.setenv("SADT_PROGRESS_FILE", str(events_file))
    _write_scan(tmp_path / "input" / "patient01.nii.gz")
    bundle = _make_model_bundle(tmp_path / "bundle", ["MAND", "MAX"])

    pipeline.segment(
        input_path=str(tmp_path / "input"),
        model_path=bundle,
        output_dir=str(tmp_path / "out"),
        structures=("MAND", "MAX"),
        merge=("SEPARATE",),
        device="cpu",
        sup=_Supervisor(grant=2),
    )

    records = [json.loads(line) for line in events_file.read_text().splitlines()]
    widths = {r["message"].split()[0]: r.get("width") for r in records}
    assert widths["structure"] == 2, widths
    # The narrowest width the server would keep is the only one declared.
    declared = [r["width"] for r in records if "width" in r]
    assert declared and min(declared) == 2, records
    assert widths.get("reading") is None
    assert widths.get("writing") is None


def test_a_serial_run_declares_one_and_not_nothing(
    tmp_path, stub_predictor, monkeypatch
):
    """Width one IS a measurement -- it is the fixed cost, and the fit needs
    that column as much as it needs a wide one."""
    events_file = tmp_path / "events.jsonl"
    monkeypatch.setenv("SADT_PROGRESS_FILE", str(events_file))
    _write_scan(tmp_path / "input" / "patient01.nii.gz")
    bundle = _make_model_bundle(tmp_path / "bundle", ["MAND", "MAX"])

    pipeline.segment(
        input_path=str(tmp_path / "input"),
        model_path=bundle,
        output_dir=str(tmp_path / "out"),
        structures=("MAND", "MAX"),
        merge=("SEPARATE",),
        device="cpu",
    )

    records = [json.loads(line) for line in events_file.read_text().splitlines()]
    declared = [r["width"] for r in records if "width" in r]
    assert declared and min(declared) == 1, records


def test_meshes_built_side_by_side_come_back_in_the_order_they_were_asked_for():
    """The threads finish in whatever order the machine gives them; the file a
    clinician opens must not. `_meshes_in_parallel` is what the merged surface
    appends from and what names the separate ones, so its order IS the output's
    order."""
    volume = np.zeros((24, 24, 24), dtype=np.uint8)
    reference = sitk.GetImageFromArray(volume)
    jobs = []
    for index, size in enumerate((4, 6, 8, 10)):
        mask = np.zeros((24, 24, 24), dtype=np.uint8)
        mask[2:2 + size, 2:2 + size, 2:2 + size] = 1
        jobs.append((mask, reference, 1, (index, index, index), 0))

    serial = vtk_export._meshes_in_parallel(jobs, workers=1)
    threaded = vtk_export._meshes_in_parallel(jobs, workers=4)

    assert [m.GetNumberOfCells() for m in serial] == [m.GetNumberOfCells() for m in threaded]
    assert [m.GetNumberOfCells() for m in serial] == sorted(
        m.GetNumberOfCells() for m in serial), "the fixture should grow monotonically"


def test_the_separate_surfaces_are_named_after_their_own_structure(tmp_path):
    """Built in a pool, written in the mapping's order -- so a surface cannot
    end up carrying another structure's name because its thread came back
    first."""
    reference = sitk.GetImageFromArray(np.zeros((20, 20, 20), dtype=np.uint8))
    masks = {}
    for index, code in enumerate(("MAND", "MAX", "CB")):
        mask = np.zeros((20, 20, 20), dtype=np.uint8)
        mask[2:2 + 4 + index * 3, 2:2 + 4 + index * 3, 2:2 + 4 + index * 3] = 1
        masks[code] = mask

    from sadt_amasss.catalog import LABEL_COLORS, LABELS

    written = vtk_export.write_separate_surfaces(
        masks=masks, reference=reference, label_colors=LABEL_COLORS, labels=LABELS,
        smoothing=1, decimation=0, output_dir=str(tmp_path),
        name_of=lambda code: f"p_{code}.vtk", workers=3,
    )

    assert [os.path.basename(p) for p in written] == ["p_MAND.vtk", "p_MAX.vtk", "p_CB.vtk"]
    assert all(os.path.isfile(p) for p in written)


# ---------------------------------------------------------------------------
# nnUNet imported once, before the structures' threads
# ---------------------------------------------------------------------------

def test_the_pipeline_preloads_nnunet_before_any_structure_thread_starts(
    tmp_path, stub_predictor, monkeypatch
):
    """The structures' threads must only ever find nnUNet fully imported.

    Each thread used to be the first to import nnunetv2 and the trainer class
    nnUNet looks up by name, and on a five-channel run every structure failed
    on a half-initialised module. The preload has to happen once, on the
    run's own thread, before the pool hands out any work.
    """
    import threading

    events = []
    lock = threading.Lock()
    main_thread = threading.current_thread()

    def record_preload():
        with lock:
            events.append(("preload", threading.current_thread() is main_thread))

    stubbed = nnunet_runner.predict_folder

    def record_predict(model_folder, input_dir, output_dir, device, **kwargs):
        with lock:
            events.append(("predict", threading.current_thread() is main_thread))
        stubbed(model_folder, input_dir, output_dir, device, **kwargs)

    monkeypatch.setattr(nnunet_runner, "preload", record_preload)
    monkeypatch.setattr(nnunet_runner, "predict_folder", record_predict)
    _write_scan(tmp_path / "input" / "patient01.nii.gz")
    bundle = _make_model_bundle(tmp_path / "bundle", ["MAND", "MAX", "CB"])

    report = pipeline.segment(
        input_path=str(tmp_path / "input"),
        model_path=bundle,
        output_dir=str(tmp_path / "out"),
        structures=("MAND", "MAX", "CB"),
        merge=("SEPARATE",),
        num_workers=3,
    )

    assert report["summary"]["processed"] == 1
    assert events[0] == ("preload", True), "preloaded on the run's thread, first"
    assert [kind for kind, _ in events].count("preload") == 1
    assert sorted(events[1:]) == [("predict", False)] * 3, "then predicted in the pool"


def test_concurrent_callers_of_preload_import_once_and_all_wait_for_it(monkeypatch):
    """However many threads arrive at once, the imports run exactly once, and
    none of them returns before they are complete -- a caller returning early
    would reach nnUNet while another thread is still importing it."""
    import threading
    import time

    walked = []
    walk_lock = threading.Lock()

    def slow_import_tree(package_name):
        time.sleep(0.05)
        with walk_lock:
            walked.append(package_name)
        return 1

    monkeypatch.setattr(nnunet_runner, "_preloaded", False)
    monkeypatch.setattr(nnunet_runner, "_import_tree", slow_import_tree)

    callers = 8
    barrier = threading.Barrier(callers)
    complete_on_return = []

    def call():
        barrier.wait()
        nnunet_runner.preload()
        with walk_lock:
            complete_on_return.append(len(walked))

    threads = [threading.Thread(target=call) for _ in range(callers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    expected = list(nnunet_runner._RESOLVED_BY_NAME)
    assert walked == expected, "each package walked once, not once per caller"
    assert complete_on_return == [len(expected)] * callers


def test_after_preload_loading_a_model_has_nothing_left_to_import():
    """Everything nnUNet resolves BY NAME while loading and running a model is
    already in sys.modules once `preload` has run.

    A fresh interpreter, because this process may have imported nnUNet
    already. The lookups are the ones `initialize_from_trained_model_folder`
    and prediction perform -- the trainer named in the checkpoint, the network
    and its building blocks named in the plans, the resamplers, normaliser,
    preprocessor, reader/writer and label manager -- and none of them may
    import a module for the first time, since that is exactly what used to
    happen concurrently in the structures' threads.
    """
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent("""
        import pydoc
        import sys
        from os.path import join

        from sadt_amasss import nnunet_runner

        nnunet_runner.preload()
        before = set(sys.modules)

        import nnunetv2
        from nnunetv2.imageio.reader_writer_registry import (
            recursive_find_reader_writer_by_name)
        from nnunetv2.preprocessing.resampling.utils import (
            recursive_find_resampling_fn_by_name)
        from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
        from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name

        root = nnunetv2.__path__[0]
        found = [
            recursive_find_trainer_class_by_name("nnUNetTrainer"),
            recursive_find_trainer_class_by_name("nnUNetTrainerNoMirroring"),
            recursive_find_resampling_fn_by_name("resample_data_or_seg_to_shape"),
            recursive_find_resampling_fn_by_name("resample_torch_fornnunet"),
            recursive_find_reader_writer_by_name("SimpleITKIO"),
            recursive_find_reader_writer_by_name("NibabelIOWithReorient"),
            recursive_find_python_class(join(root, "preprocessing"), "DefaultPreprocessor",
                                        current_module="nnunetv2.preprocessing"),
            recursive_find_python_class(join(root, "preprocessing", "normalization"),
                                        "CTNormalization",
                                        "nnunetv2.preprocessing.normalization"),
            recursive_find_python_class(join(root, "preprocessing", "normalization"),
                                        "ZScoreNormalization",
                                        "nnunetv2.preprocessing.normalization"),
            recursive_find_python_class(join(root, "utilities", "label_handling"),
                                        "LabelManager",
                                        current_module="nnunetv2.utilities.label_handling"),
            pydoc.locate("dynamic_network_architectures.architectures.unet.PlainConvUNet"),
            pydoc.locate("dynamic_network_architectures.architectures.unet.ResidualEncoderUNet"),
            pydoc.locate("torch.nn.modules.conv.Conv3d"),
            pydoc.locate("torch.nn.modules.instancenorm.InstanceNorm3d"),
            pydoc.locate("torch.nn.LeakyReLU"),
        ]
        assert all(item is not None for item in found), found
        print("\\n".join(sorted(set(sys.modules) - before)))
    """)
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=600,
        env={**os.environ, "PYTHONWARNINGS": "ignore"},
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip() == "", f"imported after preload:\n{result.stdout}"
