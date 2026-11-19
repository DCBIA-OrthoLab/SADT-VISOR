"""Convert on synthetic images built here: nothing committed, nothing clinical."""

import json

import numpy as np
import pytest
import SimpleITK as sitk

from sadt_convert import progress, run
from sadt_convert.errors import ToolInputError


def image(shape=(6, 7, 8), spacing=(0.5, 0.6, 0.7), origin=(1.0, 2.0, 3.0)):
    array = (np.arange(np.prod(shape)) % 200).reshape(shape[::-1]).astype(np.int16)
    result = sitk.GetImageFromArray(array)
    result.SetSpacing(spacing)
    result.SetOrigin(origin)
    return result


def save(img, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(path))
    return path


def dicom_series(folder, slices=5, uid="1.2.826.0.1.3680043.2.1125.1", skip=()):
    folder.mkdir(parents=True, exist_ok=True)
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    base = image(shape=(6, 7, slices))
    for k in range(slices):
        if k in skip:
            continue
        piece = base[:, :, k]
        for tag, value in {
            "0020|000e": uid,
            "0020|0013": str(k + 1),
            "0020|0032": "\\".join(str(v) for v in base.TransformIndexToPhysicalPoint((0, 0, k))),
            "0020|0037": "1\\0\\0\\0\\1\\0",
            "0008|0060": "CT",
        }.items():
            piece.SetMetaData(tag, value)
        writer.SetFileName(str(folder / f"{uid[-1]}_slice{k:03d}.dcm"))
        writer.Execute(piece)
    return base


def report(out):
    return json.loads((out / "run_report.json").read_text())


@pytest.mark.parametrize("name,to", [("a.nrrd", "nii.gz"), ("a.nii.gz", "nrrd"), ("a.nrrd", "gipl.gz")])
def test_file_keeps_voxels_and_geometry(tmp_path, name, to):
    src = image()
    save(src, tmp_path / "in" / name)
    out = run(scans=tmp_path / "in" / name, output_dir=tmp_path / "out", to=to)
    got = sitk.ReadImage(str(out / f"a.{to}"))
    assert np.array_equal(sitk.GetArrayFromImage(got), sitk.GetArrayFromImage(src))
    assert got.GetSpacing() == pytest.approx(src.GetSpacing())
    assert got.GetOrigin() == pytest.approx(src.GetOrigin())


def test_file_already_in_target_format_is_copied_byte_for_byte(tmp_path):
    path = save(image(), tmp_path / "in" / "a.nii.gz")
    out = run(scans=path, output_dir=tmp_path / "out", to="nii.gz")
    assert (out / "a.nii.gz").read_bytes() == path.read_bytes()


def test_folder_keeps_its_structure_and_report(tmp_path):
    save(image(), tmp_path / "in" / "T1" / "case.nrrd")
    save(image(), tmp_path / "in" / "T2" / "case.nrrd")
    out = run(scans=tmp_path / "in", output_dir=tmp_path / "out")
    assert (out / "T1" / "case.nii.gz").exists() and (out / "T2" / "case.nii.gz").exists()
    cases = report(out)["cases"]
    assert set(cases) == {"T1/case", "T2/case"}
    assert cases["T1/case"] == {"input": "T1/case.nrrd", "status": "ok", "produced": ["T1/case.nii.gz"]}


def test_dicom_series_becomes_one_volume(tmp_path):
    base = dicom_series(tmp_path / "in" / "patient")
    out = run(scans=tmp_path / "in", output_dir=tmp_path / "out")
    got = sitk.ReadImage(str(out / "patient.nii.gz"))
    assert got.GetSize() == base.GetSize()
    assert report(out)["cases"]["patient"]["produced"] == ["patient.nii.gz"]


def test_a_bad_scan_does_not_stop_the_batch_and_names_no_file(tmp_path):
    save(image(), tmp_path / "in" / "good.nrrd")
    (tmp_path / "in" / "secret_patient.nii.gz").write_bytes(b"not an image")
    out = run(scans=tmp_path / "in", output_dir=tmp_path / "out")
    cases = report(out)["cases"]
    assert cases["good"]["status"] == "ok"
    assert cases["secret_patient"]["status"].startswith("failed")
    assert "secret_patient" not in cases["secret_patient"]["status"]
    assert not (out / "secret_patient.nii.gz").exists()


def test_same_name_in_two_formats_keeps_one_and_says_so(tmp_path):
    save(image(), tmp_path / "in" / "a.nrrd")
    save(image(), tmp_path / "in" / "a.nii.gz")
    out = run(scans=tmp_path / "in", output_dir=tmp_path / "out")
    statuses = sorted(c["status"] for c in report(out)["cases"].values())
    assert statuses[0].startswith("ok") and statuses[1].startswith("skipped")


def test_nothing_converted_raises(tmp_path):
    (tmp_path / "in").mkdir()
    (tmp_path / "in" / "a.nii.gz").write_bytes(b"junk")
    with pytest.raises(ToolInputError, match="None of the 1"):
        run(scans=tmp_path / "in", output_dir=tmp_path / "out")


def test_empty_folder_missing_path_and_unknown_format(tmp_path):
    (tmp_path / "in").mkdir()
    with pytest.raises(ToolInputError, match="holds no 3D scan"):
        run(scans=tmp_path / "in", output_dir=tmp_path / "out")
    with pytest.raises(ToolInputError, match="does not exist"):
        run(scans=tmp_path / "nope", output_dir=tmp_path / "out")
    save(image(), tmp_path / "in" / "a.nrrd")
    with pytest.raises(ToolInputError, match="Unknown format"):
        run(scans=tmp_path / "in", output_dir=tmp_path / "out", to="png")


def test_progress_counts_positions(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(progress, "report", lambda i, n, what, *a, **k: seen.append((i, n)))
    save(image(), tmp_path / "in" / "a.nrrd")
    save(image(), tmp_path / "in" / "b.nrrd")
    run(scans=tmp_path / "in", output_dir=tmp_path / "out")
    assert seen == [(1, 2), (2, 2)]


def test_two_series_in_one_folder_are_two_cases_not_one_silently_dropped(tmp_path):
    folder = tmp_path / "in" / "study"
    dicom_series(folder, slices=6, uid="1.2.826.0.1.3680043.2.1125.1")
    dicom_series(folder, slices=3, uid="1.2.826.0.1.3680043.2.1125.2")
    out = run(scans=tmp_path / "in", output_dir=tmp_path / "out")
    cases = report(out)["cases"]
    assert sorted(cases) == ["study/series_1", "study/series_2"]
    sizes = sorted(sitk.ReadImage(str(out / c["produced"][0])).GetSize()[2] for c in cases.values())
    assert sizes == [3, 6]


def test_a_series_with_a_missing_slice_is_refused_not_averaged(tmp_path):
    dicom_series(tmp_path / "in" / "a", slices=8, skip=(4,))
    dicom_series(tmp_path / "in" / "b", slices=8, uid="1.2.826.0.1.3680043.2.1125.9")
    out = run(scans=tmp_path / "in", output_dir=tmp_path / "out")
    cases = report(out)["cases"]
    assert "missing or unevenly spaced" in cases["a"]["status"] and cases["b"]["status"] == "ok"


def test_a_format_that_loses_the_direction_is_refused_not_silently_flattened(tmp_path):
    rotated = image()
    rotated.SetDirection((0.9553, -0.2955, 0.0, 0.2955, 0.9553, 0.0, 0.0, 0.0, 1.0))
    save(rotated, tmp_path / "in" / "a.nrrd")
    save(image(), tmp_path / "in" / "b.nrrd")
    out = run(scans=tmp_path / "in", output_dir=tmp_path / "out", to="gipl.gz")
    cases = report(out)["cases"]
    assert "cannot keep" in cases["a"]["status"] and cases["b"]["status"] == "ok"
    assert not (out / "a.gipl.gz").exists()


def test_a_plain_file_named_gz_is_not_copied_as_if_it_were_compressed(tmp_path):
    save(image(), tmp_path / "plain" / "a.nii")
    (tmp_path / "in").mkdir()
    (tmp_path / "in" / "a.nii.gz").write_bytes((tmp_path / "plain" / "a.nii").read_bytes())
    save(image(), tmp_path / "in" / "b.nii.gz")
    out = run(scans=tmp_path / "in", output_dir=tmp_path / "out")
    assert "not compressed" in report(out)["cases"]["a"]["status"]


def test_a_failure_reason_never_carries_a_file_name_even_one_with_spaces(tmp_path):
    from sadt_convert.convert import _reason

    error = OSError("No space left writing '/data/Aaa Bbb Ccc.nii.gz'")
    assert "Aaa" not in _reason(error) and "Bbb" not in _reason(error)


def test_a_gantry_tilted_series_is_refused(tmp_path):
    import math

    folder = tmp_path / "in" / "tilt"
    folder.mkdir(parents=True)
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    tilt = math.radians(20)
    for k in range(8):
        piece = image(shape=(6, 7, 1))[:, :, 0]
        position = (0.0, k * math.sin(tilt), k * math.cos(tilt))
        for tag, value in {
            "0020|000e": "1.2.826.0.1.3680043.2.1125.7", "0020|0013": str(k + 1),
            "0020|0032": "\\".join(str(v) for v in position),
            "0020|0037": "1\\0\\0\\0\\1\\0", "0008|0060": "CT",
        }.items():
            piece.SetMetaData(tag, value)
        writer.SetFileName(str(folder / f"s{k}.dcm"))
        writer.Execute(piece)
    dicom_series(tmp_path / "in" / "ok", slices=6, uid="1.2.826.0.1.3680043.2.1125.8")
    out = run(scans=tmp_path / "in", output_dir=tmp_path / "out")
    cases = report(out)["cases"]
    assert "tilted" in cases["tilt"]["status"] and cases["ok"]["status"] == "ok"


def test_a_large_origin_is_not_refused_for_float32_rounding(tmp_path):
    save(image(origin=(-1234.57, 2345.68, 4096.12)), tmp_path / "in" / "a.nrrd")
    out = run(scans=tmp_path / "in", output_dir=tmp_path / "out")
    assert report(out)["cases"]["a"]["status"] == "ok"


def test_nrrd_gz_is_not_collected_and_a_clash_never_hides_a_case(tmp_path):
    save(image(), tmp_path / "in" / "a.nrrd")
    dicom_series(tmp_path / "in" / "a", slices=5)
    out = run(scans=tmp_path / "in", output_dir=tmp_path / "out")
    cases = report(out)["cases"]
    assert len(cases) == 2 and sum(c["status"] == "ok" for c in cases.values()) == 1


def test_a_failed_scan_leaves_no_empty_folder_behind(tmp_path):
    (tmp_path / "in" / "sub").mkdir(parents=True)
    (tmp_path / "in" / "sub" / "bad.nii.gz").write_bytes(b"junk")
    save(image(), tmp_path / "in" / "ok.nrrd")
    out = run(scans=tmp_path / "in", output_dir=tmp_path / "out")
    assert sorted(p.name for p in out.iterdir()) == ["ok.nii.gz", "run_report.json"]
