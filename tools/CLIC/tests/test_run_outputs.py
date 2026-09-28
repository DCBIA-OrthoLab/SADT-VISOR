"""What `run` writes: the file names, the geometry, and the report.

The network is stubbed throughout; everything else -- discovery, the NIfTI
round trip, the per-scan failure handling, the report -- runs for real.
"""

import json

import numpy as np
import pytest

import sadt_clic


# A scan that is neither axis-aligned nor at unit spacing: 0.3 x 0.4 x 0.5 mm
# with the first two axes swapped and one flipped, which is what a real CBCT
# reconstruction looks like. An identity affine hides every mistake a copied
# header could make.
OBLIQUE_AFFINE = np.array([
    [0.00, -0.30, 0.00, 12.50],
    [0.40, 0.00, 0.00, -7.25],
    [0.00, 0.00, 0.50, 33.00],
    [0.00, 0.00, 0.00, 1.00],
])


def test_the_output_is_named_after_the_scan_with_the_suffix(tmp_path, stubbed, make_scan):
    """`<scan>_<suffix>.nii.gz`, so the segmentation sorts next to its scan."""
    make_scan(tmp_path / "in" / "patient01.nii.gz")

    out = sadt_clic.run(scans=tmp_path / "in", model=tmp_path / "m.pth",
                        output_dir=tmp_path / "out", device="cpu")

    assert (out / "patient01_seg.nii.gz").exists()


def test_the_suffix_is_configurable(tmp_path, stubbed, make_scan):
    make_scan(tmp_path / "in" / "patient01.nii.gz")

    out = sadt_clic.run(scans=tmp_path / "in", model=tmp_path / "m.pth",
                        output_dir=tmp_path / "out", output_suffix="canine",
                        device="cpu")

    assert (out / "patient01_canine.nii.gz").exists()
    assert not (out / "patient01_seg.nii.gz").exists()


def test_a_nii_input_still_produces_a_nii_gz_output(tmp_path, stubbed, make_scan):
    """The extension is not carried over: the output is always compressed, and
    a `.nii` input must not produce `patient.nii_seg.nii.gz`."""
    make_scan(tmp_path / "in" / "patient.nii")

    out = sadt_clic.run(scans=tmp_path / "in", model=tmp_path / "m.pth",
                        output_dir=tmp_path / "out", device="cpu")

    assert [p.name for p in out.glob("*.nii.gz")] == ["patient_seg.nii.gz"]


def test_an_uppercase_extension_is_stripped_from_the_output_name(tmp_path, stubbed):
    """A `.NII.GZ` scan is accepted by discovery, so its extension has to be
    stripped by the same rule or the output is `patient.NII.GZ_seg.nii.gz`."""
    import nibabel as nib

    folder = tmp_path / "in"
    folder.mkdir()
    scan = folder / "patient.NII.GZ"
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 2), dtype=np.float32), np.eye(4)),
             str(scan))
    assert scan.exists(), "nibabel renamed the file; the test would not test this"

    out = sadt_clic.run(scans=folder, model=tmp_path / "m.pth",
                        output_dir=tmp_path / "out", device="cpu")

    assert (out / "patient_seg.nii.gz").exists()


def test_a_single_scan_lands_directly_in_the_output_directory(tmp_path, stubbed, make_scan):
    """A file named directly is its own root: no folder of its parent's name
    appears in the output."""
    scan = make_scan(tmp_path / "somewhere" / "deep" / "patient.nii.gz")

    out = sadt_clic.run(scans=scan, model=tmp_path / "m.pth",
                        output_dir=tmp_path / "out", device="cpu")

    assert (out / "patient_seg.nii.gz").exists()
    assert sorted(p.name for p in out.iterdir()) == [
        "CLIC_report.json", "patient_seg.nii.gz",
    ]


def test_two_scans_of_the_same_name_do_not_overwrite_each_other(tmp_path, stubbed, make_scan):
    """A cohort is exported one folder per patient, and every scan inside is
    called the same thing. Keying the output on the base name alone wrote both
    to `scan_seg.nii.gz`: one patient's segmentation silently replaced the
    other's, while the report said both had been segmented."""
    make_scan(tmp_path / "in" / "patient_A" / "scan.nii.gz")
    make_scan(tmp_path / "in" / "patient_B" / "scan.nii.gz")

    out = sadt_clic.run(scans=tmp_path / "in", model=tmp_path / "m.pth",
                        output_dir=tmp_path / "out", device="cpu")

    written = sorted(str(p.relative_to(out)) for p in out.rglob("*.nii.gz"))
    assert written == ["patient_A/scan_seg.nii.gz", "patient_B/scan_seg.nii.gz"]

    report = json.loads((out / "CLIC_report.json").read_text())
    assert report["summary"] == "2/2 scan(s) segmented"
    assert sorted(s["input"] for s in report["cases"].values()) == [
        "patient_A/scan.nii.gz", "patient_B/scan.nii.gz",
    ]


def test_the_output_mirrors_the_input_tree(tmp_path, stubbed, make_scan):
    """The caller gets back the tree it sent, so a batch can be matched to its
    inputs without reading the report."""
    make_scan(tmp_path / "in" / "site1" / "p1" / "T1" / "scan.nii.gz")

    out = sadt_clic.run(scans=tmp_path / "in", model=tmp_path / "m.pth",
                        output_dir=tmp_path / "out", device="cpu")

    assert (out / "site1" / "p1" / "T1" / "scan_seg.nii.gz").exists()
    entry = next(iter(json.loads(
        (out / "CLIC_report.json").read_text())["cases"].values()))
    assert entry["produced"][0] == "site1/p1/T1/scan_seg.nii.gz"


def test_the_output_keeps_the_input_geometry(tmp_path, stubbed, make_scan):
    """The README's whole claim: the mask keeps the input's affine and header,
    so the scan and its segmentation open aligned in a viewer. Origin, spacing
    and direction, on a scan that has none of them trivially."""
    import nibabel as nib

    scan = make_scan(tmp_path / "in" / "patient.nii.gz", affine=OBLIQUE_AFFINE)

    out = sadt_clic.run(scans=tmp_path / "in", model=tmp_path / "m.pth",
                        output_dir=tmp_path / "out", device="cpu")

    source = nib.load(str(scan))
    result = nib.load(str(out / "patient_seg.nii.gz"))

    np.testing.assert_array_equal(result.affine, source.affine)
    assert result.header.get_zooms()[:3] == source.header.get_zooms()[:3]
    np.testing.assert_allclose(result.affine[:3, :3], OBLIQUE_AFFINE[:3, :3], atol=1e-6)
    np.testing.assert_allclose(result.affine[:3, 3], OBLIQUE_AFFINE[:3, 3], atol=1e-6)


def test_the_output_has_the_shape_of_the_input(tmp_path, stubbed, make_scan):
    """A mask of a different shape opens beside the scan and lines up with
    nothing."""
    import nibabel as nib

    make_scan(tmp_path / "in" / "patient.nii.gz", shape=(9, 7, 5))

    out = sadt_clic.run(scans=tmp_path / "in", model=tmp_path / "m.pth",
                        output_dir=tmp_path / "out", device="cpu")

    assert nib.load(str(out / "patient_seg.nii.gz")).shape == (9, 7, 5)


def test_the_label_values_survive_the_write(tmp_path, stub_network, make_scan, monkeypatch):
    """The header is the input's, so the file's stored dtype and scaling are
    the scan's. The label values still have to read back exactly, or a
    segmentation of class 3 opens as class 2.99."""
    import nibabel as nib

    def segment_volume(model, volume, device, score_threshold):
        labels = np.zeros(volume.shape, dtype=np.int16)
        labels[0, 0, 0] = 1
        labels[1, 1, 0] = 2
        labels[2, 2, 0] = 3
        return labels, 3

    monkeypatch.setattr(sadt_clic, "segment_volume", segment_volume)
    monkeypatch.setattr(sadt_clic, "resolve_device", lambda requested: "cpu")
    make_scan(tmp_path / "in" / "patient.nii.gz", affine=OBLIQUE_AFFINE)

    out = sadt_clic.run(scans=tmp_path / "in", model=tmp_path / "m.pth",
                        output_dir=tmp_path / "out", device="cpu")

    written = np.asarray(nib.load(str(out / "patient_seg.nii.gz")).dataobj)
    assert sorted(np.unique(written).tolist()) == [0, 1, 2, 3]
    assert written[0, 0, 0] == 1 and written[1, 1, 0] == 2 and written[2, 2, 0] == 3


def test_the_output_directory_is_created_when_absent(tmp_path, stubbed, make_scan):
    """The server hands over a job directory that does not exist yet."""
    make_scan(tmp_path / "in" / "patient.nii.gz")
    destination = tmp_path / "not" / "there" / "yet"
    assert not destination.exists()

    out = sadt_clic.run(scans=tmp_path / "in", model=tmp_path / "m.pth",
                        output_dir=destination, device="cpu")

    assert out == destination and destination.is_dir()


def test_run_returns_the_output_directory(tmp_path, stubbed, make_scan):
    """The runner reads the return value to find what was produced."""
    make_scan(tmp_path / "in" / "patient.nii.gz")

    out = sadt_clic.run(scans=tmp_path / "in", model=tmp_path / "m.pth",
                        output_dir=tmp_path / "out", device="cpu")

    assert out == tmp_path / "out"


def test_nothing_is_written_outside_the_output_directory(tmp_path, stubbed, make_scan):
    """The input is the caller's uploaded data. Upstream's ALI wrote its
    conversions into the user's own folder, which the next run re-ingested."""
    folder = tmp_path / "in"
    make_scan(folder / "patient.nii.gz")
    before = sorted(p.name for p in folder.rglob("*"))

    sadt_clic.run(scans=folder, model=tmp_path / "m.pth",
                  output_dir=tmp_path / "out", device="cpu")

    assert sorted(p.name for p in folder.rglob("*")) == before


def test_the_report_is_written_beside_the_segmentations(tmp_path, stubbed, make_scan):
    make_scan(tmp_path / "in" / "patient.nii.gz")

    out = sadt_clic.run(scans=tmp_path / "in", model=tmp_path / "m.pth",
                        output_dir=tmp_path / "out", device="cpu")

    assert (out / "CLIC_report.json").is_file()
    assert json.loads((out / "CLIC_report.json").read_text())["tool"] == "CLIC"


def test_the_report_names_the_model_by_basename_only(tmp_path, stubbed, make_scan):
    """A report travels to the client. A server-side path in it is a leak, and
    the name is what identifies the vintage that ran."""
    make_scan(tmp_path / "in" / "patient.nii.gz")

    out = sadt_clic.run(scans=tmp_path / "in",
                        model=tmp_path / "srv" / "models" / "final_model.pth",
                        output_dir=tmp_path / "out", device="cpu")

    report = json.loads((out / "CLIC_report.json").read_text())
    assert report["model"] == "final_model.pth"
    assert str(tmp_path) not in json.dumps(report)


def test_the_report_records_the_class_count(tmp_path, stubbed, make_scan):
    """Read from the checkpoint, so the report says what the file actually
    held rather than the 4 upstream assumed."""
    make_scan(tmp_path / "in" / "patient.nii.gz")

    out = sadt_clic.run(scans=tmp_path / "in", model=tmp_path / "m.pth",
                        output_dir=tmp_path / "out", device="cpu")

    assert json.loads((out / "CLIC_report.json").read_text())["classes"] == 4


def test_the_report_records_the_score_threshold(tmp_path, stubbed, make_scan):
    """A caller comparing two runs has to be able to say which value produced
    which."""
    make_scan(tmp_path / "in" / "patient.nii.gz")

    out = sadt_clic.run(scans=tmp_path / "in", model=tmp_path / "m.pth",
                        output_dir=tmp_path / "out", score_threshold=0.42,
                        device="cpu")

    assert json.loads((out / "CLIC_report.json").read_text())["score_threshold"] == 0.42


def test_the_threshold_reaches_the_segmentation(tmp_path, stub_network, make_scan, monkeypatch):
    """Recorded AND applied: a report naming a value the run did not use is
    worse than no report."""
    seen = []

    def segment_volume(model, volume, device, score_threshold):
        seen.append(score_threshold)
        return np.zeros(volume.shape, dtype=np.int16), 1

    monkeypatch.setattr(sadt_clic, "segment_volume", segment_volume)
    monkeypatch.setattr(sadt_clic, "resolve_device", lambda requested: "cpu")
    make_scan(tmp_path / "in" / "patient.nii.gz")

    sadt_clic.run(scans=tmp_path / "in", model=tmp_path / "m.pth",
                  output_dir=tmp_path / "out", score_threshold=0.42, device="cpu")

    assert seen == [0.42]


def test_the_default_threshold_is_the_one_upstream_hardcoded(tmp_path, stubbed, make_scan):
    """0.7, so a caller who names nothing gets what upstream did."""
    from sadt_clic import pipeline

    make_scan(tmp_path / "in" / "patient.nii.gz")

    out = sadt_clic.run(scans=tmp_path / "in", model=tmp_path / "m.pth",
                        output_dir=tmp_path / "out", device="cpu")

    assert pipeline.DEFAULT_SCORE_THRESHOLD == 0.7
    assert json.loads((out / "CLIC_report.json").read_text())["score_threshold"] == 0.7


def test_the_report_counts_the_slices(tmp_path, stubbed, make_scan):
    """The third axis, which is what the network was run over."""
    make_scan(tmp_path / "in" / "patient.nii.gz", shape=(6, 5, 11))

    out = sadt_clic.run(scans=tmp_path / "in", model=tmp_path / "m.pth",
                        output_dir=tmp_path / "out", device="cpu")

    assert next(iter(json.loads(
        (out / "CLIC_report.json").read_text())["cases"].values()))["slices"] == 11


def test_the_report_lists_the_labels_present(tmp_path, stub_network, make_scan, monkeypatch):
    """What was actually painted, sorted, with the background left out -- the
    only way a caller sees that a detection painted nothing."""
    def segment_volume(model, volume, device, score_threshold):
        labels = np.zeros(volume.shape, dtype=np.int16)
        labels[0, 0, 0] = 3
        labels[1, 1, 0] = 1
        return labels, 5

    monkeypatch.setattr(sadt_clic, "segment_volume", segment_volume)
    monkeypatch.setattr(sadt_clic, "resolve_device", lambda requested: "cpu")
    make_scan(tmp_path / "in" / "patient.nii.gz")

    out = sadt_clic.run(scans=tmp_path / "in", model=tmp_path / "m.pth",
                        output_dir=tmp_path / "out", device="cpu")

    entry = next(iter(json.loads(
        (out / "CLIC_report.json").read_text())["cases"].values()))
    assert entry["labels_present"] == [1, 3]
    assert entry["detections"] == 5


def test_the_report_has_a_duration(tmp_path, stubbed, make_scan):
    make_scan(tmp_path / "in" / "patient.nii.gz")

    out = sadt_clic.run(scans=tmp_path / "in", model=tmp_path / "m.pth",
                        output_dir=tmp_path / "out", device="cpu")

    report = json.loads((out / "CLIC_report.json").read_text())
    assert isinstance(report["duration_seconds"], float)
    assert report["duration_seconds"] >= 0


def test_a_four_dimensional_scan_is_a_per_scan_failure(tmp_path, stubbed, make_scan):
    """A DWI or a time series is not a CBCT, and the network is fed the third
    axis. Refused per scan, with the dimension count in the reason."""
    make_scan(tmp_path / "in" / "good.nii.gz")
    make_scan(tmp_path / "in" / "series.nii.gz", shape=(4, 4, 2, 3))

    out = sadt_clic.run(scans=tmp_path / "in", model=tmp_path / "m.pth",
                        output_dir=tmp_path / "out", device="cpu")

    report = json.loads((out / "CLIC_report.json").read_text())
    failed = [s for s in report["cases"].values() if s.get("status") == "failed"]
    assert [s["input"] for s in failed] == ["series.nii.gz"]
    assert "3D volume" in failed[0]["reason"]
    assert (out / "good_seg.nii.gz").exists()


def test_a_failed_scan_writes_no_output(tmp_path, stubbed, make_scan):
    """Half a segmentation on disk is worse than none: the caller would load
    it beside the scan."""
    make_scan(tmp_path / "in" / "good.nii.gz")
    (tmp_path / "in" / "broken.nii.gz").write_bytes(b"not a nifti")

    out = sadt_clic.run(scans=tmp_path / "in", model=tmp_path / "m.pth",
                        output_dir=tmp_path / "out", device="cpu")

    assert sorted(p.name for p in out.glob("*.nii.gz")) == ["good_seg.nii.gz"]


def test_a_failure_reason_names_the_exception_type(tmp_path, stubbed, make_scan):
    """`<Type>: <message>`, so a caller reading the report can tell an
    unreadable file from a shape mismatch."""
    make_scan(tmp_path / "in" / "good.nii.gz")
    (tmp_path / "in" / "broken.nii.gz").write_bytes(b"not a nifti")

    out = sadt_clic.run(scans=tmp_path / "in", model=tmp_path / "m.pth",
                        output_dir=tmp_path / "out", device="cpu")

    failed = [s for s in json.loads((out / "CLIC_report.json").read_text())["cases"].values()
              if s.get("status") == "failed"]
    assert ":" in failed[0]["reason"]
    assert failed[0]["reason"].split(":")[0].endswith("Error")


def test_a_batch_where_every_scan_fails_is_refused(tmp_path, stubbed):
    """The guard counts what was WRITTEN, not what was walked past: a run of
    two unreadable files must not return success with an empty output."""
    (tmp_path / "in").mkdir()
    (tmp_path / "in" / "a.nii.gz").write_bytes(b"not a nifti")
    (tmp_path / "in" / "b.nii.gz").write_bytes(b"not a nifti either")

    with pytest.raises(ValueError) as raised:
        sadt_clic.run(scans=tmp_path / "in", model=tmp_path / "m.pth",
                      output_dir=tmp_path / "out", device="cpu")

    message = str(raised.value)
    assert "segmented none" in message
    assert "a.nii.gz" in message and "b.nii.gz" in message


def test_no_report_is_written_when_every_scan_failed(tmp_path, stubbed):
    """The reasons travel in the exception, which the server maps to a 422. A
    report claiming 0/2 beside an empty folder would be read as a result."""
    (tmp_path / "in").mkdir()
    (tmp_path / "in" / "a.nii.gz").write_bytes(b"not a nifti")

    with pytest.raises(ValueError):
        sadt_clic.run(scans=tmp_path / "in", model=tmp_path / "m.pth",
                      output_dir=tmp_path / "out", device="cpu")

    assert not (tmp_path / "out" / "CLIC_report.json").exists()


def test_an_empty_input_names_what_was_looked_for(tmp_path, stubbed):
    """The message reaches the client verbatim as a 422, so it has to say what
    a scan is, and that NRRD and MetaImage are not it."""
    folder = tmp_path / "cohort_2026"
    folder.mkdir()
    (folder / "scan.nrrd").write_bytes(b"NRRD0004")

    with pytest.raises(ValueError) as raised:
        sadt_clic.run(scans=folder, model=tmp_path / "m.pth",
                      output_dir=tmp_path / "out", device="cpu")

    message = str(raised.value)
    assert ".nii" in message and ".nii.gz" in message
    assert "cohort_2026" in message
    assert "NRRD" in message


def test_a_missing_input_is_refused_the_same_way(tmp_path, stubbed):
    """Not an OSError out of the walk, which the server would map to a 500."""
    with pytest.raises(ValueError, match="No .nii or .nii.gz scan"):
        sadt_clic.run(scans=tmp_path / "nowhere", model=tmp_path / "m.pth",
                      output_dir=tmp_path / "out", device="cpu")


def test_an_empty_input_is_refused_before_the_model_is_loaded(tmp_path, monkeypatch):
    """Discovery comes first, so a request with no scan costs no checkpoint
    load -- which for this tool is 176 MB off disk."""
    loaded = []

    def build_model(checkpoint_path, device):
        loaded.append(checkpoint_path)
        return object(), 4

    monkeypatch.setattr(sadt_clic, "build_model", build_model)
    monkeypatch.setattr(sadt_clic, "resolve_device", lambda requested: "cpu")
    (tmp_path / "in").mkdir()

    with pytest.raises(ValueError):
        sadt_clic.run(scans=tmp_path / "in", model=tmp_path / "m.pth",
                      output_dir=tmp_path / "out", device="cpu")

    assert loaded == []


def test_a_missing_model_file_is_named(tmp_path, make_scan, no_cuda):
    """`build_model` runs before the loop, so a wrong checkpoint name fails
    once with the file in the message rather than once per scan."""
    make_scan(tmp_path / "in" / "patient.nii.gz")

    with pytest.raises(FileNotFoundError) as raised:
        sadt_clic.run(scans=tmp_path / "in", model=tmp_path / "wrong_name.pth",
                      output_dir=tmp_path / "out", device="cpu")

    assert "wrong_name.pth" in str(raised.value)


def test_the_progress_protocol_is_not_printed(tmp_path, stubbed, make_scan, capsys):
    """Upstream's `[PROGRESS]`/`[LOG]`/`[SEG]` lines drove the Qt panel's
    progress bar. Here stdout is the runner's channel for `result.json`, and a
    tool that prints into it corrupts the run."""
    make_scan(tmp_path / "in" / "patient.nii.gz")
    capsys.readouterr()

    sadt_clic.run(scans=tmp_path / "in", model=tmp_path / "m.pth",
                  output_dir=tmp_path / "out", device="cpu")

    assert capsys.readouterr().out == ""
