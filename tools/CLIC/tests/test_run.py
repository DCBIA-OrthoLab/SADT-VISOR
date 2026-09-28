"""CLIC, with the network stubbed: no GPU, no checkpoint, no card."""

import json
import logging
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import sadt_clic
from sadt_clic import pipeline


def _write_scan(path, shape=(8, 8, 3)):
    import nibabel as nib

    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.random.default_rng(0).random(shape).astype(np.float32)
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))
    return path


@pytest.fixture
def stubbed(monkeypatch):
    """Replace the network, so everything around it runs for real."""
    def build_model(checkpoint_path, device):
        return object(), 4

    def segment_volume(model, volume, device, score_threshold):
        labels = np.zeros(volume.shape, dtype=np.int16)
        labels[0, 0, 0] = 3
        return labels, 1

    monkeypatch.setattr(sadt_clic, "build_model", build_model)
    monkeypatch.setattr(sadt_clic, "segment_volume", segment_volume)
    monkeypatch.setattr(sadt_clic, "resolve_device", lambda requested: "cpu")


def test_a_folder_is_walked_recursively(tmp_path):
    """Upstream looked one level down and returned the DIRECTORIES that held a
    scan, which the runner then handed to `nib.load`."""
    _write_scan(tmp_path / "a.nii.gz")
    _write_scan(tmp_path / "deep" / "nested" / "b.nii.gz")

    found = pipeline.discover_scans(str(tmp_path))

    assert [os.path.basename(f) for f in found] == ["a.nii.gz", "b.nii.gz"]
    assert all(os.path.isfile(f) for f in found), "a directory reached the loader"


@pytest.mark.parametrize("name,expected", [
    ("scan.nii", True), ("scan.nii.gz", True), ("SCAN.NII.GZ", True),
    # Advertised by upstream's collector and unreadable by nibabel: accepted
    # and then ignored, which is the trap `.stl` fell into in ALI.
    ("scan.nrrd", False), ("scan.mha", False), ("scan.mhd", False),
])
def test_only_the_extensions_nibabel_reads_are_advertised(name, expected):
    assert pipeline.is_scan_file(name) is expected


def test_the_class_count_comes_from_the_checkpoint():
    """Upstream built both heads for 4 classes whatever the file held."""
    import torch

    state = {"roi_heads.box_predictor.cls_score.bias": torch.zeros(7)}
    assert pipeline.class_count(state) == 7

    with pytest.raises(ValueError, match="not a Mask R-CNN"):
        pipeline.class_count({"something.else": torch.zeros(3)})


def test_a_flat_slice_normalises_to_zeros_not_nans():
    flat = np.full((4, 4), 12.0, dtype=np.float32)
    assert not np.isnan(pipeline.normalise(flat)).any()
    assert (pipeline.normalise(flat) == 0).all()


def test_a_batch_reports_every_scan(tmp_path, stubbed):
    _write_scan(tmp_path / "in" / "one.nii.gz")
    _write_scan(tmp_path / "in" / "two.nii.gz")

    out = sadt_clic.run(scans=tmp_path / "in", model=tmp_path / "m.pth",
                        output_dir=tmp_path / "out", device="cpu")

    report = json.loads((out / "CLIC_report.json").read_text())
    assert report["summary"] == "2/2 scan(s) segmented"
    assert sorted(s["input"] for s in report["cases"].values()) == ["one.nii.gz", "two.nii.gz"]
    assert all(s["labels_present"] == [3] for s in report["cases"].values())
    assert (out / "one_seg.nii.gz").exists() and (out / "two_seg.nii.gz").exists()


def test_an_input_with_no_scan_is_refused(tmp_path, stubbed):
    """A batch that could read nothing is a 422, not a successful run of zero
    scans."""
    (tmp_path / "in").mkdir()
    (tmp_path / "in" / "notes.txt").write_text("nothing here")

    with pytest.raises(ValueError, match="No .nii or .nii.gz scan"):
        sadt_clic.run(scans=tmp_path / "in", model=tmp_path / "m.pth",
                      output_dir=tmp_path / "out", device="cpu")


def test_one_bad_scan_does_not_cost_the_others(tmp_path, stubbed, monkeypatch):
    _write_scan(tmp_path / "in" / "good.nii.gz")
    (tmp_path / "in" / "broken.nii.gz").write_bytes(b"not a nifti")

    out = sadt_clic.run(scans=tmp_path / "in", model=tmp_path / "m.pth",
                        output_dir=tmp_path / "out", device="cpu")

    report = json.loads((out / "CLIC_report.json").read_text())
    assert report["summary"] == "1/2 scan(s) segmented"
    failed = [s for s in report["cases"].values() if s["status"] == "failed"]
    assert len(failed) == 1 and "reason" in failed[0]


def test_a_scan_with_no_detection_says_so(tmp_path, stubbed, monkeypatch):
    """An empty segmentation is a legitimate answer AND the signature of a
    wrong checkpoint. Only the caller can tell them apart, so it is reported."""
    monkeypatch.setattr(
        sadt_clic, "segment_volume",
        lambda model, volume, device, threshold: (np.zeros(volume.shape, np.int16), 0),
    )
    _write_scan(tmp_path / "in" / "quiet.nii.gz")

    out = sadt_clic.run(scans=tmp_path / "in", model=tmp_path / "m.pth",
                        output_dir=tmp_path / "out", device="cpu")

    entry = next(iter(json.loads(
        (out / "CLIC_report.json").read_text())["cases"].values()))
    assert entry["status"] == "ok"
    assert entry["detections"] == 0
    assert "no detection" in entry["note"]


def test_a_batch_says_where_it_has_got_to(tmp_path, stubbed, monkeypatch):
    """One event per scan, rising, and the position rather than the name.

    A forty-scan batch is one HTTP request that takes an hour; without this the
    client can only show that the connection is still open. The file name stays
    out of it: these messages are stored on the server and shown, and a file
    name is patient metadata.
    """
    events_file = tmp_path / "events.jsonl"
    monkeypatch.setenv("SADT_PROGRESS_FILE", str(events_file))
    _write_scan(tmp_path / "in" / "one.nii.gz")
    _write_scan(tmp_path / "in" / "two.nii.gz")

    sadt_clic.run(scans=tmp_path / "in", model=tmp_path / "m.pth",
                  output_dir=tmp_path / "out", device="cpu")

    events = [json.loads(line) for line in events_file.read_text().splitlines() if line]
    assert [e["message"] for e in events] == ["scan 1 of 2", "scan 2 of 2"]
    assert [e["fraction"] for e in events] == [0.0, 0.5]


def test_nothing_is_written_when_no_server_asked_for_progress(tmp_path, stubbed, monkeypatch):
    """The variable is unset in a checkout and against an older server, and a
    tool must not need a fallback path for that."""
    monkeypatch.delenv("SADT_PROGRESS_FILE", raising=False)
    _write_scan(tmp_path / "in" / "one.nii.gz")

    out = sadt_clic.run(scans=tmp_path / "in", model=tmp_path / "m.pth",
                        output_dir=tmp_path / "out", device="cpu")

    assert (out / "one_seg.nii.gz").exists()


def test_a_failure_names_the_position_and_never_the_scan(tmp_path, stubbed, caplog):
    """The rule the progress messages follow, applied to the log -- and this is
    the line that needed it most.

    A tool's stderr is captured to a file in the job directory, and on a FAILED
    run the server copies its tail into its own persistent log. Every one of
    these `logger.exception` calls is on a failure path, so a name written here
    is a patient identifier that outlives the run and its job directory.

    Asserted on the composed message: an exception raised inside a third-party
    loader can still name the file it could not open, and that is a separate
    exposure this test does not claim to close.
    """
    _write_scan(tmp_path / "in" / "healthy.nii.gz")
    (tmp_path / "in" / "Smith_John_T1.nii.gz").write_bytes(b"not a nifti")

    with caplog.at_level(logging.INFO, logger="CLIC"):
        sadt_clic.run(scans=tmp_path / "in", model=tmp_path / "m.pth",
                      output_dir=tmp_path / "out", device="cpu")

    messages = [record.getMessage() for record in caplog.records]
    assert any(m.startswith("CLIC failed on scan ") and m.endswith(" of 2")
               for m in messages), messages
    assert not any("Smith_John" in m for m in messages), messages


def test_the_report_names_the_classes(tmp_path, stubbed):
    """The integers ARE the finding: buccal, bicortical or palatal decides the
    surgical approach. Upstream recorded the mapping nowhere -- it painted a
    legend over the slice views and wrote an unnamed volume."""
    _write_scan(tmp_path / "in" / "one.nii.gz")

    out = sadt_clic.run(scans=tmp_path / "in", model=tmp_path / "m.pth",
                        output_dir=tmp_path / "out", device="cpu")

    report = json.loads((out / "CLIC_report.json").read_text())
    assert report["labels"] == {"Buccal": 1, "Bicortical": 2, "Palatal": 3}
    assert report["label_colors"]["Palatal"] == [0.6, 0.4, 0.2]
    # The stub paints label 3, and what a reader wants is the word.
    assert next(iter(report["cases"].values()))["detected"] == ["Palatal"]


def test_a_checkpoint_of_another_shape_is_published_unnamed(tmp_path, stubbed, monkeypatch):
    """A wrong table does not fail, it renames the finding -- and the finding
    here is which surgical approach the canine calls for."""
    monkeypatch.setattr(sadt_clic, "build_model", lambda path, device: (object(), 7))
    _write_scan(tmp_path / "in" / "one.nii.gz")

    out = sadt_clic.run(scans=tmp_path / "in", model=tmp_path / "m.pth",
                        output_dir=tmp_path / "out", device="cpu")

    report = json.loads((out / "CLIC_report.json").read_text())
    assert report["labels"] is None and report["label_colors"] is None
    assert "7 classes" in report["labels_note"]
    assert next(iter(report["cases"].values()))["labels_present"] == [3]
    assert next(iter(report["cases"].values()))["detected"] == []


def test_the_installed_checkpoint_is_used_when_none_is_named(tmp_path, stubbed):
    """A panel left on "(automatic)" sends no model, and the tool finds its
    own -- which is what stops a caller from naming its neighbour's weights."""
    _write_scan(tmp_path / "in" / "one.nii.gz")
    models = tmp_path / "data" / "CLIC" / "models"
    models.mkdir(parents=True)
    (models / "final_model.pth").write_bytes(b"weights")

    out = sadt_clic.run(scans=tmp_path / "in", output_dir=tmp_path / "out",
                        device="cpu", data_root=tmp_path / "data")

    report = json.loads((out / "CLIC_report.json").read_text())
    assert report["model"] == "final_model.pth"


def test_several_installed_checkpoints_and_no_name_is_refused(tmp_path, stubbed):
    """Upstream took the alphabetically FIRST `.pth`, so which model vintage ran
    depended on file names. Refusing names both instead."""
    _write_scan(tmp_path / "in" / "one.nii.gz")
    models = tmp_path / "data" / "CLIC" / "models"
    models.mkdir(parents=True)
    (models / "final_model.pth").write_bytes(b"weights")
    (models / "older_model.pth").write_bytes(b"weights")

    with pytest.raises(ValueError, match="final_model.pth, older_model.pth"):
        sadt_clic.run(scans=tmp_path / "in", output_dir=tmp_path / "out",
                      device="cpu", data_root=tmp_path / "data")


def test_no_checkpoint_installed_names_the_command_that_fetches_one(tmp_path, stubbed):
    _write_scan(tmp_path / "in" / "one.nii.gz")
    (tmp_path / "data" / "CLIC" / "models").mkdir(parents=True)

    with pytest.raises(ValueError, match="setup-models.sh --tool CLIC"):
        sadt_clic.run(scans=tmp_path / "in", output_dir=tmp_path / "out",
                      device="cpu", data_root=tmp_path / "data")


def test_no_model_and_no_data_root_says_which_of_the_two_is_missing(tmp_path, stubbed):
    """A checkout running this by hand, rather than a server that publishes a
    data root."""
    _write_scan(tmp_path / "in" / "one.nii.gz")

    with pytest.raises(ValueError, match="no data root"):
        sadt_clic.run(scans=tmp_path / "in", output_dir=tmp_path / "out",
                      device="cpu")
