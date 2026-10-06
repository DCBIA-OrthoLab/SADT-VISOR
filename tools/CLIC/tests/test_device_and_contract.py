"""The device choice, and the two claims the README makes about what this
package will NOT do: import a CUDA stack to publish its schema, and reach the
network for a checkpoint.
"""

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

import sadt_clic
from sadt_clic import pipeline

SRC = str(Path(__file__).resolve().parents[1] / "src")


def test_cuda_falls_back_to_cpu_when_no_card_is_visible(no_cuda):
    """The server injects its own DEVICE when the caller names none, and that
    setting describes the deployment rather than this request. Asking for cuda
    on a CPU machine has to run, not raise."""
    assert pipeline.resolve_device("cuda") == "cpu"


def test_the_fallback_is_logged(no_cuda, caplog):
    """A run that quietly took forty times longer must say why."""
    with caplog.at_level(logging.WARNING, logger="sadt_clic"):
        pipeline.resolve_device("cuda")

    assert any("cpu" in record.message for record in caplog.records)
    assert any(record.levelno == logging.WARNING for record in caplog.records)


def test_cuda_is_honoured_when_a_card_is_visible(with_cuda):
    """The fallback is a fallback, not a policy: a GPU deployment must not be
    silently downgraded."""
    assert pipeline.resolve_device("cuda") == "cuda"


def test_cpu_is_never_upgraded_to_cuda(with_cuda):
    """A caller who asked for cpu gets cpu even on a machine with a card --
    which is how a tabular-sized request stays off a busy GPU."""
    assert pipeline.resolve_device("cpu") == "cpu"


def test_the_report_records_the_device_actually_used(tmp_path, stub_network,
                                                     make_scan, no_cuda):
    """Not the device that was requested. A run silently on the CPU is the
    explanation for a batch that took hours."""
    make_scan(tmp_path / "in" / "patient.nii.gz")

    out = sadt_clic.run(scans=tmp_path / "in", model=tmp_path / "m.pth",
                        output_dir=tmp_path / "out", device="cuda")

    assert json.loads((out / "CLIC_report.json").read_text())["device"] == "cpu"


def test_the_device_reaches_the_segmentation(tmp_path, stub_network, make_scan,
                                             no_cuda, monkeypatch):
    """Recorded AND used: the tensors are built on whatever this resolved to."""
    seen = []

    def segment_volume(model, volume, device, score_threshold):
        import numpy as np

        seen.append(device)
        return np.zeros(volume.shape, dtype=np.int16), 0

    monkeypatch.setattr(sadt_clic, "segment_volume", segment_volume)
    make_scan(tmp_path / "in" / "patient.nii.gz")

    sadt_clic.run(scans=tmp_path / "in", model=tmp_path / "m.pth",
                  output_dir=tmp_path / "out", device="cuda")

    assert seen == ["cpu"]


def test_the_default_device_is_cuda():
    """A GPU server is the deployment; the fallback covers the rest."""
    import inspect

    assert inspect.signature(sadt_clic.run).parameters["device"].default == "cuda"


def _import_in_a_fresh_interpreter(statement):
    """Run `statement` in a subprocess, because this one has already imported
    torch and could never observe a lazy import."""
    result = subprocess.run(
        [sys.executable, "-c", statement],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": SRC},
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_importing_the_package_does_not_import_torch():
    """CI imports this package on every pull request to publish its schema,
    and that must not cost a CUDA stack."""
    loaded = _import_in_a_fresh_interpreter(
        "import sys, sadt_clic; print('torch' in sys.modules)"
    )

    assert loaded == "False"


def test_importing_the_package_does_not_import_torchvision():
    loaded = _import_in_a_fresh_interpreter(
        "import sys, sadt_clic; print('torchvision' in sys.modules)"
    )

    assert loaded == "False"


def test_importing_the_package_does_not_import_nibabel():
    """The same argument: reading a schema must not open the I/O stack."""
    loaded = _import_in_a_fresh_interpreter(
        "import sys, sadt_clic; print('nibabel' in sys.modules)"
    )

    assert loaded == "False"


def test_the_public_api_is_run_alone():
    """What the runner calls, and the only thing a schema is generated from."""
    assert sadt_clic.__all__ == ["run"]
    assert callable(sadt_clic.run)


def test_run_declares_every_argument_the_schema_publishes():
    """`describe.py` builds the published schema from this signature, so an
    argument that is not here cannot be sent, and one renamed here silently
    changes the wire contract."""
    import inspect

    signature = inspect.signature(sadt_clic.run)
    parameters = signature.parameters

    # `model` moved AFTER `output_dir` and gained a default, deliberately: that
    # is the shape that lets a panel show "(automatic)" and a supervised call
    # omit it. There is one published checkpoint, so naming it is a step that
    # can only be got wrong.
    published = [
        name for name, spec in parameters.items()
        if spec.kind is not inspect.Parameter.KEYWORD_ONLY
    ]
    assert published == [
        "scans", "output_dir", "model", "score_threshold", "output_suffix", "device",
    ]
    assert parameters["scans"].default is inspect.Parameter.empty
    assert parameters["output_dir"].default is inspect.Parameter.empty
    assert parameters["model"].default == ""

    # Keyword-only, and that is what keeps it OFF the wire: the server injects
    # the data root, a caller never names it. A test, because moving it back to
    # a positional would publish a server path as a clinical argument.
    assert parameters["data_root"].kind is inspect.Parameter.KEYWORD_ONLY


@pytest.mark.parametrize("forbidden", ["urllib", "requests", "urlretrieve",
                                       "http://", "https://"])
def test_the_runtime_model_download_is_not_ported(forbidden):
    """Upstream downloaded the checkpoint from a GitHub release at run time. A
    server holding patient data does not make outbound calls mid-request; the
    bundle is staged by `setup-models.sh --tool CLIC`."""
    source = "\n".join(
        path.read_text() for path in sorted(Path(SRC).rglob("*.py"))
    )

    assert forbidden not in source


def test_the_advertised_extensions_are_the_ones_that_are_read():
    """Upstream advertised `.nrrd`, `.mha` and `.mhd`, none of which nibabel
    reads: accepted and then ignored, the trap `.stl` fell into in ALI."""
    assert pipeline.SCAN_EXTENSIONS == (".nii", ".nii.gz")
    for extension in pipeline.SCAN_EXTENSIONS:
        assert pipeline.is_scan_file("patient" + extension) is True


# ---------------------------------------------------------------------------
# the real checkpoint
# ---------------------------------------------------------------------------
#
# Deselected by default (`addopts` in pyproject.toml). Everything above runs
# without it; these two are the only claims that cannot be made with a stub,
# because what they check is the agreement between the real 176 MB file and
# the architecture this package builds for it.

REAL_MODEL = os.environ.get("SADT_CLIC_MODEL")


@pytest.mark.models
@pytest.mark.skipif(not REAL_MODEL, reason="set SADT_CLIC_MODEL to the .pth")
def test_the_real_checkpoint_loads_into_the_heads_built_for_it():
    """`load_state_dict` is strict, so this fails on any tensor whose shape the
    class count read out of the file did not predict. A stub cannot make that
    claim: it is precisely the agreement between the file and the two resized
    heads that is being asserted."""
    network, classes = pipeline.build_model(REAL_MODEL, "cpu")

    assert classes >= 2, "background plus at least one canine class"
    assert network.roi_heads.box_predictor.cls_score.out_features == classes
    assert network.roi_heads.mask_predictor.mask_fcn_logits.out_channels == classes
    assert not network.training, "the network is put in eval mode before it is used"


@pytest.mark.models
@pytest.mark.skipif(not REAL_MODEL, reason="set SADT_CLIC_MODEL to the .pth")
def test_the_real_network_returns_what_segment_volume_reads(tmp_path):
    """The real torchvision detector's output dict, against the keys and shapes
    `segment_volume` indexes: `scores`, `labels`, and `masks` with a channel
    axis to squeeze. A stub agrees with itself by construction."""
    import numpy as np

    network, _ = pipeline.build_model(REAL_MODEL, "cpu")
    volume = np.random.default_rng(0).random((64, 64, 2)).astype(np.float32)

    labels, detections = pipeline.segment_volume(network, volume, "cpu", 0.7)

    assert labels.shape == volume.shape
    assert labels.dtype == np.int16
    assert detections >= 0
