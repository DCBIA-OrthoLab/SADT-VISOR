"""`build_model` builds the network without fetching anything.

`maskrcnn_resnet50_fpn(weights=None)` still defaults `weights_backbone` to the
ImageNet ResNet50 and downloads it into TORCH_HOME. In a container where that
directory is read-only the request died with a PermissionError while "loading
checkpoint", and where it is writable a server holding patient data made an
outbound call mid-request -- all for weights the checkpoint then replaces.
"""

import socket

import pytest
import torch

from sadt_clic import pipeline


def _reference_state(classes):
    """A checkpoint in the layout of the published one, with random weights.

    Built from torchvision's own `maskrcnn_resnet50_fpn` (with no pretrained
    weights, so building it fetches nothing either), minus the
    `num_batches_tracked` buffers that only the BatchNorm2d it then picks has:
    what is left is exactly the key set of the FrozenBatchNorm2d network the
    published checkpoint was saved from.
    """
    from torchvision.models.detection import maskrcnn_resnet50_fpn

    torch.manual_seed(0)
    reference = maskrcnn_resnet50_fpn(
        weights=None, weights_backbone=None, num_classes=classes
    )
    return {
        key: value
        for key, value in reference.state_dict().items()
        if not key.endswith("num_batches_tracked")
    }


@pytest.fixture
def no_network(monkeypatch, tmp_path):
    """Every route to a download raises, and the weights cache starts empty."""
    torch_home = tmp_path / "torch-home"
    torch_home.mkdir()
    monkeypatch.setenv("TORCH_HOME", str(torch_home))

    def refuse(*args, **kwargs):
        raise AssertionError("build_model attempted a download")

    import torchvision.models._api as weights_api

    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", refuse)
    monkeypatch.setattr(torch.hub, "download_url_to_file", refuse)
    monkeypatch.setattr(weights_api, "load_state_dict_from_url", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)
    return torch_home


def test_building_the_network_downloads_nothing(tmp_path, no_network):
    checkpoint = tmp_path / "clic.pth"
    torch.save(_reference_state(classes=5), checkpoint)

    network, classes = pipeline.build_model(str(checkpoint), "cpu")

    assert classes == 5
    assert network.roi_heads.box_predictor.cls_score.out_features == 5
    assert list(no_network.iterdir()) == [], "nothing was written to TORCH_HOME"


def test_every_weight_comes_from_the_checkpoint(tmp_path, no_network):
    """The load is strict and covers every tensor the network holds, so no
    weight is left to an initialisation -- pretrained or random -- and the
    network computes exactly what the checkpoint saved."""
    state = _reference_state(classes=4)
    checkpoint = tmp_path / "clic.pth"
    torch.save(state, checkpoint)

    network, _ = pipeline.build_model(str(checkpoint), "cpu")

    built = network.state_dict()
    assert set(built) == set(state)
    for key, value in state.items():
        assert torch.equal(built[key], value), key


def test_the_backbone_keeps_its_frozen_batch_norm(tmp_path, no_network):
    """Dropping the pretrained backbone must not change the architecture:
    torchvision swaps FrozenBatchNorm2d for BatchNorm2d when it is given no
    pretrained weights, and that layer is pinned here."""
    from torchvision.ops.misc import FrozenBatchNorm2d

    checkpoint = tmp_path / "clic.pth"
    torch.save(_reference_state(classes=4), checkpoint)

    network, _ = pipeline.build_model(str(checkpoint), "cpu")

    norms = [
        module for module in network.backbone.modules()
        if isinstance(module, (FrozenBatchNorm2d, torch.nn.BatchNorm2d))
    ]
    assert norms
    assert all(isinstance(module, FrozenBatchNorm2d) for module in norms)
