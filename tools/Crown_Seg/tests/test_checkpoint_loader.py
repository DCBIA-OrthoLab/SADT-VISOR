"""The `torch.load` wrapper the macOS block of `_run_shapeaxi` installs.

It must load a plain state dict on the CPU, refuse a checkpoint that carries
code (`weights_only=True` by default, even with
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 in the environment), and be installed once
per process, however many times `_run_shapeaxi` runs.
"""

import pickle

import pytest
from sadt_crownseg import pipeline

torch = pytest.importorskip("torch")


class _Trap:
    """Unpickling this calls `marker_writer(path)`: code execution on load."""

    def __init__(self, path):
        self.path = path

    def __reduce__(self):
        return (_write_marker, (self.path,))


def _write_marker(path):
    with open(path, "w") as handle:
        handle.write("executed")
    return {}


@pytest.fixture
def installed(monkeypatch):
    """Install the wrapper on a throwaway `torch.load`, restored afterwards."""
    monkeypatch.setattr(torch, "load", torch.load)
    return pipeline._install_checkpoint_loader()


def test_a_plain_state_dict_loads(installed, tmp_path):
    state = {"model.0.weight": torch.arange(4.0), "model.0.bias": torch.zeros(2)}
    path = tmp_path / "plain.pth"
    torch.save(state, path)
    loaded = torch.load(path)
    assert set(loaded) == set(state)
    assert torch.equal(loaded["model.0.weight"], state["model.0.weight"])


@pytest.mark.parametrize("force_env", [False, True])
def test_a_trapped_checkpoint_is_refused_and_not_executed(installed, tmp_path, monkeypatch, force_env):
    if force_env:
        monkeypatch.setenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    marker = tmp_path / "marker"
    path = tmp_path / "trapped.pth"
    torch.save({"w": torch.zeros(1), "x": _Trap(str(marker))}, path)
    with pytest.raises(pickle.UnpicklingError):
        torch.load(path)
    assert not marker.exists()


def test_an_explicit_choice_of_the_caller_is_not_overridden(installed, tmp_path):
    """The default is set, never forced: the caller keeps `map_location`."""
    path = tmp_path / "plain.pth"
    torch.save({"w": torch.ones(1)}, path)
    assert torch.load(path, map_location="cpu", weights_only=True)["w"].item() == 1.0


def test_it_is_installed_once(monkeypatch):
    monkeypatch.setattr(torch, "load", torch.load)
    first = pipeline._install_checkpoint_loader()
    second = pipeline._install_checkpoint_loader()
    assert first is second is torch.load
    assert not getattr(first.__wrapped__, pipeline._LOADER_MARK, False)


def test_the_installer_is_only_called_inside_the_darwin_block():
    import inspect

    body = inspect.getsource(pipeline._run_shapeaxi)
    before_call = body.split("_install_checkpoint_loader()")[0]
    assert "if gl_raster.wanted(device):" in before_call
