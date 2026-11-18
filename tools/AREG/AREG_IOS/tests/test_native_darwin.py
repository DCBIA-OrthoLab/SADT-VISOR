"""On macOS, AREG_IOS's patch network renders without importing pytorch3d, and only there.

The platform is simulated the way `test_embree_raster.py` does it, by setting
`sys.platform`; `sys.modules` holds `None` under `pytorch3d` while the tests run,
so ANY attempt to import it raises, and each test then asserts nothing named
`pytorch3d` was loaded. The fresh-interpreter test is the same claim with
nothing else imported first.

The renderer needs `embreex` (not a dependency of the Linux deployments), so
this file skips without it.
"""

import subprocess
import sys

import numpy as np
import pytest
import torch

from sadt_areg_ios import net, render_native

pytest.importorskip("embreex")
pytest.importorskip("monai")


def bumpy():
    from test_render_native import icosphere

    verts, faces = icosphere(subdivisions=3)
    colors = 0.5 + 0.5 * verts / np.abs(verts).max()
    return (torch.tensor(verts)[None], torch.tensor(faces)[None],
            torch.tensor(colors, dtype=torch.float32)[None])


def loaded_pytorch3d():
    return sorted(name for name, module in sys.modules.items()
                  if module is not None and name.split(".")[0] == "pytorch3d")


@pytest.fixture
def darwin(monkeypatch):
    """macOS, with pytorch3d made unimportable for the duration of the test."""
    # Loaded BEFORE the platform is faked: a Linux build of monai cannot be
    # imported once `sys.platform` says darwin (sysconfig looks for a darwin file).
    import monai.networks.nets  # noqa: F401

    monkeypatch.setattr(sys, "platform", "darwin")
    for name in [n for n in sys.modules if n == "pytorch3d" or n.startswith("pytorch3d.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "pytorch3d", None)
    yield
    assert loaded_pytorch3d() == []


def test_the_dependency_check_passes_without_pytorch3d(darwin):
    net.check_dependencies()
    assert set(net._import_pytorch3d()) == {"Meshes", "TexturesVertex", "look_at_rotation"}


def test_a_missing_embreex_is_reported_by_the_dependency_check(darwin, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name.startswith("embreex"):
            raise ImportError("no embreex", name="embreex")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    with pytest.raises(render_native.ToolUnavailableError, match="embreex"):
        net.check_dependencies()
    with pytest.raises(render_native.ToolUnavailableError, match="embreex"):
        net.build("cpu")


def test_the_native_network_has_the_parameters_the_checkpoint_holds(darwin):
    """The checkpoint stores `model.module.*` weights (a monai UNet under a
    TimeDistributed) and the `ico_verts` buffer; the native network must offer
    exactly those names, since `load` copies them in by name."""
    import monai

    model = net.build("cpu")
    keys = set(model.state_dict())
    unet = monai.networks.nets.UNet(spatial_dims=2, in_channels=4, out_channels=net.OUT_CHANNELS,
                                    channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2)
    assert keys == {"ico_verts"} | {"model.module." + key for key in unet.state_dict()}
    assert torch.equal(model.ico_verts, torch.tensor(net._CAMERA_POSITIONS))


def test_the_forward_pass_renders_seven_views_in_one_pass_each(darwin):
    model = net.build("cpu").eval()
    verts, faces, colors = bumpy()
    with torch.no_grad():
        logits, views, pixel_to_face = model((verts, faces, colors))

    size = net.IMAGE_SIZE
    assert logits.shape == (1, 7, net.OUT_CHANNELS, size, size)
    assert views.shape == (1, 7, 4, size, size)
    assert pixel_to_face.shape == (1, 7, 1, size, size)
    hit = pixel_to_face[:, :, 0] >= 0
    assert bool(hit.any())
    # Image and fragments come out of the same rasterization, so they agree on
    # what was hit: depth is -1 exactly where the pixel map is -1, and the
    # background is the white pytorch3d's default BlendParams gave this network.
    assert bool(((views[:, :, 3] >= 0) == hit).all())
    assert bool((views[:, :, :3].permute(0, 1, 3, 4, 2)[~hit] == 1.0).all())


def test_each_view_is_aimed_by_its_own_pose(darwin):
    """The seven images differ and each matches a direct render from its position."""
    model = net.build("cpu").eval()
    verts, faces, colors = bumpy()
    with torch.no_grad():
        _, views, pixel_to_face = model((verts, faces, colors))
    mesh = render_native.Mesh(verts[0], faces[0], colors[0])
    for index in (0, 4):
        R, T = render_native.pose(net._CAMERA_POSITIONS[index])
        rgb, zbuf, p2f = render_native.render(mesh, R, T, render_native.AmbientLights(), 60.0, 1.0,
                                              net.IMAGE_SIZE, (1.0, 1.0, 1.0))
        assert torch.equal(pixel_to_face[0, index, 0], p2f)
        assert torch.allclose(views[0, index, 3], zbuf)
    assert not torch.equal(pixel_to_face[0, 0], pixel_to_face[0, 4])


def test_off_macos_the_pytorch3d_network_is_the_one_built(monkeypatch):
    """Linux and Windows never reach `render_native`."""
    pytest.importorskip("pytorch3d")
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setitem(sys.modules, "sadt_areg_ios.render_native", None)
    model = net.build("cpu")
    assert type(model.renderer).__module__.startswith("pytorch3d")
    assert "FoVPerspectiveCameras" in net._import_pytorch3d()


def test_a_fresh_interpreter_on_macos_never_loads_pytorch3d():
    """Only the third-party stack imported first, then the whole forward pass."""
    script = """
import sys
import torch, numpy, monai.networks.nets
sys.platform = "darwin"
from sadt_areg_ios import net
net.check_dependencies()
model = net.build("cpu").eval()
verts = torch.tensor([[-1., -1, 0.5], [1, -1, 0.5], [1, 1, 0.5], [-1, 1, 0.5]])[None]
faces = torch.tensor([[0, 1, 2], [0, 2, 3]])[None]
with torch.no_grad():
    logits, views, p2f = model((verts, faces, torch.ones(1, 4, 3)))
assert bool((p2f >= 0).any())
print("LOADED", sorted(name for name in sys.modules if name.split(".")[0] == "pytorch3d"))
"""
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    assert "LOADED []" in done.stdout
