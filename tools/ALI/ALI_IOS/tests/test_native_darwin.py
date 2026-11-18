"""On macOS, ALI_IOS renders without importing pytorch3d, and only there.

The platform is simulated the way `test_embree_raster.py` does it, by setting
`sys.platform`. pytorch3d is not merely absent from the checks: `sys.modules`
holds `None` under its name while the tests run, so ANY attempt to import it
raises, and the tests then assert that nothing named `pytorch3d` was loaded.
The fresh-interpreter test at the end is the same claim with nothing else
imported before it.

The renderer needs `embreex` (not a dependency of the Linux deployments), so
the tests that render skip without it.
"""

import subprocess
import sys

import numpy as np
import pytest
import torch

from sadt_ali_ios import render, render_native

pytest.importorskip("embreex")


def bumpy(device="cpu"):
    from test_render_native import icosphere

    verts, faces = icosphere(subdivisions=3)
    colors = 0.5 + 0.5 * verts / np.abs(verts).max()
    return (torch.tensor(verts)[None].to(device), torch.tensor(faces)[None].to(device),
            torch.tensor(colors, dtype=torch.float32)[None].to(device))


def loaded_pytorch3d():
    return sorted(name for name, module in sys.modules.items()
                  if module is not None and name.split(".")[0] == "pytorch3d")


@pytest.fixture
def darwin(monkeypatch):
    """macOS, with pytorch3d made unimportable for the duration of the test."""
    # Loaded BEFORE the platform is faked: a Linux build of monai cannot be
    # imported once `sys.platform` says darwin (sysconfig looks for a darwin file).
    import monai.networks.nets  # noqa: F401
    import vtk  # noqa: F401

    monkeypatch.setattr(sys, "platform", "darwin")
    for name in [n for n in sys.modules if n == "pytorch3d" or n.startswith("pytorch3d.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "pytorch3d", None)
    yield
    assert loaded_pytorch3d() == []


def test_the_names_the_engine_takes_from_pytorch3d_come_from_render_native(darwin):
    p3d = render.import_pytorch3d()
    assert set(p3d) == {"Meshes", "TexturesVertex", "look_at_rotation"}
    assert p3d["look_at_rotation"] is render_native.look_at_rotation


def test_the_engine_dependency_check_passes_without_pytorch3d(darwin):
    from sadt_ali_ios import engine

    engine.check_dependencies()


def test_a_missing_embreex_is_reported_by_the_dependency_check(darwin, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name.startswith("embreex"):
            raise ImportError("no embreex", name="embreex")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    with pytest.raises(render_native.ToolUnavailableError, match="embreex"):
        render.import_pytorch3d()
    with pytest.raises(render_native.ToolUnavailableError, match="embreex"):
        render.build_renderer("cpu")


def test_the_trained_rasterisation_settings_are_the_only_ones_offered(darwin):
    with pytest.raises(ValueError, match="one face per pixel"):
        render.build_renderer("cpu", blur_radius=0.1)
    with pytest.raises(ValueError, match="one face per pixel"):
        render.build_renderer("cpu", faces_per_pixel=3)


def test_the_crown_views_are_rendered_by_the_real_render_views(darwin):
    p3d = render.import_pytorch3d()
    verts, faces, colors = bumpy()
    mesh = p3d["Meshes"](verts=verts, faces=faces,
                         textures=p3d["TexturesVertex"](verts_features=colors)).to("cpu")
    renderer = render.build_renderer("cpu")
    center = torch.zeros(1, 3)
    positions = render.CAMERA_POSITIONS["O"]["Upper"]

    images, pix_to_face = render.render_views(renderer, mesh, center, 1.4, positions, "cpu")

    assert images.shape == (1, len(positions), 4, 224, 224)
    assert pix_to_face.shape == (len(positions), 1, 224, 224, 1)
    assert torch.isfinite(images).all()
    hit = pix_to_face[:, 0, :, :, 0] >= 0
    assert 0.1 < float(hit.float().mean()) < 1.0
    # Black background (what the network was trained on), and depth -1 off the mesh.
    assert float(images[0, :, :3].permute(0, 2, 3, 1)[~hit].abs().max()) == 0.0
    assert bool((images[0, :, 3][~hit] == -1).all())
    assert bool((images[0, :, 3][hit] > 0).all())


def test_the_mucogingival_views_are_rendered_by_the_real_render_mg_views(darwin):
    p3d = render.import_pytorch3d()
    verts, faces, colors = bumpy()
    mesh = p3d["Meshes"](verts=verts, faces=faces,
                         textures=p3d["TexturesVertex"](verts_features=colors)).to("cpu")
    renderer = render.build_renderer("cpu")
    aim = torch.zeros(3)
    directions = render.mg_camera_directions(torch.tensor([0.0, 1.0, 0.0]), "cpu")

    images, pix_to_face = render.render_mg_views(renderer, mesh, aim, directions, 1.4, "cpu")

    assert images.shape == (1, 3, 4, 224, 224)
    assert pix_to_face.shape == (3, 1, 224, 224, 1)
    assert bool((pix_to_face >= 0).any())


def test_off_macos_the_pytorch3d_path_is_the_one_taken(monkeypatch):
    """Linux and Windows never reach `render_native`: the renderer is pytorch3d's."""
    pytest.importorskip("pytorch3d")
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setitem(sys.modules, "sadt_ali_ios.render_native", None)
    assert "FoVPerspectiveCameras" in render.import_pytorch3d()
    renderer = render.build_renderer("cpu")
    assert type(renderer).__module__.startswith("pytorch3d")


def test_a_fresh_interpreter_on_macos_never_loads_pytorch3d():
    """Only the third-party stack imported first, then the whole inference path."""
    script = """
import sys
import torch, numpy, vtk, monai.networks.nets
sys.platform = "darwin"
from sadt_ali_ios import engine, render
engine.check_dependencies()
p3d = render.import_pytorch3d()
verts = torch.tensor([[-1., -1, 2], [1, -1, 2], [1, 1, 2], [-1, 1, 2]])[None]
faces = torch.tensor([[0, 1, 2], [0, 2, 3]])[None]
mesh = p3d["Meshes"](verts=verts, faces=faces, textures=p3d["TexturesVertex"](verts_features=torch.ones(1, 4, 3)))
images, p2f = render.render_views(render.build_renderer("cpu"), mesh, torch.zeros(1, 3), 0.1, [[0, 0, -1]], "cpu")
assert bool((p2f >= 0).any())
loaded = sorted(name for name in sys.modules if name.split(".")[0] == "pytorch3d")
print("LOADED", loaded)
"""
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    assert "LOADED []" in done.stdout
