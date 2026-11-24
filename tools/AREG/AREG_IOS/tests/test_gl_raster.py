"""The OpenGL rasterizer against pytorch3d's own, on a synthetic mesh.

The reference is pytorch3d on the CPU, which every Linux build has. Each check
runs twice: through OpenGL, which skips where no OpenGL context can be made
(no `moderngl`, which is a macOS-only dependency, or no GPU), and through the
PyTorch fallback, which runs wherever torch does. A run without pytorch3d is
green and reports a skip rather than a failure.

`gl_raster.py` is a byte-identical copy in ALI_IOS, AREG_IOS, Crown_Seg and
DOCShapeAXI (the tools share no package); so is this file, apart from the
import below. Change one, change all four.
"""

import sys

import numpy as np
import pytest
from sadt_areg_ios import gl_raster

# ---------------------------------------------------------------------------
# The switch: nothing changes unless it is asked for
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("platform", ["linux", "win32"])
@pytest.mark.parametrize("device", ["cpu", "cuda", "cuda:0", "mps", None])
def test_off_macos_the_default_is_pytorch3d_whatever_the_device(platform, device, monkeypatch):
    monkeypatch.setattr(sys, "platform", platform)
    assert gl_raster.wanted(device) is False
    assert gl_raster.wanted() is False


@pytest.mark.parametrize("device", ["cpu", "mps", None])
def test_on_macos_opengl_is_used_whatever_the_device(device, monkeypatch):
    """Crown_Seg and AREG_IOS never name `mps`: the platform decides."""
    monkeypatch.setattr(sys, "platform", "darwin")
    assert gl_raster.wanted(device) is True
    assert gl_raster.wanted() is True


def test_without_opengl_the_same_faces_come_from_pytorch(monkeypatch):
    """No GPU, no OpenGL, a future macOS without it: an answer, not an error."""
    torch = pytest.importorskip("torch")
    if gl_raster._gl() is None:
        pytest.skip("no OpenGL context here to compare the fallback with")
    verts, faces = icosphere(subdivisions=3)
    verts, faces = torch.tensor(verts), torch.tensor(faces)
    rotation, translation = torch.eye(3), torch.tensor([0.0, 0.0, 3.0])
    with_gl = gl_raster.rasterize(verts, faces, rotation, translation, (1.7, 1.7, 0.0, 0.0), 0.5, 64)
    monkeypatch.setattr(gl_raster, "_gl", lambda: None)
    without = gl_raster.rasterize(verts, faces, rotation, translation, (1.7, 1.7, 0.0, 0.0), 0.5, 64)
    assert float((with_gl[0] == without[0]).float().mean()) > 0.995


# ---------------------------------------------------------------------------
# Parity with pytorch3d
# ---------------------------------------------------------------------------

def icosphere(subdivisions=4, seed=0, amplitude=0.2):
    """A bumpy unit sphere: no patient data, and not convex."""
    t = (1 + 5 ** 0.5) / 2
    verts = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t],
                      [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    faces = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4],
                      [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8],
                      [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]])
    for _ in range(subdivisions):
        n = len(verts)
        edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), 1)
        unique, inverse = np.unique(edges[:, 0] * n + edges[:, 1], return_inverse=True)
        mid = (verts[unique // n] + verts[unique % n]) / 2
        mid /= np.linalg.norm(mid, axis=1, keepdims=True)
        m = inverse.reshape(3, -1) + n
        verts = np.concatenate([verts, mid])
        faces = np.concatenate([
            np.stack([faces[:, 0], m[0], m[2]], 1), np.stack([faces[:, 1], m[1], m[0]], 1),
            np.stack([faces[:, 2], m[2], m[1]], 1), np.stack([m[0], m[1], m[2]], 1)])
    rng = np.random.default_rng(seed)
    k, phase, freq = rng.normal(size=(6, 3)), rng.uniform(0, 6.28, 6), rng.uniform(2, 6, 6)
    verts = verts * (1 + sum(amplitude / 6 * np.sin(freq[i] * (verts @ k[i]) + phase[i]) for i in range(6)))[:, None]
    verts = (verts - verts.mean(0)) / np.linalg.norm(verts - verts.mean(0), axis=1).max()
    return verts.astype(np.float32), faces.astype(np.int64)


@pytest.fixture(scope="module", params=["opengl", "pytorch"])
def parity(request):
    torch = pytest.importorskip("torch")
    renderer = pytest.importorskip("pytorch3d.renderer")
    structures = pytest.importorskip("pytorch3d.structures")
    patch = pytest.MonkeyPatch()
    if request.param == "opengl":
        if gl_raster._gl() is None:
            pytest.skip("no OpenGL context here (moderngl is a macOS dependency)")
    else:
        patch.setattr(gl_raster, "_gl", lambda: None)
    request.addfinalizer(patch.undo)

    verts, faces = icosphere()
    size = 96
    cameras = renderer.FoVPerspectiveCameras(fov=60, znear=1.0, zfar=100.0)
    settings = renderer.RasterizationSettings(image_size=size, blur_radius=0, faces_per_pixel=1,
                                              perspective_correct=True)
    reference = renderer.MeshRasterizer(cameras=cameras, raster_settings=settings)
    candidate = gl_raster.rasterizer_class()(cameras=cameras, raster_settings=settings)
    mesh = structures.Meshes(verts=[torch.tensor(verts)], faces=[torch.tensor(faces)])

    torch.manual_seed(0)
    directions = torch.nn.functional.normalize(torch.randn(6, 3), dim=1)
    pairs = []
    # 0.9 from the centre puts the surface across the znear/2 clip plane.
    for distance in (0.9, 1.35):
        for position in directions * distance:
            position = position[None]
            rotation = renderer.look_at_rotation(position)
            translation = -torch.bmm(rotation.transpose(1, 2), position[:, :, None])[:, :, 0]
            pairs.append((reference(mesh, R=rotation, T=translation),
                          candidate(mesh, R=rotation, T=translation)))
    return pairs


def test_pix_to_face_is_the_same_nearly_everywhere(parity):
    same = [float((a.pix_to_face == b.pix_to_face).float().mean()) for a, b in parity]
    assert np.mean(same) > 0.999
    assert min(same) > 0.995


def test_zbuf_and_barycentrics_agree_where_the_face_agrees(parity):
    for reference, candidate in parity:
        both = (reference.pix_to_face == candidate.pix_to_face) & (reference.pix_to_face >= 0)
        assert float((reference.zbuf - candidate.zbuf).abs()[both].max()) < 1e-4
        bary = (reference.bary_coords - candidate.bary_coords).abs()
        assert float(bary[both.unsqueeze(-1).expand_as(bary)].max()) < 1e-3


def test_a_miss_is_minus_one_everywhere(parity):
    assert any(bool((candidate.pix_to_face < 0).any()) for _, candidate in parity)
    for _, candidate in parity:
        miss = candidate.pix_to_face < 0
        assert bool((candidate.zbuf[miss] == -1).all())
        assert bool((candidate.bary_coords[miss.unsqueeze(-1).expand_as(candidate.bary_coords)] == -1).all())


def test_unsupported_settings_are_refused_not_approximated():
    renderer = pytest.importorskip("pytorch3d.renderer")
    settings = renderer.RasterizationSettings(image_size=8, blur_radius=0.01, faces_per_pixel=3)
    with pytest.raises(NotImplementedError, match="blur_radius"):
        gl_raster._check_settings(settings, True)


def test_the_mps_background_blend_matches_pytorch3d_on_the_cpu():
    """The replacement is only used on MPS; here it is called directly, with CPU
    tensors, to pin that it computes what `hard_rgb_blend` computes."""
    torch = pytest.importorskip("torch")
    blending = pytest.importorskip("pytorch3d.renderer.blending")
    fragments_module = pytest.importorskip("pytorch3d.renderer.mesh.rasterizer")

    torch.manual_seed(0)
    colors = torch.rand(1, 16, 16, 1, 3) - 0.5
    faces = (torch.rand(1, 16, 16, 1) * 6 - 3).long()
    fragments = fragments_module.Fragments(
        pix_to_face=faces, zbuf=torch.zeros(1, 16, 16, 1), dists=torch.zeros(1, 16, 16, 1),
        bary_coords=torch.zeros(1, 16, 16, 1, 3),
    )
    params = blending.BlendParams(background_color=(0.0, 0.0, 0.0))
    expected = blending.hard_rgb_blend(colors, fragments, params)

    replaced = gl_raster._hard_rgb_blend_for_mps(blending.hard_rgb_blend)
    # On the CPU the wrapper defers to the original, which is what keeps every
    # other device on pytorch3d's own code.
    assert torch.equal(replaced(colors, fragments, params), expected)
    assert replaced._sadt_wrapped
