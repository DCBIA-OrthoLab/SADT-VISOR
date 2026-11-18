"""`render_native`, the macOS renderer that replaces pytorch3d, tested on its own.

Four layers, from the one that needs nothing to the one that needs pytorch3d:

* the camera maths and the shading, on analytic cases (`torch` only);
* the rasterizer and the renderer on hand-built geometry (needs `embreex`,
  which is not a dependency of the Linux deployments: those tests skip);
* parity with images pytorch3d produced ONCE and that are stored in
  `data/render_native_reference.npz`, so the test needs neither pytorch3d nor
  a GPU. They come from a synthetic bumpy sphere, so no patient data is stored;
  the measurement on the public `T1_01_U_segmented.vtk` lives in the PR;
* a check that the stored images are still pytorch3d's: when pytorch3d is
  importable the reference is rebuilt live and compared to the file.
  `python test_render_native.py` rewrites the file (needs pytorch3d).

`render_native.py` is a byte-identical copy in ALI_IOS and AREG_IOS (the tools
share no package); so is this file, apart from the import below, and so is the
reference file. Change one, change both.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from sadt_ali_ios import render_native as native

DATA = Path(__file__).parent / "data" / "render_native_reference.npz"

# ---------------------------------------------------------------------------
# Test geometry
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


def bumpy_mesh(device="cpu"):
    verts, faces = icosphere()
    colors = 0.5 + 0.5 * verts / np.abs(verts).max()
    return native.Mesh(verts, faces, colors.astype(np.float32), device)


def quad(depth, colors=(0.2, 0.4, 0.8), half=1.0):
    """A square at z = `depth`, facing a camera at the origin looking down +z."""
    verts = np.array([[-half, -half, depth], [half, -half, depth], [half, half, depth], [-half, half, depth]],
                     np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    return native.Mesh(verts, faces, np.tile(np.array(colors, np.float32), (4, 1)))


IDENTITY = (torch.eye(3), torch.zeros(3))

# The two configurations the tools use, as `(name, fov, znear, size, lights, background)`.
ALI = dict(fov=90.0, znear=0.01, size=224, background=(0.0, 0.0, 0.0))
AREG = dict(fov=60.0, znear=1.0, size=320, background=(1.0, 1.0, 1.0))

# The views the reference file holds (`areg_inside_0` is the tool's first camera,
# inside the bumpy sphere): `(config, eye, at, up)`.
AREG_EYES = ((0.0, 0.0, 0.9), (0.5, 0.4, 2.2))
VIEWS = {
    "ali_outside": (ALI, (0.4, 0.5, 1.4), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    "ali_cervical": (ALI, (0.9, -0.9, -0.45), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    "ali_steep": (ALI, (0.02, 1.5, 0.01), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    "areg_inside_0": (AREG, AREG_EYES[0], (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    "areg_outside": (AREG, AREG_EYES[1], (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
}


def lights_of(config):
    return native.PointLights() if config is ALI else native.AmbientLights()


def render_view(mesh, name):
    config, eye, at, up = VIEWS[name]
    R, T = native.pose(eye, at, up)
    return native.render(
        mesh, R, T, lights_of(config), config["fov"], config["znear"], config["size"], config["background"]
    )


def test_the_platform_is_the_only_switch_and_nothing_here_reads_the_environment():
    """The callers choose this module with `sys.platform == "darwin"`, nothing
    else; a variable read in here would be a second, hidden switch."""
    import ast

    tree = ast.parse(Path(native.__file__).read_text())
    imported = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import)
                for alias in node.names}
    imported |= {node.module.split(".")[0] for node in ast.walk(tree)
                 if isinstance(node, ast.ImportFrom) and node.module}
    names = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    names |= {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert not imported & {"os", "pytorch3d", "threading", "sys"}
    assert not names & {"environ", "getenv"}


# ---------------------------------------------------------------------------
# Camera maths
# ---------------------------------------------------------------------------

def test_look_at_rotation_down_the_z_axis_is_the_identity_flipped_in_x():
    """A camera on +z looking at the origin with up = +y: view z = world -z, and
    pytorch3d's +X-to-the-left convention makes view x = world -x."""
    rotation = native.look_at_rotation([[0.0, 0.0, 5.0]])[0]
    expected = torch.diag(torch.tensor([-1.0, 1.0, -1.0]))
    assert torch.allclose(rotation, expected, atol=1e-6)


def test_look_at_rotation_is_a_rotation_for_random_cameras():
    generator = torch.Generator().manual_seed(0)
    eyes = torch.randn(64, 3, generator=generator) * 2
    rotations = native.look_at_rotation(eyes)
    assert rotations.shape == (64, 3, 3)
    assert torch.allclose(rotations.transpose(1, 2) @ rotations, torch.eye(3).expand(64, 3, 3), atol=1e-5)
    assert torch.allclose(torch.linalg.det(rotations), torch.ones(64), atol=1e-5)


def test_the_third_axis_points_from_the_eye_to_the_target():
    eye, at = torch.tensor([[1.0, 2.0, 3.0]]), torch.tensor([[0.5, -1.0, 0.0]])
    rotation = native.look_at_rotation(eye, at=at)[0]
    direction = torch.nn.functional.normalize(at - eye, dim=1)[0]
    assert torch.allclose(rotation[:, 2], direction, atol=1e-6)


def test_the_eye_lands_on_the_view_origin_whatever_the_pose():
    for eye in ((0.0, 0.0, 0.9), (0.3, -0.7, 1.2), (-1.0, 0.2, 0.1)):
        R, T = native.pose(eye)
        assert torch.allclose(torch.tensor(eye) @ R + T, torch.zeros(3), atol=1e-6)


def test_a_steep_camera_keeps_a_proper_rotation():
    """A camera almost straight above the target, `up` almost parallel to the view."""
    rotation = native.look_at_rotation([[0.02, 1.5, 0.01]])[0]
    assert torch.allclose(rotation.T @ rotation, torch.eye(3), atol=1e-5)
    assert float(torch.linalg.det(rotation)) == pytest.approx(1.0, abs=1e-5)


def test_an_up_vector_exactly_parallel_to_the_view_is_degenerate_like_pytorch3d_s():
    """pytorch3d's own formula has no valid answer there (its `isclose` branch
    rebuilds x from a y that is already zero), and its renderer then fails on a
    singular matrix. Reproduced, not repaired: the rotation is finite, only its
    x and y axes are zero, and no tool aims a camera exactly along `up`."""
    rotation = native.look_at_rotation([[0.0, 1.5, 0.0]])[0]
    assert torch.isfinite(rotation).all()
    assert torch.allclose(rotation[:, 2], torch.tensor([0.0, -1.0, 0.0]), atol=1e-6)
    assert float(rotation[:, :2].abs().max()) == 0.0


def test_at_and_up_broadcast_against_a_batch_of_positions():
    eyes = torch.tensor([[0.0, 0.0, 2.0], [2.0, 0.0, 0.0], [0.0, 2.0, 2.0]])
    rotations = native.look_at_rotation(eyes, at=[0.0, 0.0, 0.0], up=[0.0, 1.0, 0.0])
    assert rotations.shape == (3, 3, 3)
    for index in range(3):
        assert torch.allclose(rotations[index], native.look_at_rotation(eyes[index:index + 1])[0], atol=1e-6)


# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------

def test_vertex_normals_of_a_flat_square_are_the_face_normal():
    mesh = quad(2.0)
    normals = mesh.normals
    assert torch.allclose(normals.abs(), torch.tensor([[0.0, 0.0, 1.0]]).expand(4, 3), atol=1e-6)
    assert torch.allclose(normals, normals[0].expand(4, 3))


def test_vertex_normals_are_unit_length_and_area_weighted():
    # Two triangles share vertex 0; the bigger one (area 2) must dominate the smaller (area 0.5).
    verts = torch.tensor([[0.0, 0, 0], [2.0, 0, 0], [0, 2.0, 0], [0, 0, 1.0], [1.0, 0, 0]])
    faces = torch.tensor([[0, 1, 2], [0, 3, 4]])
    mesh = native.Mesh(verts, faces, torch.zeros(5, 3))
    assert torch.allclose(mesh.normals.norm(dim=1), torch.ones(5), atol=1e-6)
    big = torch.tensor([0.0, 0.0, 1.0])
    small = torch.nn.functional.normalize(
        torch.cross(verts[4] - verts[3], verts[0] - verts[3], dim=0), dim=0)
    expected = torch.nn.functional.normalize(2.0 * big + 0.5 * small, dim=0)
    assert torch.allclose(mesh.normals[0].abs(), expected.abs(), atol=1e-5)


@pytest.mark.parametrize("bad", ["faces_out_of_range", "negative_face", "wrong_vertex_shape", "wrong_color_shape"])
def test_a_malformed_mesh_is_refused_before_it_reaches_embree(bad):
    verts, faces, colors = torch.zeros(4, 3), torch.tensor([[0, 1, 2]]), torch.zeros(4, 3)
    if bad == "faces_out_of_range":
        faces = torch.tensor([[0, 1, 9]])
    elif bad == "negative_face":
        faces = torch.tensor([[0, 1, -1]])
    elif bad == "wrong_vertex_shape":
        verts = torch.zeros(4, 2)
    else:
        colors = torch.zeros(3, 3)
    with pytest.raises(ValueError):
        native.Mesh(verts, faces, colors)


def test_the_meshes_facade_takes_a_batch_of_one_and_nothing_else():
    verts, faces = torch.zeros(1, 4, 3), torch.tensor([[[0, 1, 2]]])
    mesh = native.Meshes(verts=verts, faces=faces, textures=native.TexturesVertex(verts_features=torch.zeros(1, 4, 3)))
    assert len(mesh.v) == 4
    with pytest.raises(ValueError, match="batch of one"):
        native.Meshes(verts=torch.zeros(2, 4, 3), faces=torch.zeros(2, 1, 3, dtype=torch.int64),
                      textures=native.TexturesVertex(verts_features=torch.zeros(2, 4, 3)))


# ---------------------------------------------------------------------------
# Shading (no rays: fragments are supplied)
# ---------------------------------------------------------------------------

def one_pixel(mesh_depth=2.0, bary=(1 / 3, 1 / 3, 1 / 3)):
    mesh = quad(mesh_depth, colors=(0.2, 0.4, 0.8))
    p2f = torch.zeros(1, 1, dtype=torch.int64)
    return mesh, p2f, torch.tensor(bary, dtype=torch.float32).reshape(1, 1, 3)


def test_ambient_lights_give_the_interpolated_texture_unchanged():
    mesh, p2f, bary = one_pixel()
    texels = native.interpolate(mesh, mesh.c, p2f, bary)
    assert torch.allclose(texels[0, 0], torch.tensor([0.2, 0.4, 0.8]), atol=1e-6)
    out = native.AmbientLights().shade(mesh, p2f, bary, texels, *IDENTITY)
    assert torch.equal(out, texels)


def test_point_lights_follow_the_phong_formula_on_a_flat_quad():
    """Quad at z = 2 facing -z, light at (0, 1, 0), camera at the origin. The
    pixel's point, normal and the vectors to the light and the camera are
    computed here in numpy, independently of the module."""
    mesh, p2f, bary = one_pixel(2.0, bary=(0.5, 0.25, 0.25))
    texels = native.interpolate(mesh, mesh.c, p2f, bary)
    lights = native.PointLights()
    got = lights.shade(mesh, p2f, bary, texels, *IDENTITY)[0, 0].numpy()

    corners = mesh.v[mesh.f[0]].numpy()
    point = (np.array([0.5, 0.25, 0.25])[:, None] * corners).sum(0)
    normal = np.array([0.0, 0.0, 1.0]) * np.sign(np.cross(corners[2] - corners[1], corners[0] - corners[1])[2])
    to_light = np.array([0.0, 1.0, 0.0]) - point
    to_light /= np.linalg.norm(to_light)
    cosine = float(normal @ to_light)
    to_camera = -point / np.linalg.norm(point)
    reflected = -to_light + 2 * cosine * normal
    specular = 0.2 * max(float(to_camera @ reflected), 0.0) ** 64 * (cosine > 0)
    expected = (0.5 + 0.3 * max(cosine, 0.0)) * np.array([0.2, 0.4, 0.8]) + specular
    assert got == pytest.approx(expected, abs=1e-5)


def test_the_specular_term_reads_the_pose_it_is_given():
    """The camera centre comes from the R and T of the call. Looking at a mirror
    configuration from the reflection direction lights the highlight; from the
    opposite side it does not. (pytorch3d reads it from state stored in a camera
    object, which is what this API avoids.)"""
    mesh, p2f, bary = one_pixel(2.0)
    # Wound the other way, so the normal (+z for `quad`) points back at the camera.
    mesh = native.Mesh(mesh.v, mesh.f[:, [0, 2, 1]], mesh.c)
    texels = native.interpolate(mesh, mesh.c, p2f, bary)
    lights = native.PointLights(ambient=0.0, diffuse=0.0, specular=1.0, location=(0.0, 0.0, 0.0),
                                 shininess=1.0)
    # Light and camera both at the origin: the reflection comes straight back at it.
    near = lights.shade(mesh, p2f, bary, texels, *IDENTITY)[0, 0]
    far_R, far_T = native.pose((9.0, 0.0, 0.0), at=(0.0, 0.0, 2.0))
    far = lights.shade(mesh, p2f, bary, texels, far_R, far_T)[0, 0]
    assert float(near.max()) > 0.8
    assert float(far.max()) < 0.5 * float(near.max())


def test_a_surface_facing_away_from_the_light_gets_ambient_light_only():
    mesh, p2f, bary = one_pixel(2.0)
    texels = native.interpolate(mesh, mesh.c, p2f, bary)
    behind = native.PointLights(location=(0.0, 0.0, 9.0))  # behind the quad, from the camera
    got = behind.shade(mesh, p2f, bary, texels, *IDENTITY)[0, 0]
    # The quad's normal may be +z or -z by winding; one of the two lights has cos < 0.
    other = native.PointLights(location=(0.0, 0.0, -9.0)).shade(mesh, p2f, bary, texels, *IDENTITY)[0, 0]
    ambient_only = 0.5 * texels[0, 0]
    assert torch.allclose(got, ambient_only, atol=1e-5) or torch.allclose(other, ambient_only, atol=1e-5)


# ---------------------------------------------------------------------------
# Rasterization (needs embreex)
# ---------------------------------------------------------------------------

@pytest.fixture
def embree():
    pytest.importorskip("embreex")


def test_a_missing_embreex_says_how_to_install_it(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name.startswith("embreex"):
            raise ImportError("no embreex", name="embreex")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    with pytest.raises(native.ToolUnavailableError, match="uv pip install embreex"):
        native.require_embree()
    with pytest.raises(native.ToolUnavailableError):
        native.namespace()


def test_a_quad_in_front_of_the_camera_fills_its_footprint_with_the_right_depth(embree):
    p2f, zbuf, bary = native.rasterize(quad(2.0), *IDENTITY, fov=90.0, znear=0.01, size=64)
    # fov 90 at depth 2 sees x, y in [-2, 2]: a half-size-1 quad covers the central half.
    assert int(p2f[32, 32]) >= 0
    assert float(zbuf[32, 32]) == pytest.approx(2.0, abs=1e-5)
    assert int(p2f[0, 0]) == -1 and float(zbuf[0, 0]) == -1.0
    inside = p2f >= 0
    assert 0.2 < float(inside.float().mean()) < 0.3  # (1/2)^2 = 0.25 of the image
    assert torch.allclose(bary[inside].sum(-1), torch.ones(int(inside.sum())), atol=1e-5)
    assert torch.allclose(zbuf[inside], torch.full((int(inside.sum()),), 2.0), atol=1e-4)


def test_plus_x_in_view_space_is_on_the_left_of_the_image(embree):
    """pytorch3d's convention: a quad on the camera's +x side lands on the LEFT columns."""
    verts = np.array([[0.2, -0.5, 2], [1.0, -0.5, 2], [1.0, 0.5, 2], [0.2, 0.5, 2]], np.float32)
    mesh = native.Mesh(verts, np.array([[0, 1, 2], [0, 2, 3]]), np.zeros((4, 3), np.float32))
    p2f, _, _ = native.rasterize(mesh, *IDENTITY, fov=90.0, znear=0.01, size=32)
    columns = torch.nonzero((p2f >= 0).any(0))[:, 0]
    assert int(columns.max()) < 16


def test_geometry_between_the_camera_and_the_half_near_plane_is_cut(embree):
    """pytorch3d clips at z = znear / 2: with znear 1 a quad at 0.4 is gone and one at 0.6 is seen."""
    cut = native.rasterize(quad(0.4, half=5.0), *IDENTITY, fov=60.0, znear=1.0, size=16)
    seen = native.rasterize(quad(0.6, half=5.0), *IDENTITY, fov=60.0, znear=1.0, size=16)
    assert bool((cut[0] == -1).all()) and bool((cut[1] == -1).all())
    assert bool((seen[0] >= 0).all())
    assert float(seen[1][8, 8]) == pytest.approx(0.6, abs=1e-5)


def test_faces_are_not_culled_so_a_back_face_is_hit(embree):
    mesh = quad(2.0)
    flipped = native.Mesh(mesh.v, mesh.f[:, [0, 2, 1]], mesh.c)
    for candidate in (mesh, flipped):
        p2f, _, _ = native.rasterize(candidate, *IDENTITY, fov=90.0, znear=0.01, size=16)
        assert int(p2f[8, 8]) >= 0


def test_a_mesh_with_no_faces_renders_as_background_only(embree):
    mesh = native.Mesh(torch.zeros(3, 3), torch.zeros(0, 3, dtype=torch.int64), torch.zeros(3, 3))
    rgb, zbuf, p2f = native.render(mesh, *IDENTITY, native.AmbientLights(), 60.0, 1.0, 8, (0.25, 0.5, 0.75))
    assert bool((p2f == -1).all()) and bool((zbuf == -1).all())
    assert torch.allclose(rgb, torch.tensor([0.25, 0.5, 0.75]).expand(8, 8, 3))


def test_a_camera_looking_away_from_the_mesh_sees_only_the_background(embree):
    R, T = native.pose((0.0, 0.0, -3.0), at=(0.0, 0.0, -9.0))
    rgb, _, p2f = native.render(quad(2.0), R, T, native.PointLights(), 60.0, 1.0, 8, (0.0, 0.0, 0.0))
    assert bool((p2f == -1).all()) and float(rgb.abs().max()) == 0.0


def test_a_camera_on_the_target_does_not_crash_the_renderer(embree):
    """eye == at has no viewing direction; pytorch3d returns a degenerate rotation
    too. What is pinned: no exception, finite output, a fixed-size image."""
    R, T = native.pose((0.0, 0.0, 2.0), at=(0.0, 0.0, 2.0))
    rgb, zbuf, p2f = native.render(quad(2.0), R, T, native.PointLights(), 60.0, 1.0, 8)
    assert rgb.shape == (8, 8, 3) and torch.isfinite(rgb).all() and torch.isfinite(zbuf).all()
    assert p2f.shape == (8, 8)


def test_the_renderer_returns_rgba_and_fragments_shaped_like_pytorch3d(embree):
    renderer = native.Renderer(native.AmbientLights(), fov=90, znear=0.01, size=32, background=(1.0, 1.0, 1.0))
    R, T = native.pose((0.0, 0.0, 0.0), at=(0.0, 0.0, 1.0))
    rgba, fragments = renderer(meshes_world=quad(2.0), R=R[None], T=T[None])
    assert rgba.shape == (1, 32, 32, 4)
    assert fragments.pix_to_face.shape == (1, 32, 32, 1) and fragments.zbuf.shape == (1, 32, 32, 1)
    hit = fragments.pix_to_face[0, ..., 0] >= 0
    assert bool((rgba[0, ..., 3] == hit.float()).all())
    assert torch.allclose(rgba[0][hit][:, :3], torch.tensor([0.2, 0.4, 0.8]).expand(int(hit.sum()), 3), atol=1e-6)
    assert torch.allclose(rgba[0][~hit][:, :3], torch.ones(int((~hit).sum()), 3))
    assert renderer.to("cpu") is renderer


def test_the_scene_is_built_once_per_mesh(embree):
    mesh = bumpy_mesh()
    first = mesh.scene
    render_view(mesh, "ali_outside")
    render_view(mesh, "areg_inside_0")
    assert mesh.scene is first
    assert mesh.clone().scene is first


# ---------------------------------------------------------------------------
# Parity with pytorch3d, from the stored images
# ---------------------------------------------------------------------------

# Only a WINDOW of each image is stored, placed on the silhouette so that it holds
# surface, background and edge pixels; the pixel grid is the full image's, so a
# half-pixel shift anywhere would still show.
WINDOW = 112


def load_reference():
    if not DATA.exists():
        pytest.skip("tests/data/render_native_reference.npz is missing")
    return np.load(DATA)


def compare(got, name, reference):
    rgb, zbuf, p2f = (t.cpu().numpy() for t in got)
    r0, c0 = (int(v) for v in reference[name + "_origin"])
    window = (slice(r0, r0 + WINDOW), slice(c0, c0 + WINDOW))
    rgb, zbuf, p2f = rgb[window], zbuf[window], p2f[window]
    ref_p2f = reference[name + "_p2f"]
    ref_rgb = reference[name + "_rgb"].astype(np.float32) / 65535.0
    ref_z = reference[name + "_z"].astype(np.float32) / 65535.0 * 8.0 - 2.0
    same = ref_p2f == p2f
    rgb_close = np.abs(ref_rgb - rgb).max(-1) < 1e-3
    return {
        "same_face": float(same.mean()),
        "rgb_close": float(rgb_close.mean()),
        "rgb_close_where_same": float(rgb_close[same].mean()),
        "z_error_where_same": float(np.abs(ref_z - zbuf)[same & (ref_p2f >= 0)].max()),
        # A pixel on which only one of the two saw the mesh would be a real defect.
        "background_vs_mesh": int(((ref_p2f < 0) != (p2f < 0)).sum()),
        "hit": float((ref_p2f >= 0).mean()),
    }


@pytest.mark.parametrize("name", sorted(VIEWS))
def test_the_images_match_pytorch3d_to_the_edge_pixels(embree, name):
    reference = load_reference()
    got = render_view(bumpy_mesh(), name)
    scores = compare(got, name, reference)
    # The inside view sees surface in every pixel (and crosses the near-plane cut);
    # every other window must hold both the mesh and the background.
    assert scores["hit"] > 0.2
    assert name.endswith("inside_0") or scores["hit"] < 0.95
    assert scores["same_face"] > 0.998
    assert scores["rgb_close"] > 0.998
    assert scores["rgb_close_where_same"] > 0.9999
    assert scores["z_error_where_same"] < 5e-4
    assert scores["background_vs_mesh"] <= 3


# ---------------------------------------------------------------------------
# The stored images are still pytorch3d's
# ---------------------------------------------------------------------------

def pytorch3d_reference():
    """Every view of `VIEWS` rendered by pytorch3d, on the CPU rasterizer."""
    from pytorch3d.renderer import (
        AmbientLights, FoVPerspectiveCameras, HardPhongShader, MeshRasterizer,
        MeshRendererWithFragments, PointLights, RasterizationSettings, TexturesVertex, blending,
        look_at_rotation,
    )
    from pytorch3d.structures import Meshes

    verts, faces = icosphere()
    colors = 0.5 + 0.5 * verts / np.abs(verts).max()
    meshes = Meshes(verts=[torch.tensor(verts)], faces=[torch.tensor(faces)],
                    textures=TexturesVertex(verts_features=[torch.tensor(colors, dtype=torch.float32)]))
    out = {}
    for name, (config, eye, at, up) in VIEWS.items():
        cameras = FoVPerspectiveCameras(fov=config["fov"], znear=config["znear"], zfar=100.0)
        settings = RasterizationSettings(image_size=config["size"], blur_radius=0, faces_per_pixel=1,
                                         perspective_correct=True)
        lights = PointLights() if config is ALI else AmbientLights()
        renderer = MeshRendererWithFragments(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=settings),
            shader=HardPhongShader(cameras=cameras, lights=lights,
                                   blend_params=blending.BlendParams(background_color=config["background"])),
        )
        position = torch.tensor([eye], dtype=torch.float32)
        rotation = look_at_rotation(position, at=torch.tensor([at]), up=torch.tensor([up]))
        translation = -torch.bmm(rotation.transpose(1, 2), position[:, :, None])[:, :, 0]
        image, fragments = renderer(meshes_world=meshes, R=rotation, T=translation)
        p2f = fragments.pix_to_face[0, ..., 0].numpy().astype(np.int32)
        rgb = np.round(image[0, ..., :3].clamp(0, 1).numpy() * 65535).astype(np.uint16)
        zbuf = np.round((fragments.zbuf[0, ..., 0].numpy() + 2.0) / 8.0 * 65535).astype(np.uint16)
        rows, columns = np.nonzero(p2f >= 0)
        size = config["size"]
        r0 = int(np.clip(rows.min() - 24, 0, size - WINDOW))  # the silhouette's top left, with margin
        c0 = int(np.clip(columns.min() - 24, 0, size - WINDOW))
        window = (slice(r0, r0 + WINDOW), slice(c0, c0 + WINDOW))
        out[name + "_origin"] = np.array([r0, c0])
        out[name + "_p2f"], out[name + "_rgb"], out[name + "_z"] = p2f[window], rgb[window], zbuf[window]
    return out


def test_the_stored_images_are_what_pytorch3d_renders_now():
    pytest.importorskip("pytorch3d")
    reference = load_reference()
    live = pytorch3d_reference()
    for key, value in live.items():
        if key.endswith("_origin"):
            assert (reference[key] == value).all(), key
        elif key.endswith("_p2f"):
            assert (reference[key] == value).mean() > 0.9999, key
        else:
            assert np.abs(reference[key].astype(np.int64) - value.astype(np.int64)).max() <= 66, key


if __name__ == "__main__":
    np.savez_compressed(DATA, **pytorch3d_reference())
    print("wrote", DATA, DATA.stat().st_size, "bytes")
