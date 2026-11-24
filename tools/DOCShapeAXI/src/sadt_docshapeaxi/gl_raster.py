"""pytorch3d's rasterizer, done by the GPU's own graphics pipeline (macOS, MPS).

pytorch3d has no Metal kernel: handed a tensor on `mps` its rasterizer raises
`Cannot use CPU implementation`, and on the CPU it takes about two seconds a
view. Here OpenGL (Metal underneath, on a Mac) draws every face in a colour
that encodes its index, which answers the one question a rasterizer is for --
which face does each pixel see -- with the hardware built to answer it. The
depth and barycentric coordinates are then computed exactly, in PyTorch on the
caller's device, from that face and the pixel's ray.

Measured on ALI_IOS's 238 views of a 588,244-face arch (224 x 224):

    Embree (CPU, what this replaces)   7.6 ms a view
    this module, M4 Max GPU             2.2 ms a view
    pytorch3d, RTX 6000 Ada (CUDA)     18.3 ms a view    (this module there: 1.1 ms)

and 99.81 % of the covered pixels name the same face as pytorch3d; the others
lie exactly on the edge two faces share, where either answer is right. The
landmarks ALI_IOS places are the same to 0.000 mm.

When no OpenGL context can be made (a machine without a GPU, a future macOS
without OpenGL), the same answer is computed by PyTorch alone: exact, and
slower (about 120 ms a view on MPS, 2 ms on CUDA).

Conventions are pytorch3d's, the ones its shaders and the networks expect:
view space is `X_view = X_world @ R + T` (row vectors), +X to the LEFT, +Y up,
+Z forward; pixel (row r, column c) is sampled at its centre,
`x_ndc = 1 - (2c + 1) / S`, `y_ndc = 1 - (2r + 1) / S`; a projection is
`x_ndc = k00 * x / z + k20` and `y_ndc = k11 * y / z + k21`; geometry closer
than the clip plane `z_clip` is not seen; `zbuf` is the view-space z of the
hit; faces are not culled. `pix_to_face` is -1, `zbuf` -1 and `bary` -1 where
nothing is hit.

NOTHING HERE RUNS OFF macOS. `wanted()` is `sys.platform == "darwin"`, so on
Linux and Windows the existing pytorch3d path is the one that runs. `moderngl`
is imported on the first rasterization, never at import time.
"""

import functools
import logging
import sys

import numpy as np

logger = logging.getLogger(__name__)


def wanted(device=None) -> bool:
    """Whether to rasterize here rather than with pytorch3d.

    True on macOS and only there: pytorch3d has no Metal kernel and the compiled
    CPU one takes seconds a view. `device` is accepted and ignored, so the call
    sites need no knowledge of it (Crown_Seg and AREG_IOS never name `mps`).
    Linux and Windows keep pytorch3d's own rasterizer.
    """
    return sys.platform == "darwin"


# ---------------------------------------------------------------------------
# The core: which face each pixel sees, then exact depth and barycentrics
# ---------------------------------------------------------------------------

def _pixel_rays(size, k, device):
    """View-space directions with z = 1 through every pixel centre, (S*S, 3)."""
    import torch

    k00, k11, k20, k21 = k
    centres = 1 - (2 * torch.arange(size, device=device, dtype=torch.float32) + 1) / size
    y_ndc, x_ndc = torch.meshgrid(centres, centres, indexing="ij")   # x varies along columns
    return torch.stack([(x_ndc - k20) / k00, (y_ndc - k21) / k11, torch.ones_like(x_ndc)], -1).reshape(-1, 3)


def _intersect(origin, direction, a, b, c, slack=0.0):
    """Moller-Trumbore, batched: `(t, u, v, hit)` of rays against triangles (a, b, c).

    `slack` widens the triangle by that much in barycentric terms, so that a
    pixel centre lying exactly on the edge two faces share is hit by both and
    the depth test chooses, instead of by neither through rounding.
    """
    import torch

    e1, e2 = b - a, c - a
    p = torch.cross(direction, e2, dim=-1)
    det = (e1 * p).sum(-1)
    parallel = det.abs() < 1e-12
    inv = 1.0 / torch.where(parallel, torch.ones_like(det), det)
    s = origin - a
    u = (s * p).sum(-1) * inv
    q = torch.cross(s, e1, dim=-1)
    v = (direction * q).sum(-1) * inv
    t = (e2 * q).sum(-1) * inv
    hit = ~parallel & (u >= -slack) & (v >= -slack) & (u + v <= 1 + slack) & (t > 0)
    return t, u, v, hit


def _exact(view, faces, face, k, z_clip, size):
    """Depth and barycentrics of the face each pixel sees, from the pixel's ray.

    A pixel the pipeline gave to a face whose exact test just misses (it lies
    on the edge two faces share) keeps that face, with its barycentrics brought
    back onto the triangle.
    """
    import torch

    rays = _pixel_rays(size, k, view.device)
    hit = face >= 0
    corners = view[faces[face.clamp(min=0)]]
    t, u, v, _ = _intersect(z_clip * rays, rays, corners[:, 0], corners[:, 1], corners[:, 2])
    u, v = u.clamp(0, 1), v.clamp(0, 1)
    over = (u + v).clamp(min=1)
    u, v = u / over, v / over
    zbuf = torch.where(hit, z_clip + t, torch.full_like(t, -1.0))
    bary = torch.where(hit[:, None], torch.stack([1 - u - v, u, v], -1), torch.full((len(t), 3), -1.0, device=t.device))
    return (face.reshape(size, size), zbuf.reshape(size, size).to(torch.float32),
            bary.reshape(size, size, 3).to(torch.float32))


def rasterize(verts, faces, R, T, k, z_clip, size):
    """`(pix_to_face (S, S) int64, zbuf (S, S), bary (S, S, 3))` of one mesh, one view.

    `verts` (V, 3) and `faces` (F, 3) are tensors on the device the answer is
    wanted on; `R` (3, 3) and `T` (3,) the pose; `k` the projection
    `(k00, k11, k20, k21)`; `z_clip` the plane closer than which nothing is
    seen.
    """
    import torch

    size = int(size)
    device = verts.device
    faces = faces.to(device=device, dtype=torch.int64)
    if len(faces) == 0:
        miss = torch.full((size, size), -1, dtype=torch.int64, device=device)
        return miss, miss.to(torch.float32), torch.full((size, size, 3), -1.0, device=device)
    view = verts.to(torch.float32) @ R.to(device, torch.float32) + T.to(device, torch.float32)
    face = None
    if _gl() is not None:
        try:
            face = _gl_faces(verts, faces, view, R, T, k, z_clip, size)
        except Exception:  # noqa: BLE001 - the fallback below gives the same answer
            logger.warning("OpenGL rasterization failed; rasterizing with PyTorch instead", exc_info=True)
            face = None
    if face is None:
        face = _torch_faces(view, faces, k, z_clip, size)
    return _exact(view, faces, face, k, z_clip, size)


# -- OpenGL ------------------------------------------------------------------

_VERTEX = """
#version 410
uniform mat3 R; uniform vec3 T; uniform vec4 k; uniform float near; uniform float far;
in vec3 pos; flat out int fid;
void main() {
    vec3 p = pos * R + T;               // row vector times R: X_view = X @ R + T
    float zc = (far + near) / (far - near) * p.z - 2.0 * far * near / (far - near);
    // pytorch3d's +X is the LEFT of the image, so it becomes -x here; the clip
    // coordinates are multiplied by w = z, which the divide then removes.
    gl_Position = vec4(-(k.x * p.x + k.z * p.z), k.y * p.y + k.w * p.z, zc, p.z);
    fid = gl_VertexID / 3;
}
"""

# The face index + 1 in four 8-bit channels: a normalized format, which clears
# to zero reliably everywhere, where integer formats do not on every driver.
_FRAGMENT = """
#version 410
flat in int fid; out vec4 colour;
void main() {
    int id = fid + 1;
    colour = vec4(id & 255, (id >> 8) & 255, (id >> 16) & 255, (id >> 24) & 255) / 255.0;
}
"""


@functools.cache
def _gl():
    """`{"ctx", "program", "meshes", "targets"}`, or None when no context can be made."""
    try:
        import moderngl

        options = {"backend": "egl"} if sys.platform.startswith("linux") else {}
        ctx = moderngl.create_standalone_context(require=410, **options)
        program = ctx.program(vertex_shader=_VERTEX, fragment_shader=_FRAGMENT)
    except Exception:  # noqa: BLE001 - no GPU, no OpenGL, no moderngl: the fallback
        logger.warning("No OpenGL context; rasterizing with PyTorch instead", exc_info=True)
        return None
    return {"ctx": ctx, "program": program, "mesh": None, "targets": {}}


def _gl_faces(verts, faces, view, R, T, k, z_clip, size):
    import torch

    gl = _gl()
    ctx, program = gl["ctx"], gl["program"]
    with ctx:  # current again: another library in this process may have taken the context
        held = gl["mesh"]
        # The geometry is uploaded only when it changed. `meshes.clone()` per
        # view (AREG_IOS, shapeaxi) hands over new tensors with the same
        # content every time, so the test is by value, not by identity.
        v_np = verts.detach().cpu().to(torch.float32).numpy()
        f_np = faces.detach().cpu().numpy()
        if held is None or not (np.array_equal(held[0], v_np) and np.array_equal(held[1], f_np)):
            if held is not None:
                held[2].release()
                held[3].release()
            corners = np.ascontiguousarray(v_np[f_np].reshape(-1, 3), dtype="f4")
            buffer = ctx.buffer(corners.tobytes())
            vao = ctx.vertex_array(program, [(buffer, "3f", "pos")])
            gl["mesh"] = held = (v_np, f_np, vao, buffer)
        if size not in gl["targets"]:
            gl["targets"][size] = ctx.framebuffer(
                [ctx.renderbuffer((size, size), 4, dtype="f1")], ctx.depth_renderbuffer((size, size)))
        target = gl["targets"][size]
        target.use()
        target.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
        ctx.enable(ctx.DEPTH_TEST)
        ctx.disable(ctx.CULL_FACE)
        rotation = R.detach().cpu().to(torch.float32).numpy()
        # GLSL matrices are column-major, so R itself goes up in column order
        # for `pos * R` to multiply by its columns.
        program["R"].write(np.ascontiguousarray(rotation.T, dtype="f4").tobytes())
        program["T"].value = tuple(float(x) for x in T.detach().cpu().reshape(-1))
        program["k"].value = tuple(float(x) for x in k)
        far = float(view[:, 2].max()) * 1.01 + 1e-6
        program["near"].value = float(z_clip) if z_clip > 0 else 1e-6
        program["far"].value = max(far, float(z_clip) * 2 + 1e-6)
        held[2].render()
        rgba = np.frombuffer(target.read(components=4, dtype="f1"), np.uint8).reshape(size, size, 4)[::-1]
    ids = rgba.astype(np.int64)
    ids = ids[..., 0] | (ids[..., 1] << 8) | (ids[..., 2] << 16) | (ids[..., 3] << 24)
    return torch.from_numpy(ids.reshape(-1) - 1).to(verts.device)


# -- PyTorch alone (the fallback) ---------------------------------------------

def _torch_faces(view, faces, k, z_clip, size):
    """Which face each pixel sees, by testing every pixel centre inside each
    face's bounding box against that face, then keeping the nearest hit.

    A face that crosses the clip plane is bounded by its part in front of it:
    its vertices beyond the plane and the points where its edges cross it.
    """
    import torch

    device = view.device
    k00, k11, k20, k21 = k
    near = max(float(z_clip), 1e-6)
    corners = view[faces]                                   # (F, 3, 3)
    z = corners[..., 2]
    front = z > near
    keep = front.any(1)
    # Where each edge (i -> i+1) crosses the plane, if it does.
    tail, head = corners, corners.roll(-1, dims=1)
    zt, zh = tail[..., 2], head[..., 2]
    crosses = (zt > near) != (zh > near)
    fraction = ((near - zt) / torch.where(crosses, zh - zt, torch.ones_like(zt))).clamp(0, 1)
    crossing = tail + fraction[..., None] * (head - tail)
    points = torch.cat([corners, crossing], 1)              # (F, 6, 3)
    valid = torch.cat([front, crosses], 1)
    pz = points[..., 2].clamp(min=near)
    col = ((1 - (k00 * points[..., 0] / pz + k20)) * size - 1) / 2
    row = ((1 - (k11 * points[..., 1] / pz + k21)) * size - 1) / 2
    big = torch.full_like(col, float("inf"))
    lo_c = torch.where(valid, col, big).min(1).values.ceil()
    hi_c = torch.where(valid, col, -big).max(1).values.floor()
    lo_r = torch.where(valid, row, big).min(1).values.ceil()
    hi_r = torch.where(valid, row, -big).max(1).values.floor()
    lo_c = torch.where(keep, lo_c, torch.zeros_like(lo_c)).clamp(-1, size)
    hi_c = torch.where(keep, hi_c, -torch.ones_like(hi_c)).clamp(-1, size)
    lo_r = torch.where(keep, lo_r, torch.zeros_like(lo_r)).clamp(-1, size)
    hi_r = torch.where(keep, hi_r, -torch.ones_like(hi_r)).clamp(-1, size)
    in_front = keep
    w = (hi_c.clamp(max=size - 1) - lo_c.clamp(min=0) + 1).clamp(min=0).to(torch.int64)
    h = (hi_r.clamp(max=size - 1) - lo_r.clamp(min=0) + 1).clamp(min=0).to(torch.int64)
    n = torch.where(in_front, w * h, torch.zeros_like(w))
    chosen = torch.nonzero(n > 0).squeeze(1)
    counts = n[chosen]
    total = int(counts.sum())
    face = torch.full((size * size,), -1, dtype=torch.int64, device=device)
    if total == 0:
        return face
    face_of = torch.repeat_interleave(chosen, counts)
    start = torch.repeat_interleave(torch.cumsum(counts, 0) - counts, counts)
    index = torch.arange(total, device=device) - start
    width = w[face_of]
    pc = lo_c.clamp(min=0)[face_of].to(torch.int64) + index % width
    pr = lo_r.clamp(min=0)[face_of].to(torch.int64) + index // width
    rays = _pixel_rays(size, k, device)[pr * size + pc]
    corners = view[faces[face_of]]
    t, _u, _v, hit = _intersect(z_clip * rays, rays, corners[:, 0], corners[:, 1], corners[:, 2], slack=1e-5)
    pixel = (pr * size + pc)[hit]
    depth = (z_clip + t)[hit]
    candidate = torch.nonzero(hit).squeeze(1)
    nearest = torch.full((size * size,), float("inf"), device=device)
    nearest = nearest.scatter_reduce(0, pixel, depth, reduce="amin", include_self=True)
    winner = depth <= nearest[pixel]
    # int32: MPS has no int64 scatter_reduce, and the candidates are far fewer than 2**31.
    best = torch.full((size * size,), total, dtype=torch.int32, device=device)
    best = best.scatter_reduce(0, pixel[winner], candidate[winner].to(torch.int32), reduce="amin", include_self=True)
    seen = best < total
    face[seen] = face_of[best[seen].to(torch.int64)]
    return face


# ---------------------------------------------------------------------------
# pytorch3d's MeshRasterizer, answered by `rasterize`
# ---------------------------------------------------------------------------

def _check_settings(settings, perspective_correct: bool) -> None:
    """Refuse what this rasterizer does not reproduce, rather than approximate it."""
    unsupported = []
    if settings.blur_radius != 0:
        unsupported.append("blur_radius != 0")
    if settings.faces_per_pixel != 1:
        unsupported.append("faces_per_pixel != 1")
    if settings.cull_backfaces:
        unsupported.append("cull_backfaces")
    if settings.cull_to_frustum:
        unsupported.append("cull_to_frustum")
    if not perspective_correct:
        unsupported.append("perspective_correct=False")
    if not isinstance(settings.image_size, int):
        unsupported.append("a non-square image_size")
    if unsupported:
        raise NotImplementedError(
            "The OpenGL rasterizer does not support: " + ", ".join(unsupported)
        )


def _hard_rgb_blend_for_mps(original):
    """pytorch3d's `hard_rgb_blend`, with an MPS-safe background.

    On MPS, `masked_scatter` with a broadcast mask writes the background colour
    into covered pixels as well (the image comes out black where the mesh is,
    from the black background the networks were trained on), while the alpha
    channel is right. `torch.where` gives the same values as `masked_scatter`
    everywhere else. Other devices call pytorch3d's own function.
    """
    import torch

    def hard_rgb_blend(colors, fragments, blend_params):
        if colors.device.type != "mps":
            return original(colors, fragments, blend_params)
        background = blend_params.background_color
        if isinstance(background, torch.Tensor):
            background = background.to(colors.device)
        else:
            background = colors.new_tensor(background)
        is_background = fragments.pix_to_face[..., 0] < 0
        pixel_colors = torch.where(is_background[..., None], background, colors[..., 0, :])
        alpha = (~is_background).type_as(pixel_colors)[..., None]
        return torch.cat([pixel_colors, alpha], dim=-1)

    hard_rgb_blend._sadt_wrapped = True
    return hard_rgb_blend


@functools.cache
def rasterizer_class():
    """`GLRasterizer`, a `MeshRasterizer` whose `forward` is `rasterize`.

    Built on first use so that importing this module does not import pytorch3d.
    It is a subclass, so `.to()`, `.cameras` and `.raster_settings` behave as
    pytorch3d's, and it can stand wherever a `MeshRasterizer` is accepted.
    """
    from pytorch3d.renderer import MeshRasterizer
    from pytorch3d.renderer.mesh import shader
    from pytorch3d.renderer.mesh.rasterizer import Fragments

    # The shaders are pytorch3d's, and on MPS their background blend is wrong
    # (see `_hard_rgb_blend_for_mps`); this is the one place that reaches them.
    if not hasattr(shader.hard_rgb_blend, "_sadt_wrapped"):
        shader.hard_rgb_blend = _hard_rgb_blend_for_mps(shader.hard_rgb_blend)

    class GLRasterizer(MeshRasterizer):
        # Not thread-safe, like pytorch3d's own rasterizer (the cameras keep
        # the last pose): one renderer per thread.

        def forward(self, meshes_world, **kwargs):
            import torch

            settings = kwargs.get("raster_settings", self.raster_settings)
            cameras = kwargs.get("cameras", self.cameras)
            if cameras is None:
                raise ValueError("Cameras must be specified either at initialization or in the forward pass")
            n_meshes, n_cameras = len(meshes_world), len(cameras)
            if n_cameras != 1 and n_cameras != n_meshes:
                raise ValueError(f"Wrong number ({n_cameras}) of cameras for {n_meshes} meshes")

            perspective_correct = (
                settings.perspective_correct
                if settings.perspective_correct is not None
                else cameras.is_perspective()
            )
            _check_settings(settings, perspective_correct)
            if settings.z_clip_value is not None:
                z_clip = float(settings.z_clip_value)
            else:
                znear = cameras.get_znear()
                if isinstance(znear, torch.Tensor):
                    znear = znear.min().item()
                z_clip = 0.0 if znear is None else znear / 2

            # Same calls as MeshRasterizer.transform, and the same side effect:
            # the cameras remember the R and T they were given, which is what
            # lets `rasterizer(meshes)` with no R and T replay the last view.
            world_to_view = cameras.get_world_to_view_transform(**kwargs).get_matrix()
            projection = cameras.get_projection_transform(**kwargs).compose(
                cameras.get_ndc_camera_transform(**kwargs)
            ).get_matrix()
            world_to_view = world_to_view.detach().cpu()
            projection = projection.detach().cpu().double().numpy()  # MPS has no float64
            if np.abs(projection[:, 2, 3] - 1).max() > 1e-6 or np.abs(projection[:, 3, 3]).max() > 1e-6:
                raise NotImplementedError("This rasterizer needs a perspective projection (w = z)")

            size = settings.image_size
            device = meshes_world.device
            verts_list, faces_list = meshes_world.verts_list(), meshes_world.faces_list()
            first_face = np.cumsum([0] + [len(f) for f in faces_list])
            pix, zbuf, bary = [], [], []
            for i, (verts, faces) in enumerate(zip(verts_list, faces_list)):
                c = i if len(world_to_view) > 1 else 0
                m, p = world_to_view[c], projection[c]
                k = (float(p[0, 0]), float(p[1, 1]), float(p[2, 0]), float(p[2, 1]))
                one_pix, one_z, one_bary = rasterize(verts, faces, m[:3, :3], m[3, :3], k, z_clip, size)
                pix.append(torch.where(one_pix >= 0, one_pix + int(first_face[i]), one_pix))
                zbuf.append(one_z)
                bary.append(one_bary)
            pix_to_face = torch.stack(pix)[..., None].to(device)
            zbuf = torch.stack(zbuf)[..., None].to(device)
            bary_coords = torch.stack(bary)[..., None, :].to(device)
            dists = torch.where(pix_to_face >= 0, torch.zeros_like(zbuf), torch.full_like(zbuf, -1.0))
            return Fragments(pix_to_face=pix_to_face, zbuf=zbuf, bary_coords=bary_coords, dists=dists)

    return GLRasterizer
