"""pytorch3d's rasterizer, replaced by Embree ray casting (macOS, MPS).

pytorch3d has no Metal kernel: handed a tensor on `mps` its rasterizer raises
`Cannot use CPU implementation`, and on the CPU it takes about two seconds a
view. This module casts one ray per pixel through Embree instead and builds the
`Fragments` pytorch3d's own rasterizer would have returned. Everything after
the rasterizer (HardPhongShader, the cameras, the textures) is plain PyTorch
and stays pytorch3d's.

NOTHING HERE RUNS OFF macOS. `wanted()` is `sys.platform == "darwin"`, so on
Linux and Windows the existing code path is the one that runs. `embreex` is imported on the first rasterization, never
at import time, and is not a dependency of this package: the macOS recipe
installs it.

Six facts the parity with pytorch3d rests on, each measured (phase 0 and 1
reports):

* View space is `X_view = X_world @ R + T` (row vectors); NDC has +X to the
  LEFT and +Y up; pixel (row r, column c) is sampled at its centre,
  `x = 1 - (2c + 1) / S`, `y = 1 - (2r + 1) / S`.
* A ray whose view-space direction has z = 1 reaches the surface at
  `t = z_view`, which is exactly pytorch3d's `zbuf`.
* `bary_coords` are `(1 - u - v, u, v)` from Embree's Moller-Trumbore `u, v`.
  They agree with pytorch3d's perspective-correct coordinates.
* One face per pixel, `pix_to_face = -1` (and `zbuf = -1`) where nothing is hit.
* pytorch3d, with perspective-correct interpolation, clips the geometry at
  `z = znear / 2`. A mesh that crosses that plane is cut there, so the rays
  start ON the plane rather than at the camera. Without that, 45 % of the pixels
  of a close view differ; with it, 99.999 %.
* Faces are not culled, so back faces are hit like front ones.
"""

import functools
import logging
import sys

import numpy as np

logger = logging.getLogger(__name__)

_INSTALL_HINT = (
    "The Embree rasterizer, used on macOS, "
    "needs the `embreex` package, which the macOS install recipe provides: "
    "`uv pip install embreex` in this tool's environment. It is not a dependency "
    "of the Linux deployments, which rasterize with pytorch3d."
)


class ToolUnavailableError(RuntimeError):
    """The server maps this class by NAME to a 503 (see runner.CALLER_FACING_ERRORS)."""


def wanted(device=None) -> bool:
    """Whether to rasterize with Embree rather than with pytorch3d.

    True on macOS and only there: pytorch3d has no Metal kernel and the compiled
    CPU one takes seconds a view. `device` is accepted and ignored, so the call
    sites need no knowledge of it (Crown_Seg and AREG_IOS never name `mps`).
    Linux and Windows keep pytorch3d's own rasterizer.
    """
    return sys.platform == "darwin"


def _embreex():
    try:
        from embreex import rtcore_scene
        from embreex.mesh_construction import TriangleMesh
    except ImportError as exc:
        raise ToolUnavailableError(f"{_INSTALL_HINT} (missing: {exc.name or 'embreex'})") from exc
    return rtcore_scene, TriangleMesh


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
            "The Embree rasterizer does not support: " + ", ".join(unsupported)
        )


def _hard_rgb_blend_for_mps(original):
    """`hard_rgb_blend` with `torch.where` in place of `masked_scatter` on MPS.

    Measured on torch 2.8.0 (macOS, M4 Max): on the shaded colours of a real
    render, `colors.masked_scatter(is_background, background)` leaves the
    background pixels UNWRITTEN (they keep the shaded texture, up to 1.0 away
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
    """`EmbreeRasterizer`, a `MeshRasterizer` whose `forward` casts rays.

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

    class EmbreeRasterizer(MeshRasterizer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Not thread-safe, like pytorch3d's own rasterizer (the cameras keep
            # the last pose): one renderer per thread.
            self._scenes = {}     # mesh index -> (verts, faces, scene)
            self._last = None     # (key, arrays) of the previous call

        # The geometry is rebuilt only when it changed. `meshes.clone()` per
        # view (AREG_IOS, shapeaxi) hands over new tensors with the same
        # content every time, so the test is by value, not by identity.
        def _scene(self, index, verts, faces):
            held = self._scenes.get(index)
            if held is not None and np.array_equal(held[0], verts) and np.array_equal(held[1], faces):
                return held
            rtcore_scene, triangle_mesh = _embreex()
            scene = rtcore_scene.EmbreeScene()
            if len(faces):
                triangle_mesh(scene=scene, vertices=verts, indices=faces)
            held = (verts, faces, scene)
            self._scenes[index] = held
            return held

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
            world_to_view = world_to_view.detach().cpu().double().numpy()  # MPS has no float64
            projection = projection.detach().cpu().double().numpy()
            if np.abs(projection[:, 2, 3] - 1).max() > 1e-6 or np.abs(projection[:, 3, 3]).max() > 1e-6:
                raise NotImplementedError("The Embree rasterizer needs a perspective projection (w = z)")

            size = settings.image_size
            device = meshes_world.device
            verts_list, faces_list = meshes_world.verts_list(), meshes_world.faces_list()
            first_face = np.cumsum([0] + [len(f) for f in faces_list])
            pixel = 1 - (2 * np.arange(size) + 1) / size   # NDC of a pixel centre
            x_ndc, y_ndc = np.meshgrid(pixel, pixel)       # x varies along columns

            scenes = [
                self._scene(
                    i,
                    np.ascontiguousarray(v.detach().cpu().numpy(), dtype=np.float32),
                    np.ascontiguousarray(f.detach().cpu().numpy(), dtype=np.int32),
                )
                for i, (v, f) in enumerate(zip(verts_list, faces_list))
            ]
            key = (
                tuple(id(s[2]) for s in scenes), world_to_view.tobytes(),
                projection.tobytes(), size, z_clip,
            )
            if self._last is not None and self._last[0] == key:
                arrays = self._last[1]
            else:
                arrays = self._cast(scenes, first_face, world_to_view, projection,
                                    x_ndc, y_ndc, size, z_clip)
                self._last = (key, arrays)

            pix_to_face, zbuf, bary, dists = (torch.from_numpy(a.copy()).to(device) for a in arrays)
            return Fragments(pix_to_face=pix_to_face, zbuf=zbuf, bary_coords=bary, dists=dists)

        @staticmethod
        def _cast(scenes, first_face, world_to_view, projection, x_ndc, y_ndc, size, z_clip):
            n = len(scenes)
            pix_to_face = np.full((n, size, size, 1), -1, np.int64)
            zbuf = np.full((n, size, size, 1), -1, np.float32)
            bary = np.full((n, size, size, 1, 3), -1, np.float32)
            dists = np.full((n, size, size, 1), -1, np.float32)
            for i, (_, _, scene) in enumerate(scenes):
                c = i if len(world_to_view) > 1 else 0
                m, k = world_to_view[c], projection[c]
                rotation_inverse = np.linalg.inv(m[:3, :3])
                # View-space ray with z = 1 so that the hit distance IS z_view.
                view = np.stack([
                    (x_ndc - k[2, 0]) / k[0, 0],
                    (y_ndc - k[2, 1]) / k[1, 1],
                    np.ones_like(x_ndc),
                ], -1).reshape(-1, 3)
                directions = view @ rotation_inverse
                origin = -m[3, :3] @ rotation_inverse
                starts = origin + z_clip * directions   # on the clip plane
                hit = scene.run(
                    np.ascontiguousarray(starts, dtype=np.float32),
                    np.ascontiguousarray(directions, dtype=np.float32),
                    output=1,
                )
                face = np.asarray(hit["primID"]).astype(np.int64)
                found = (face >= 0) & (np.asarray(hit["geomID"]) >= 0)
                shape = (size, size, 1)
                pix_to_face[i] = np.where(found, face + first_face[i], -1).reshape(shape)
                zbuf[i] = np.where(found, np.asarray(hit["tfar"]) + z_clip, -1).reshape(shape)
                u, v = np.asarray(hit["u"]), np.asarray(hit["v"])
                weights = np.stack([1 - u - v, u, v], -1)
                bary[i] = np.where(found[:, None], weights, -1).reshape(size, size, 1, 3)
                dists[i] = np.where(found, 0.0, -1.0).reshape(shape)
            return pix_to_face, zbuf, bary, dists

    return EmbreeRasterizer
