"""The rigid registration on the GPU: the same problem elastix solves, faster.

Same inputs as `elastix.register` (a masked fixed image, a moving image), same
output (a `sitk.Euler3DTransform` mapping fixed points to moving points,
centred on the fixed image's geometric centre), same criterion: Mattes mutual
information over 64 bins, zero-order Parzen window on the fixed image and cubic
B-spline on the moving one, three resolutions coarse to fine, starting from
coinciding geometric centres (elastix's `GeometricalCenter`).

Measured on a clinical pair (fixed 732x732x647 at 0.25 mm, moving 610x610x538
at 0.3 mm), on the deployment's RTX 6000 Ada: elastix 114 s on ten threads,
this 19 s, the two transforms 0.027 mm apart on average over the mask's voxels
(0.058 mm at worst). On the phantom `test_run.py` holds elastix to, 0.048 mm
from the ground truth, elastix 0.025 mm.

**Deterministic, bit for bit**, like elastix's single-threaded map: this is
patient data being resampled, and two runs of one request must agree. A CUDA
scatter-add sums in whatever order its atomics land, so neither the histogram
nor the interpolation uses one: the histogram is a matrix product of a one-hot
fixed bin per sample and four B-spline weights per sample, and the moving image
is interpolated by gathering its eight neighbours (`F.grid_sample`'s backward
accumulates with atomics).
"""

import logging
import math
import os

import numpy as np
import SimpleITK as sitk

logger = logging.getLogger(__name__)

# cuBLAS picks a non-deterministic reduction without a fixed workspace, and the
# variable is read when the context is created: it has to be set before torch
# touches the card.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

BINS = 64
# Three levels, coarse to fine, each half the voxel size of the one before. The
# coarsest is at most 8x the fixed image's voxels and keeps at least
# `COARSEST_VOXELS` along its shortest side, so a small image starts finer.
COARSEST_FACTOR = 8
COARSEST_VOXELS = 32
ITERATIONS = (150, 120, 80)
# Adam's step at each level, in millimetres: rotations are expressed as the arc
# they sweep at `RADIUS`, so one unit means about a millimetre either way.
STEPS = (2.0, 0.5, 0.1)
RADIUS = 50.0
# The cost of an iteration is its number of samples, not the resolution of the
# volume: above this many, a level samples every other voxel of its grid.
MAX_SAMPLES = 8_000_000


def available() -> bool:
    """Is there a CUDA device this process can use?"""
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


def register(fixed: sitk.Image, moving: sitk.Image) -> sitk.Euler3DTransform:
    """Rigidly register `moving` onto `fixed`; return the resampling transform."""
    import torch

    torch.use_deterministic_algorithms(True)
    logger.info(
        "GPU rigid registration: fixed %s voxels at %s mm, moving %s voxels at %s mm "
        "(Mattes mutual information, %d bins, levels %s)",
        "x".join(str(v) for v in fixed.GetSize()),
        "x".join(f"{v:.3g}" for v in fixed.GetSpacing()),
        "x".join(str(v) for v in moving.GetSize()),
        "x".join(f"{v:.3g}" for v in moving.GetSpacing()),
        BINS, _levels(fixed.GetSize()),
    )
    try:
        return _Registration(fixed, moving, torch.device("cuda")).run()
    finally:
        # The card is shared: what this run held goes back the moment it ends,
        # not when the process does.
        torch.cuda.empty_cache()


def _levels(size) -> list:
    """`(factor, stride)` per level, coarse to fine."""
    factor = COARSEST_FACTOR
    while factor > 1 and min(size) // factor < COARSEST_VOXELS:
        factor //= 2
    levels = []
    for _ in range(3):
        samples = int(np.prod([n // factor for n in size]))
        levels.append((factor, 2 if samples > MAX_SAMPLES else 1))
        factor = max(1, factor // 2)
    return levels


def _geometric_centre(image: sitk.Image) -> np.ndarray:
    size = np.array(image.GetSize())
    return np.array(image.TransformContinuousIndexToPhysicalPoint(((size - 1) / 2.0).tolist()))


def _blur_and_shrink(volume, factor: int):
    """Gaussian blur (sigma = factor / 2 voxels), then keep every factor-th voxel."""
    import torch
    import torch.nn.functional as F

    if factor == 1:
        return volume
    sigma = factor / 2.0
    radius = int(math.ceil(3 * sigma))
    x = torch.arange(-radius, radius + 1, device=volume.device, dtype=volume.dtype)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()
    v = volume[None, None]
    for axis in range(3):
        shape = [1, 1, 1, 1, 1]
        shape[2 + axis] = kernel.numel()
        pad = [0] * 6
        pad[2 * (2 - axis)] = pad[2 * (2 - axis) + 1] = radius
        v = F.conv3d(F.pad(v, pad, mode="replicate"), kernel.view(shape))
        # Subsampled right after its own axis is blurred, so the next axis
        # convolves a volume already factor times smaller.
        index = [slice(None)] * 5
        index[2 + axis] = slice(None, None, factor)
        v = v[tuple(index)]
    return v[0, 0].contiguous()


def _euler_matrix(angles):
    """sitk.Euler3DTransform's rotation with ComputeZYX off: Rz @ Rx @ Ry."""
    import torch

    ax, ay, az = angles
    cx, sx, cy, sy, cz, sz = ax.cos(), ax.sin(), ay.cos(), ay.sin(), az.cos(), az.sin()
    one, zero = torch.ones_like(ax), torch.zeros_like(ax)
    rx = torch.stack([one, zero, zero, zero, cx, -sx, zero, sx, cx]).view(3, 3)
    ry = torch.stack([cy, zero, sy, zero, one, zero, -sy, zero, cy]).view(3, 3)
    rz = torch.stack([cz, -sz, zero, sz, cz, zero, zero, zero, one]).view(3, 3)
    return rz @ rx @ ry


def _bspline3(u):
    """The cubic B-spline kernel."""
    import torch

    a = u.abs()
    return torch.where(a < 1, (4 - 6 * a ** 2 + 3 * a ** 3) / 6,
                       torch.where(a < 2, (2 - a) ** 3 / 6, torch.zeros_like(a)))


def _trilinear(volume, index):
    """`volume` (z, y, x) at continuous `index` (N, [x, y, z]), all inside it.

    The gradient reaches the coordinates through the weights only; the volume
    is gathered, never scattered into.
    """
    import torch

    depth, height, width = volume.shape
    upper = torch.tensor([width - 2, height - 2, depth - 2], device=index.device).clamp(min=0)
    base = torch.minimum(index.detach().floor().long().clamp(min=0), upper)
    fraction = index - base
    flat = volume.reshape(-1)
    x0, y0, z0 = base.unbind(1)
    fx, fy, fz = fraction.unbind(1)
    result = 0
    for dz, wz in ((0, 1 - fz), (1, fz)):
        for dy, wy in ((0, 1 - fy), (1, fy)):
            for dx, wx in ((0, 1 - fx), (1, fx)):
                corner = flat[((z0 + dz) * height + (y0 + dy)) * width + (x0 + dx)]
                result = result + corner * (wz * wy * wx)
    return result


class _Level:
    """One resolution: the fixed samples and the moving volume, on the card."""

    def __init__(self, registration, factor: int, stride: int):
        import torch
        import torch.nn.functional as F

        device = registration.device
        fixed, moving = registration.fixed, registration.moving
        f_spacing = np.array(fixed.GetSpacing())
        f_direction = np.array(fixed.GetDirection()).reshape(3, 3)
        m_spacing = np.array(moving.GetSpacing())
        m_direction = np.array(moving.GetDirection()).reshape(3, 3)

        self.moving = _blur_and_shrink(registration.moving_volume, factor)
        fixed_level = _blur_and_shrink(registration.fixed_volume, factor)
        fixed_level = fixed_level[::stride, ::stride, ::stride].contiguous()

        # Every sample's physical point: p = origin + D (spacing * index).
        zz, yy, xx = [torch.arange(n, device=device, dtype=torch.float32) for n in fixed_level.shape]
        grid = torch.stack(torch.meshgrid(zz, yy, xx, indexing="ij"), dim=-1).flip(-1).reshape(-1, 3)
        scale = torch.tensor(f_spacing * factor * stride, device=device, dtype=torch.float32)
        direction = torch.tensor(f_direction, device=device, dtype=torch.float32)
        origin = torch.tensor(fixed.GetOrigin(), device=device, dtype=torch.float32)
        self.points = (grid * scale) @ direction.T + origin

        values = fixed_level.reshape(-1)
        bins = ((values - registration.f_low) / registration.f_width).floor().clamp(0, BINS - 5).long() + 2
        self.fixed_onehot = F.one_hot(bins, BINS).float()

        # Physical point -> continuous index of this level's moving volume.
        self.m_origin = torch.tensor(moving.GetOrigin(), device=device, dtype=torch.float32)
        self.m_inverse = torch.tensor(np.linalg.inv(m_direction @ np.diag(m_spacing * factor)),
                                      device=device, dtype=torch.float32)
        self.m_last = torch.tensor(self.moving.shape[::-1], device=device, dtype=torch.float32) - 1


class _Registration:
    def __init__(self, fixed: sitk.Image, moving: sitk.Image, device):
        import torch

        self.device = device
        self.fixed, self.moving = fixed, moving
        self.fixed_volume = torch.from_numpy(
            sitk.GetArrayFromImage(fixed).astype(np.float32)).to(device)
        self.moving_volume = torch.from_numpy(
            sitk.GetArrayFromImage(moving).astype(np.float32)).to(device)
        self.centre = _geometric_centre(fixed)
        self.initial_translation = _geometric_centre(moving) - self.centre
        # Two bins of padding either side for the B-spline's support.
        self.f_low, self.f_width = self._bin_layout(self.fixed_volume)
        self.m_low, self.m_width = self._bin_layout(self.moving_volume)
        self._centre = torch.tensor(self.centre, device=device, dtype=torch.float32)
        self._initial = torch.tensor(self.initial_translation, device=device, dtype=torch.float32)

    @staticmethod
    def _bin_layout(volume):
        low, high = volume.min(), volume.max()
        return low, (high - low).clamp(min=1e-6) / (BINS - 4)

    def _negative_mi(self, level: _Level, params):
        import torch

        rotation = _euler_matrix(params[:3] / RADIUS)
        mapped = (level.points - self._centre) @ rotation.T + self._centre + params[3:] + self._initial
        index = (mapped - level.m_origin) @ level.m_inverse.T
        # Samples mapped outside the moving image take no part, as in elastix.
        # Weighted out rather than filtered, so every iteration has one shape.
        inside = ((index >= 0) & (index <= level.m_last)).all(dim=1).float()
        index = torch.minimum(index.clamp(min=0), level.m_last)
        moving_values = _trilinear(level.moving, index)

        position = ((moving_values - self.m_low) / self.m_width + 2).clamp(1, BINS - 3)
        columns = torch.stack([position.detach().floor() + k for k in (-1, 0, 1, 2)], dim=1)
        weights = _bspline3(position[:, None] - columns) * inside[:, None]
        moving_rows = torch.zeros(index.shape[0], BINS, device=self.device).scatter(
            1, columns.long(), weights)
        joint = level.fixed_onehot.T @ moving_rows
        joint = joint / joint.sum()
        marginal = joint.sum(1, keepdim=True) @ joint.sum(0, keepdim=True)
        present = joint > 0
        return -(joint[present] * (joint[present] / marginal[present]).log()).sum()

    def run(self) -> sitk.Euler3DTransform:
        import torch

        params = torch.zeros(6, device=self.device, requires_grad=True)
        for (factor, stride), iterations, step in zip(
                _levels(self.fixed.GetSize()), ITERATIONS, STEPS):
            level = _Level(self, factor, stride)
            optimiser = torch.optim.Adam([params], lr=step)
            schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimiser, iterations, eta_min=step / 20)
            for _ in range(iterations):
                optimiser.zero_grad()
                self._negative_mi(level, params).backward()
                optimiser.step()
                schedule.step()
            del level
        values = params.detach().cpu().numpy().astype(float)
        transform = sitk.Euler3DTransform()
        transform.SetCenter(self.centre.tolist())
        transform.SetRotation(*(values[:3] / RADIUS).tolist())
        transform.SetTranslation((values[3:] + self.initial_translation).tolist())
        return transform
