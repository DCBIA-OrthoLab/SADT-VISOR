"""The GPU registration engine, held to what elastix is held to.

The tests marked `gpu` need a CUDA device; the others run anywhere.
"""

import numpy as np
import pytest
import SimpleITK as sitk
import torch

from conftest import phantom
from sadt_areg_cbct import elastix, gpu_rigid, pipeline
from test_run import _moved

needs_gpu = pytest.mark.skipif(not gpu_rigid.available(), reason="no CUDA device")


def _probes(image):
    return [image.TransformContinuousIndexToPhysicalPoint([float(v) for v in index])
            for index in ([0, 0, 0], [24, 24, 24], [47, 47, 47], [4, 40, 12])]


def _distance(a, b, points):
    return max(np.linalg.norm(np.subtract(a.TransformPoint(p), b.TransformPoint(p))) for p in points)


# ---------------------------------------------------------------------------
# Anywhere
# ---------------------------------------------------------------------------

def test_the_rotation_is_sitks_own_euler_convention():
    """A transposed or reordered matrix still registers -- onto the wrong
    angles, which only shows once the .tfm is read back by someone else."""
    angles = (0.11, -0.07, 0.19)
    expected = sitk.Euler3DTransform()
    expected.SetRotation(*angles)
    matrix = gpu_rigid._euler_matrix(torch.tensor(angles, dtype=torch.float64))
    assert np.allclose(matrix.numpy(), np.array(expected.GetMatrix()).reshape(3, 3), atol=1e-12)


def test_a_clinical_volume_starts_eight_times_coarser_and_samples_sparsely_at_the_end():
    assert gpu_rigid._levels((732, 732, 647)) == [(8, 1), (4, 1), (2, 2)]


def test_a_small_volume_starts_finer_rather_than_at_a_handful_of_voxels():
    assert gpu_rigid._levels((48, 48, 48)) == [(1, 1), (1, 1), (1, 1)]
    assert gpu_rigid._levels((128, 128, 128)) == [(4, 1), (2, 1), (1, 1)]


def test_without_a_card_the_registration_is_elastixs(monkeypatch):
    monkeypatch.setattr(gpu_rigid, "available", lambda: False)
    monkeypatch.setattr(elastix, "register", lambda fixed, moving: "elastix transform")
    assert pipeline._register(None, None) == ("elastix transform", "elastix")


def test_a_failure_on_the_card_falls_back_to_elastix(monkeypatch):
    def out_of_memory(fixed, moving):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(gpu_rigid, "available", lambda: True)
    monkeypatch.setattr(gpu_rigid, "register", out_of_memory)
    monkeypatch.setattr(elastix, "register", lambda fixed, moving: "elastix transform")
    assert pipeline._register(None, None) == ("elastix transform", "elastix")


# ---------------------------------------------------------------------------
# On the card
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@needs_gpu
def test_the_phantom_lands_on_its_ground_truth():
    """The bound `TestElastix` holds elastix to: within a fifth of a voxel."""
    fixed = phantom(size=48)
    moving, truth = _moved(fixed)
    transform = gpu_rigid.register(fixed, moving)
    error = _distance(transform, truth, _probes(fixed))
    assert error < 0.2, f"registration is {error:.3f} mm from the truth"


@pytest.mark.gpu
@needs_gpu
def test_two_runs_give_the_same_transform_to_the_bit():
    fixed = phantom(size=48)
    moving, _truth = _moved(fixed)
    first = gpu_rigid.register(fixed, moving).GetParameters()
    second = gpu_rigid.register(fixed, moving).GetParameters()
    assert first == second


@pytest.mark.gpu
@needs_gpu
def test_it_agrees_with_elastix():
    fixed = phantom(size=48)
    moving, _truth = _moved(fixed)
    gap = _distance(gpu_rigid.register(fixed, moving), elastix.register(fixed, moving), _probes(fixed))
    assert gap < 0.1, f"the engines disagree by {gap:.3f} mm"


@pytest.mark.gpu
@needs_gpu
def test_the_transform_is_centred_where_elastix_centres_it():
    fixed = phantom(size=48)
    moving, _truth = _moved(fixed)
    centre = gpu_rigid.register(fixed, moving).GetCenter()
    assert np.allclose(centre, gpu_rigid._geometric_centre(fixed))
