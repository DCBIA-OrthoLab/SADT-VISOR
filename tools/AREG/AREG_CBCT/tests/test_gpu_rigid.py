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

needs_gpu = pytest.mark.skipif(not gpu_rigid.available(), reason="no GPU (CUDA or MPS)")


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


def test_a_walk_that_goes_uphill_keeps_the_best_point_it_had(monkeypatch):
    """Adam moves by its step whatever the slope, and on a masked CBCT it
    walked off an already-aligned pair by 20 degrees and 40 mm, to a WORSE
    mutual information than its start. Each level now keeps the best point it
    visited. Modelled on the CPU: every step makes the metric worse."""
    def uphill(self, level, params):
        params.grad = -torch.ones_like(params)
        return float(params.detach().abs().sum())

    monkeypatch.setattr(gpu_rigid._Registration, "_gradient", uphill)
    fixed = phantom(size=24)
    registration = gpu_rigid._Registration(fixed, fixed, torch.device("cpu"))
    transform = registration.run()
    assert transform.GetAngleX() == transform.GetAngleY() == transform.GetAngleZ() == 0
    assert np.allclose(transform.GetTranslation(), registration.initial_translation)


def test_the_steps_are_the_ones_measured_against_elastix():
    """(2.0, 0.5, 0.1) walked off both hosted CBCT pairs; (0.5, 0.2, 0.05)
    lands on elastix's answer to 0.04 mm. Changing them needs that measure again."""
    assert gpu_rigid.STEPS == (0.5, 0.2, 0.05)


def test_without_a_card_the_registration_is_elastixs(monkeypatch):
    monkeypatch.setattr(gpu_rigid, "available", lambda: False)
    monkeypatch.setattr(elastix, "register", lambda fixed, moving, cost=None: "elastix transform")
    assert pipeline._register(None, None) == ("elastix transform", "elastix")


def test_a_failure_on_the_card_falls_back_to_elastix(monkeypatch):
    def out_of_memory(fixed, moving, cost=None):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(gpu_rigid, "available", lambda: True)
    monkeypatch.setattr(gpu_rigid, "register", out_of_memory)
    monkeypatch.setattr(elastix, "register", lambda fixed, moving, cost=None: "elastix transform")
    assert pipeline._register(None, None) == ("elastix transform", "elastix")


def test_the_device_is_chosen_on_availability_cuda_first(monkeypatch):
    mps = torch.backends.mps
    for cuda, apple, expected in ((True, True, "cuda"), (True, False, "cuda"),
                                  (False, True, "mps"), (False, False, None)):
        monkeypatch.setattr(torch.cuda, "is_available", lambda cuda=cuda: cuda)
        monkeypatch.setattr(mps, "is_available", lambda apple=apple: apple)
        device = gpu_rigid._device()
        assert (device.type if device else None) == expected
        assert gpu_rigid.available() is (expected is not None)


def test_the_cache_emptied_is_the_one_of_the_device_used(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("cuda"))
    monkeypatch.setattr(torch.mps, "empty_cache", lambda: calls.append("mps"))
    gpu_rigid._empty_cache(torch.device("cuda"))
    gpu_rigid._empty_cache(torch.device("mps"))
    gpu_rigid._empty_cache(torch.device("cpu"))
    assert calls == ["cuda", "mps"]


def test_registering_without_a_gpu_raises_so_that_elastix_takes_over(monkeypatch):
    monkeypatch.setattr(gpu_rigid, "_device", lambda: None)
    fixed = phantom(size=16)
    with pytest.raises(RuntimeError, match="no GPU"):
        gpu_rigid.register(fixed, fixed)


def test_a_failure_on_mps_falls_back_with_the_same_message(monkeypatch, caplog):
    def refused(fixed, moving, cost=None):
        raise RuntimeError("index_put_with_accumulate_mps does not have a deterministic implementation")

    monkeypatch.setattr(gpu_rigid, "available", lambda: True)
    monkeypatch.setattr(gpu_rigid, "register", refused)
    monkeypatch.setattr(elastix, "register", lambda fixed, moving, cost=None: "elastix transform")
    with caplog.at_level("WARNING"):
        assert pipeline._register(None, None) == ("elastix transform", "elastix")
    assert "GPU registration failed" in caplog.text
    assert "registering with elastix" in caplog.text


def test_the_edges_are_replicated_as_replicate_pad_does_to_the_bit():
    """MPS's own replicate pad is wrong on a large volume, so `_blur_axis`
    builds the padding itself; on CPU the two must agree exactly."""
    import torch.nn.functional as F

    volume = torch.from_numpy(np.random.default_rng(1).random((14, 17, 19)).astype(np.float32))
    kernel, radius = gpu_rigid._gaussian(4, volume.device)
    for axis in range(3):
        shape = [1, 1, 1, 1, 1]
        shape[2 + axis] = kernel.numel()
        pad = [0] * 6
        pad[2 * (2 - axis)] = pad[2 * (2 - axis) + 1] = radius
        expected = F.conv3d(F.pad(volume[None, None], pad, mode="replicate"), kernel.view(shape))
        index = [slice(None)] * 5
        index[2 + axis] = slice(None, None, 4)
        expected = expected[tuple(index)][0, 0]
        assert torch.equal(gpu_rigid._blur_axis(volume, axis, 4, kernel, radius), expected)


def test_the_mutual_information_and_its_gradient_are_those_of_boolean_indexing_to_the_bit():
    """`masked_select` replaced `joint[present]` (whose gradient MPS refuses in
    deterministic mode): the value and the gradient must not move."""
    rng = np.random.default_rng(2)
    values = torch.from_numpy(rng.random((64, 64)).astype(np.float32))
    values[values < 0.4] = 0

    def old(joint):
        joint = joint / joint.sum()
        marginal = joint.sum(1, keepdim=True) @ joint.sum(0, keepdim=True)
        present = joint > 0
        return -(joint[present] * (joint[present] / marginal[present]).log()).sum()

    results = []
    for function in (old, gpu_rigid._Registration._negative_mi):
        joint = values.clone().requires_grad_(True)
        value = function(joint)
        value.backward()
        results.append((value.detach(), joint.grad))
    assert torch.equal(results[0][0], results[1][0])
    assert torch.equal(results[0][1], results[1][1])


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


@pytest.mark.gpu
@needs_gpu
def test_a_level_built_slab_by_slab_is_the_whole_volume_blurred(monkeypatch):
    """The slabs are cut across an axis the blur does not run along, so none
    needs its neighbours: put back together they are the whole volume."""
    array = np.random.default_rng(0).random((40, 36, 44)).astype(np.float32)
    device = gpu_rigid._device()
    kernel, radius = gpu_rigid._gaussian(2, device)
    whole = torch.from_numpy(array).to(device)
    for axis in range(3):
        whole = gpu_rigid._blur_axis(whole, axis, 2, kernel, radius)
    monkeypatch.setattr(gpu_rigid, "SLAB_BYTES", 4 * 36 * 44 * 3)  # a few rows per slab
    slabbed = gpu_rigid._level_volume(array, 2, device)
    assert slabbed.shape == whole.shape
    assert torch.allclose(slabbed, whole, atol=1e-5)


@pytest.mark.gpu
@needs_gpu
def test_the_batched_gradient_is_the_whole_histogram_s_gradient(monkeypatch):
    """Two passes -- the histogram, then dMI/dJ pushed back batch by batch --
    give the gradient one pass over every sample gives, by the chain rule."""
    fixed = phantom(size=48)
    moving, _truth = _moved(fixed)
    registration = gpu_rigid._Registration(fixed, moving, gpu_rigid._device())
    level = gpu_rigid._Level(registration, factor=1, stride=1)
    params = torch.tensor([0.5, -0.3, 0.2, 0.4, -0.6, 0.3], device=gpu_rigid._device(), requires_grad=True)

    whole = registration._batch_joint(level, params, 0, level.points.shape[0])
    registration._negative_mi(whole).backward()
    expected = params.grad.clone()

    params.grad = None
    monkeypatch.setattr(gpu_rigid, "BATCH_SAMPLES", 10_000)
    assert len(level.batches()) > 5
    registration._gradient(level, params)
    assert torch.allclose(params.grad, expected, rtol=1e-4, atol=1e-7)


# The 48-voxel phantom's six parameters, measured on the RTX 6000 Ada before
# the device became a parameter. Other cards sum in another order and land a
# few bits away, so this pins that one and skips elsewhere.
PHANTOM_CUDA_BITS = ["0x1.41d7ca3d70a3dp-5", "-0x1.8c618f5c28f5cp-6", "0x1.ecd1147ae147bp-6",
                     "0x1.3b524c0000000p+0", "-0x1.9b5a480000000p+0", "0x1.c2fc8e0000000p-1"]


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or "RTX 6000 Ada" not in torch.cuda.get_device_name(0),
    reason="the reference bits were measured on an RTX 6000 Ada")
def test_the_cuda_path_still_gives_the_bits_it_gave():
    fixed = phantom(size=48)
    moving, _truth = _moved(fixed)
    bits = [float(v).hex() for v in gpu_rigid.register(fixed, moving).GetParameters()]
    assert bits == PHANTOM_CUDA_BITS
