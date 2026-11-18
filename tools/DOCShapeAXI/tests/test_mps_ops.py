"""The two pytorch3d ops replaced on macOS / MPS, checked on the CPU against the originals.

`use_embree_and_cdist` is only reached when `embree_raster.wanted(device)`; these
tests call the replacements directly, with CPU tensors, so they run anywhere
pytorch3d imports.
"""

import pytest

from sadt_docshapeaxi import pipeline

torch = pytest.importorskip("torch")
knn = pytest.importorskip("pytorch3d.ops.knn")


def test_cdist_knn_picks_the_same_neighbours_as_pytorch3d():
    torch.manual_seed(0)
    x = torch.randn(1, 512, 64)
    reference = knn.knn_points(x, x, K=16, return_sorted=True)
    replaced = pipeline._knn_by_cdist(knn.knn_points)(x, x, K=16, return_sorted=True)
    assert (reference.idx == replaced.idx).float().mean() > 0.999
    assert torch.allclose(reference.dists, replaced.dists, atol=1e-3, rtol=1e-3)
    assert replaced.knn is None


def test_cdist_knn_leaves_what_it_does_not_cover_to_pytorch3d():
    """More neighbours than points: pytorch3d pads, and `topk` cannot, so the
    original answers."""
    x = torch.randn(1, 4, 3)
    reference = knn.knn_points(x, x, K=6)
    replaced = pipeline._knn_by_cdist(knn.knn_points)(x, x, K=6)
    assert replaced.idx.shape == (1, 4, 6)
    assert (reference.idx == replaced.idx).all()


def test_sampling_on_a_cpu_mesh_is_passed_straight_through():
    structures = pytest.importorskip("pytorch3d.structures")
    ops = pytest.importorskip("pytorch3d.ops")
    verts = torch.tensor([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    mesh = structures.Meshes(verts=[verts], faces=[torch.tensor([[0, 1, 2]])])
    wrapped = pipeline._on_cpu(ops.sample_points_from_meshes)
    points, normals = wrapped(mesh, 8, return_normals=True)
    assert points.shape == (1, 8, 3) and normals.shape == (1, 8, 3)
    assert getattr(wrapped, "_sadt_wrapped", False)


def test_a_module_that_is_not_shapeaxis_renderer_is_left_alone():
    """On macOS `load_network` reaches the patch even in the tests that stand a
    stub in for shapeaxi; the stub has no `MeshRasterizer`, so nothing happens."""
    import types

    stub = types.ModuleType("shapeaxi.saxi_nets_lightning")
    pipeline.use_embree_and_cdist(stub)
    assert not hasattr(stub, "MeshRasterizer")
