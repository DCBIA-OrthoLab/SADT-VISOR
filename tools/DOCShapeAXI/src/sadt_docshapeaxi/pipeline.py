"""Classify a 3D shape with shapeaxi, and explain the decision on the surface.

Ported from `DOCShapeAXI_CLI/DOCShapeAXI_CLI.py`. Each surface is rendered from
several viewpoints, a shapeaxi network predicts a class (or a value, for the
regression checkpoint), and captum's LayerGradCam projects the attribution back
onto the mesh as a point array.

The prediction loop is upstream's rather than shapeaxi's own `saxi_predict`,
and deliberately: `saxi_predict` resolves the network with
`getattr(saxi_nets, args.nn)`, while `SaxiMHAFBClassification` and
`SaxiMHAFBRegression` live in `saxi_nets_lightning`. It raises AttributeError
before touching a mesh -- the same defect shapeaxi fixed for `dentalmodelseg`
in 2.0.3 and has not fixed here.
"""

import logging
import os

logger = logging.getLogger("DOCShapeAXI")

# What a surface file may be called. Upstream globbed `.vtk` only.
SURFACE_EXTENSIONS = (".vtk",)

# How many of the most likely pixels are averaged when a class wins nothing.
GRADCAM_IMAGE_SIZE = (224, 224)

# The percentiles the attribution map is clipped to before being normalised to
# [-1, 1]. Upstream's values, borrowed from pytorch-grad-cam.
CLIP_LOW, CLIP_HIGH = 1, 99


def import_torch():
    """torch, imported late: CI publishes the schema on every pull request and
    that must not cost a CUDA stack."""
    import torch

    return torch


def discover_surfaces(input_path) -> list:
    """Every `.vtk` under `input_path`, sorted, as file paths.

    Recursive, unlike upstream's `os.listdir`, and it returns FILES. Upstream
    also appended to its CSV with mode `'a'` and only rebuilt it when absent,
    so a second run on the same output folder silently reused a stale list.

    `os.fspath` because the runner hands a tool `pathlib.Path` objects for its
    `path` arguments while every test writes strings. A single uploaded file
    reached `is_surface_file` as a `Path` and died on `.lower()`.
    """
    input_path = os.fspath(input_path)
    if os.path.isfile(input_path):
        return [input_path] if is_surface_file(input_path) else []

    found = []
    for directory, _subdirs, names in os.walk(input_path):
        for name in sorted(names):
            if not name.startswith(".") and is_surface_file(name):
                found.append(os.path.join(directory, name))
    return sorted(found)


def is_surface_file(name) -> bool:
    return os.fspath(name).lower().endswith(SURFACE_EXTENSIONS)


def resolve_device(requested: str) -> str:
    torch = import_torch()
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("cuda requested but no CUDA device is visible; running on cpu")
        return "cpu"
    return requested


# The 2D backbone shapeaxi builds around the mesh renderer. It is NOT in the
# checkpoint: `EfficientNet.from_pretrained` fetches it from GitHub the first
# time a network is constructed, which is an outbound call from inside a
# request on a server holding patient data. It is staged instead, and
# `check_backbone_is_staged` refuses rather than reaching for it.
BACKBONE_FILE = "efficientnet-b0-355c32eb.pth"
BACKBONE_URL = (
    "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/"
    + BACKBONE_FILE
)


def backbone_cache_path() -> str:
    """Where torch.hub would look for the backbone, honouring `TORCH_HOME`."""
    import torch

    return os.path.join(torch.hub.get_dir(), "checkpoints", BACKBONE_FILE)


def check_backbone_is_staged() -> None:
    """Refuse to build a network whose backbone would have to be downloaded."""
    cached = backbone_cache_path()
    if os.path.isfile(cached):
        return
    raise FileNotFoundError(
        f"The EfficientNet backbone '{BACKBONE_FILE}' is not staged. shapeaxi "
        f"builds it around every checkpoint and would otherwise download it "
        f"mid-request, which a server holding patient data must not do. Stage "
        f"it at '{cached}' (from {BACKBONE_URL}), or point TORCH_HOME at a "
        f"directory that already holds it."
    )


def allow_checkpoint_globals() -> None:
    """Allowlist the classes the published checkpoints pickle.

    torch 2.6 changed `torch.load`'s `weights_only` default to True, and these
    checkpoints carry their training transform pipeline in `hyper_parameters`.
    An allowlist names exactly what is trusted; `weights_only=False` would
    trust everything the file happens to contain.
    """
    import inspect

    import torch
    import torchvision.transforms.transforms as transforms
    from shapeaxi import saxi_transforms

    allowed = [
        cls for _, cls in inspect.getmembers(saxi_transforms, inspect.isclass)
        if cls.__module__.startswith("shapeaxi")
    ]
    allowed.append(transforms.Compose)
    torch.serialization.add_safe_globals(allowed)


def load_network(checkpoint: str, network_name: str, device: str):
    """The shapeaxi network named by the checkpoint's row in the catalog.

    Resolved from `saxi_nets_lightning`, which is where these two classes live.
    """
    from shapeaxi import saxi_nets_lightning

    # The name check first: a catalog that disagrees with the installed
    # shapeaxi is a startup-shaped mistake, and saying so costs no filesystem.
    network_class = getattr(saxi_nets_lightning, network_name, None)
    if network_class is None:
        raise ValueError(
            f"shapeaxi {_shapeaxi_version()} has no `{network_name}` in "
            f"saxi_nets_lightning. The checkpoint catalog and the installed "
            f"shapeaxi disagree."
        )
    check_backbone_is_staged()
    allow_checkpoint_globals()
    from . import embree_raster

    if embree_raster.wanted(device):
        use_embree_and_cdist(saxi_nets_lightning)
    model = network_class.load_from_checkpoint(checkpoint, strict=False)
    model.eval()
    model.to(device)
    return model


def use_embree_and_cdist(saxi_nets_lightning) -> None:
    """Swap the three pytorch3d pieces that have no Metal kernel, in this process.

    Only reached on macOS (`embree_raster.wanted`). shapeaxi looks these
    names up in its own module namespaces at call time, so rebinding them is
    enough and shapeaxi's files are untouched:

    * `MeshRasterizer`, which `SaxiMHAFBClassification` and `...Regression` use
      to build their renderer: the Embree subclass.
    * `sample_points_from_meshes`, a compiled op that refuses any tensor not on
      the CPU: run on the CPU (7 ms for a scan), results moved back.
    * `knn_points` in `MHA_KNN` (256 channels, which the compiled kernel
      handles in 1.7 s on the CPU): `torch.cdist` plus `topk`, 24 ms, the same
      indices on 99.99 % of the entries.
    """
    if not hasattr(saxi_nets_lightning, "MeshRasterizer"):
        # Not a shapeaxi renderer module (the tests stand a stub in its place):
        # nothing to replace, and the rendering stays as it is.
        logger.warning("shapeaxi has no MeshRasterizer to replace; rendering unchanged")
        return

    from shapeaxi import saxi_layers

    from . import embree_raster

    saxi_nets_lightning.MeshRasterizer = embree_raster.rasterizer_class()
    if not hasattr(saxi_nets_lightning.sample_points_from_meshes, "_sadt_wrapped"):
        saxi_nets_lightning.sample_points_from_meshes = _on_cpu(
            saxi_nets_lightning.sample_points_from_meshes
        )
        saxi_layers.knn_points = _knn_by_cdist(saxi_layers.knn_points)
    logger.info("Rasterizing with Embree, knn with cdist, sampling on the CPU")


def _to(value, device):
    if hasattr(value, "_fields"):  # a namedtuple, as pytorch3d's knn returns
        return type(value)(*(_to(item, device) for item in value))
    if isinstance(value, (tuple, list)):
        return type(value)(_to(item, device) for item in value)
    return value.to(device) if hasattr(value, "to") else value


def _on_cpu(original):
    def sample_points_from_meshes(meshes, *args, **kwargs):
        if meshes.device.type == "cpu":
            return original(meshes, *args, **kwargs)
        return _to(original(meshes.to("cpu"), *args, **kwargs), meshes.device)

    sample_points_from_meshes._sadt_wrapped = True
    return sample_points_from_meshes


def _knn_by_cdist(original):
    def knn_points(p1, p2, lengths1=None, lengths2=None, norm=2, K=1, version=-1,
                   return_nn=False, return_sorted=True):
        import torch
        from pytorch3d.ops.knn import _KNN

        plain = (lengths1 is None and lengths2 is None and norm == 2 and not return_nn
                 and K <= p2.shape[1])
        if plain and p1.dtype == torch.float32:
            squared = torch.cdist(p1, p2).pow(2)
            dists, idx = squared.topk(K, dim=-1, largest=False, sorted=True)
            return _KNN(dists=dists, idx=idx, knn=None)
        result = original(p1.cpu(), p2.cpu(), lengths1, lengths2, norm, K, version,
                          return_nn, return_sorted)
        return _to(result, p1.device)

    return knn_points


def _shapeaxi_version() -> str:
    import importlib.metadata as metadata

    try:
        return metadata.version("shapeaxi")
    except metadata.PackageNotFoundError:  # pragma: no cover
        return "an unknown version"


def scale_attribution(attribution, target_size=GRADCAM_IMAGE_SIZE):
    """One batch of attribution maps, resized and normalised to [-1, 1].

    The arithmetic is upstream's, which took it from pytorch-grad-cam: resize
    to the renderer's frame, clip to the 1st and 99th percentiles so one hot
    pixel does not flatten the rest, then rescale.

    The resize is torch's rather than `cv2.resize`. Bilinear with
    `align_corners=False` is the half-pixel convention `INTER_LINEAR` uses, so
    it is the same arithmetic -- and it does not pull in a second array library
    whose every published wheel caps numpy below what the rest of this stack
    needs. `test_run.py` pins the two against each other.
    """
    import numpy as np
    import torch

    stack = torch.as_tensor(np.ascontiguousarray(attribution), dtype=torch.float32)
    if stack.ndim == 2:
        stack = stack[None]
    resized = torch.nn.functional.interpolate(
        stack[:, None], size=tuple(target_size), mode="bilinear", align_corners=False
    )[:, 0].numpy()

    scaled = []
    for image in resized:
        low = np.percentile(image.flatten(), q=CLIP_LOW)
        high = np.percentile(image.flatten(), q=CLIP_HIGH)
        image = np.clip(image, low, high)
        spread = np.max(image) - np.min(image)
        if spread > 0:
            image = 2 * ((image - np.min(image)) / spread) - 1
        else:
            # A flat map normalises to zeros rather than to NaNs. Upstream
            # divided unconditionally, and wrote the NaNs onto the surface.
            image = np.zeros_like(image)
        scaled.append(image)
    return np.float32(scaled)
