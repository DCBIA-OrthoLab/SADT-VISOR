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

from . import progress
from .errors import ToolInputError, ToolUnavailableError

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


def check_surfaces(found: list) -> None:
    """Refuse the batch when a surface has no points or no faces.

    Read with the reader shapeaxi itself uses for `.vtk`. A file it cannot
    parse comes back as an empty polydata rather than as an exception, and the
    renderer then fails several calls deeper with a tensor-shape error that
    says nothing about which surface, or why. Checked up front, by position,
    so the message names the item without naming the file.
    """
    import vtk

    total = len(found)
    for index, path in enumerate(found, start=1):
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(path)
        reader.Update()
        surface = reader.GetOutput()
        points = surface.GetNumberOfPoints() if surface is not None else 0
        faces = surface.GetNumberOfPolys() if surface is not None else 0
        if points == 0 or faces == 0:
            raise ToolInputError(
                f"'surfaces': surface {index} of {total} is empty or unreadable "
                f"({points} points, {faces} faces); DOCShapeAXI needs a "
                f"triangle mesh in legacy VTK polydata format."
            )


def check_unique_names(found: list) -> None:
    """Refuse surfaces that share a file name across subfolders.

    Each explained surface is written flat into the output folder under its
    own file name, and the outputs mapping is keyed by that name's stem, so two
    same-named surfaces from different subfolders would silently overwrite
    each other. Only the count is reported: the names are patient data.
    """
    seen = {}
    for path in found:
        name = os.path.basename(path)
        seen[name] = seen.get(name, 0) + 1
    clashing = sum(count for count in seen.values() if count > 1)
    if clashing:
        raise ToolInputError(
            f"'surfaces': {clashing} of {len(found)} surfaces share a file name "
            f"with another surface in a different subfolder, and their "
            f"explained outputs would overwrite each other. Give each surface "
            f"a unique file name, or run with explain off."
        )


def resolve_device(requested: str) -> str:
    torch = import_torch()
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("cuda requested but no CUDA device is visible; running on cpu")
        progress.log(
            "a GPU was requested but none is visible; running on the CPU", "warning"
        )
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
    # ToolUnavailableError, not FileNotFoundError: the server reads the
    # latter as the caller's fault, and no request can stage a file here.
    raise ToolUnavailableError(
        f"The EfficientNet backbone '{BACKBONE_FILE}' is not staged on this "
        f"server. shapeaxi builds it around every checkpoint and would "
        f"otherwise download it mid-request, which a server holding patient data must not do. Stage "
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


def load_network(checkpoint: str, network_name: str, device: str,
                 anatomy: str = "the requested analysis"):
    """The shapeaxi network named by the checkpoint's row in the catalog.

    Resolved from `saxi_nets_lightning`, which is where these two classes live.
    """
    from shapeaxi import saxi_nets_lightning

    # The name check first: a catalog that disagrees with the installed
    # shapeaxi is a startup-shaped mistake, and saying so costs no filesystem.
    network_class = getattr(saxi_nets_lightning, network_name, None)
    if network_class is None:
        # RuntimeError, not ValueError: this is the server's installation
        # disagreeing with itself, and nothing the caller sends will fix it.
        raise RuntimeError(
            f"shapeaxi {_shapeaxi_version()} has no `{network_name}` in "
            f"saxi_nets_lightning. The checkpoint catalog and the installed "
            f"shapeaxi disagree."
        )
    check_backbone_is_staged()
    allow_checkpoint_globals()
    # Said before the call: a failed load's own message is a long Lightning or
    # pickle traceback whose useful part is its last line, and the operator
    # needs to know what was being attempted to read it.
    logger.info("loading %s checkpoint for %s on %s", network_name, anatomy, device)
    try:
        model = network_class.load_from_checkpoint(checkpoint, strict=False)
        model.eval()
        model.to(device)
    except Exception as exc:
        raise RuntimeError(
            f"loading the {network_name} checkpoint on {device} failed: "
            f"{last_line(exc)}"
        ) from exc
    return model


def last_line(exc: BaseException) -> str:
    """The last non-empty line of an exception's message, with its class.

    Third-party loaders put the actual cause at the end of a long message,
    and the server cuts long reasons from the end.
    """
    lines = [line.strip() for line in str(exc).splitlines() if line.strip()]
    return f"{type(exc).__name__}: {lines[-1] if lines else ''}".rstrip(": ")


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
