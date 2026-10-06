"""Mask R-CNN over a CBCT, one axial slice at a time.

Ported from `CLIC/runner/clic_runner.py`. The network is torchvision's
`maskrcnn_resnet50_fpn` with its two heads resized to the checkpoint's class
count, applied to each slice of the volume as a 3-channel image; the detections
that clear a score threshold are painted into a label volume that keeps the
input's affine and header, so the two open aligned in a viewer.

Four things differ from upstream, each because a server runs this rather than a
laptop with someone watching:

* **the checkpoint is named, not guessed.** Upstream took
  `sorted(model_dir.glob("*.pth"))[0]` -- the alphabetically first file in the
  folder, with no check that it was the only one and an IndexError when the
  folder was empty. Which model vintage ran must never be a surprise.
* **the class count is read from the checkpoint**, not hardcoded to 4. The
  head's shape is in the state dict; a checkpoint trained on a different number
  of classes fails on `load_state_dict` with a shape mismatch nobody can read,
  where here it simply works.
* **a folder is walked recursively, and files are what is returned.** Upstream's
  collector returned SUBDIRECTORIES whenever any of them held a scan, and the
  runner then called `nib.load` on a directory. It also advertised `.nrrd`,
  `.mha` and `.mhd`, none of which nibabel reads -- accepted and then ignored,
  the trap `.stl` fell into in ALI.
* **one scan failing costs one scan.** Upstream ran one process per scan from
  the Slicer module, so a crash ended that scan; here the loop is inside, and
  it has to be told to keep going.
"""

import logging
import os

# Named after the module, not the tool: the server shows the records of the
# loggers under this package and nothing else, so a logger called "CLIC" was
# written for nobody.
logger = logging.getLogger(__name__)

# What nibabel actually reads, which is the list this tool advertises.
SCAN_EXTENSIONS = (".nii", ".nii.gz")

# Detections below this score are not painted. Upstream hardcoded 0.7; it is an
# argument here because it is the one knob that moves the segmentation, and a
# caller comparing runs needs to be able to say which value produced which.
DEFAULT_SCORE_THRESHOLD = 0.7

# Above this, a mask pixel counts as inside the object.
MASK_BINARY_THRESHOLD = 0.5


def import_torch():
    """torch, imported late.

    CI imports this package on every pull request to publish its schema, and
    that must not cost a CUDA stack.
    """
    import torch

    return torch


def discover_scans(input_path: str) -> list:
    """Every volume under `input_path`, sorted, as file paths.

    A single file is returned as itself. A folder is walked RECURSIVELY --
    upstream looked one level down, and returned the directories rather than
    the files they held.
    """
    if os.path.isfile(input_path):
        return [input_path] if is_scan_file(input_path) else []

    found = []
    for directory, _subdirs, names in os.walk(input_path):
        for name in sorted(names):
            if name.startswith("."):
                continue
            if is_scan_file(name):
                found.append(os.path.join(directory, name))
    return sorted(found)


def is_scan_file(name: str) -> bool:
    return name.lower().endswith(SCAN_EXTENSIONS)


def resolve_device(requested: str) -> str:
    """`requested`, or "cpu" when no card is visible.

    A tool that asks for cuda on a machine without one should run, not raise:
    the server injects its own DEVICE when the caller names none, and that
    setting describes the deployment rather than this request.
    """
    torch = import_torch()
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("cuda requested but no CUDA device is visible; running on cpu")
        return "cpu"
    return requested


def class_count(state_dict) -> int:
    """How many classes the checkpoint's box predictor was trained for.

    Upstream built the heads for 4 classes whatever the file held, so a
    checkpoint trained on a different count died inside `load_state_dict` on a
    shape mismatch. The count is in the weights: the classification head's bias
    has one entry per class, background included.
    """
    for key in ("roi_heads.box_predictor.cls_score.bias",
                "roi_heads.box_predictor.cls_score.weight"):
        if key not in state_dict:
            continue
        shape = tuple(int(size) for size in state_dict[key].shape)
        if not shape or shape[0] < 1:
            # A present but malformed entry used to reach `shape[0]` and raise
            # `IndexError: tuple index out of range`, which names neither the
            # key nor the checkpoint. The reason a file cannot be loaded has to
            # survive as far as the caller.
            raise ValueError(
                f"`{key}` has shape {shape} in this checkpoint, so no class "
                f"count can be read from it: a Mask R-CNN box predictor's "
                f"classification head has one entry per class."
            )
        return shape[0]
    raise ValueError(
        "This checkpoint has no `roi_heads.box_predictor.cls_score` entry, so it "
        "is not a Mask R-CNN box predictor this tool can load."
    )


def build_model(checkpoint_path: str, device: str):
    """The network, with its heads resized to the checkpoint's class count."""
    torch = import_torch()
    from torchvision.models.detection import maskrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    # A file that is not there stays the OSError it is: a wrong name is the
    # caller's to fix, and the server answers it as such. A file that IS there
    # and cannot be used is the deployment's fault -- the checkpoint is
    # installed by whoever runs the server -- so everything past this point is
    # a RuntimeError, never the ValueError that would blame the request.
    try:
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except OSError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"checkpoint could not be read ({type(exc).__name__}: "
            f"{_first_line(exc)})"
        ) from exc
    try:
        classes = class_count(state)
    except ValueError as exc:
        raise RuntimeError(f"checkpoint is unusable: {exc}") from exc

    model = maskrcnn_resnet50_fpn(weights=None)
    box_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(box_features, classes)
    mask_features = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_features, 256, classes)

    try:
        model.load_state_dict(state)
    except RuntimeError as exc:
        # torch lists every mismatched tensor, one per line, and the operator
        # sees only the first few hundred characters: the cause goes first.
        raise RuntimeError(
            f"checkpoint weights do not fit the Mask R-CNN layout "
            f"({_first_line(exc)})"
        ) from exc
    model.to(device).eval()
    return model, classes


def _first_line(exc) -> str:
    """The first non-empty line of an exception's message, which is the cause."""
    for line in str(exc).splitlines():
        if line.strip():
            return line.strip()
    return ""


def normalise(slice_2d):
    """One slice to [0, 1]. A flat slice becomes zeros rather than NaNs."""
    import numpy as np

    low, high = slice_2d.min(), slice_2d.max()
    if high - low <= 1e-8:
        return np.zeros_like(slice_2d)
    return (slice_2d - low) / (high - low)


def segment_volume(model, volume, device: str, score_threshold: float):
    """The label volume for one scan, painted slice by slice.

    Returns (labels, detections) -- the array, and how many detections cleared
    the threshold across the whole volume, which is what tells a caller the
    scan was actually seen rather than silently returning zeros.
    """
    import numpy as np

    torch = import_torch()
    labels = np.zeros(volume.shape, dtype=np.int16)
    detections = 0

    for index in range(volume.shape[2]):
        image = normalise(volume[..., index])
        tensor = (
            torch.from_numpy(image).unsqueeze(0).repeat(3, 1, 1).float().to(device)
        )
        with torch.no_grad():
            predicted = model([tensor])[0]

        keep = predicted["scores"] >= score_threshold
        masks = (predicted["masks"][keep] > MASK_BINARY_THRESHOLD).squeeze(1)
        for mask, label in zip(masks.cpu().numpy(), predicted["labels"][keep].cpu().numpy()):
            labels[..., index][mask] = int(label)
            detections += 1

    return labels, detections
