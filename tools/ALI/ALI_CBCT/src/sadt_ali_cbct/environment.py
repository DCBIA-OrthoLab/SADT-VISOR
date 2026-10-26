"""The volume an agent navigates: one scan, loaded at every scale.

Ported from ALI_CBCT_utils/environment.py, inference only. Everything the
original carried for training -- loading ground-truth fiducials, sampling
random poses around a known landmark, computing rewards -- is gone; so is the
`sys.version_info >= (3, 10)` branch choosing between monai's
`EnsureChannelFirst` and the `AddChannel` transform removed from monai years
ago. This package pins monai 1.6.0, where only the former exists.
"""

import logging

import numpy as np
import SimpleITK as sitk

from .errors import ToolUnavailableError

from .brain import import_torch

logger = logging.getLogger(__name__)

_MONAI_HINT = "ALI's CBCT engine needs monai. Run `uv sync` in tools/ALI."


def import_transforms():
    try:
        from monai.transforms import BorderPad, Compose, EnsureChannelFirst, ScaleIntensity, SpatialCrop
    except ImportError as exc:  # pragma: no cover - depends on the deployment
        raise ToolUnavailableError(f"{_MONAI_HINT} (missing: monai)") from exc
    return BorderPad, Compose, EnsureChannelFirst, ScaleIntensity, SpatialCrop


class Environment:
    """One patient's scan at every scale, plus the landmarks found in it.

    The agent walks in VOXEL coordinates of the current scale. `padding` (half
    the agent's field of view) is added around the volume so a field of view
    centred near an edge is still a full box, and `physical_position` converts
    a voxel position back to the scanner's own millimetre frame at the end.
    """

    def __init__(self, patient_id: str, padding, device: str):
        BorderPad, Compose, EnsureChannelFirst, ScaleIntensity, SpatialCrop = import_transforms()

        self.patient_id = patient_id
        self.padding = np.asarray(padding).astype(np.int16)
        self.device = device
        self._crop = SpatialCrop
        self._rescale = ScaleIntensity(minv=-1.0, maxv=1.0, factor=None)
        self._transform = Compose(
            [
                EnsureChannelFirst(channel_dim="no_channel"),
                BorderPad(spatial_border=self.padding.tolist()),
            ]
        )
        self.data = {}
        self.predicted_landmarks = {}

    @property
    def scale_count(self) -> int:
        return len(self.data)

    def load_images(self, images_per_scale: dict) -> None:
        """Read the resampled volume for every scale into memory."""
        torch = import_torch()
        for scale_key, path in images_per_scale.items():
            image = sitk.ReadImage(path)
            array = sitk.GetArrayFromImage(image)
            self.data[scale_key] = {
                "path": path,
                "image": self._transform(array).to(dtype=torch.int16),
                "spacing": np.array(image.GetSpacing()),
                # Kept whole, in SimpleITK's (x, y, z) order: physical_position
                # needs the direction as well as the origin.
                "origin": np.array(image.GetOrigin()),
                "direction": np.array(image.GetDirection()).reshape(3, 3),
                "size": np.array(np.shape(array)),
            }

    def release(self) -> None:
        """Free the volumes. A cohort would not fit in memory otherwise --
        each scan is held at two spacings, the finer one being the larger."""
        self.data = {}

    def spacing(self, scale_key: str):
        return self.data[scale_key]["spacing"]

    def size(self, scale_key: str):
        return self.data[scale_key]["size"]

    def field_of_view(self, scale_key: str, center, crop_size):
        """The rescaled cube of voxels the agent currently sees.

        The centre is shifted by `padding` because `load_images` padded the
        volume: a position in agent coordinates sits `padding` voxels further
        in inside the padded array. (The original expressed this as
        `center.tolist() + self.padding`, which reads like list concatenation
        and is in fact numpy broadcasting the addition.)
        """
        torch = import_torch()
        crop = self._crop((np.asarray(center) + self.padding).tolist(), crop_size)
        return self._rescale(crop(self.data[scale_key]["image"])).type(torch.float32)

    def add_predicted_landmark(self, label: str, position) -> None:
        self.predicted_landmarks[label] = position

    def physical_position(self, scale_key: str, position):
        """A voxel position at `scale_key`, in the scanner's LPS millimetres.

        `position` is in array order (z, y, x); the value returned is (x, y, z),
        the order a markups file expects. The full index-to-physical mapping,
        origin + direction @ (spacing * index): the original computed
        `(position - abs(origin / spacing)) * spacing`, which is right only for
        an identity direction AND an all-negative origin. A scan stored with
        its Y axis flipped -- direction diag(1, -1, 1), origin y positive --
        came out mirrored front to back, every landmark in the air.
        """
        reference = self.data[scale_key]
        index = np.asarray(position, dtype=float)[::-1]
        return reference["origin"] + reference["direction"] @ (reference["spacing"] * index)
