"""Crop both temporomandibular joints of a CBCT into fixed-size volumes.

The whole public surface is `run`. The orientation comes from ASO (Frankfurt
horizontal frame, landmarks from ALI_CBCT) and the mandible from AMASSS, both
reached through the supervisor; this package adds the crop itself. See
`dispatch.py` for the chain and `crop.py` for the geometry.

Heavy imports live inside the functions that need them: `scripts/describe.py`
imports this module in every CI job.
"""

from pathlib import Path
from typing import Literal

from .dispatch import crop_scans


def run(
    scans: Path,
    output_dir: Path,
    sides: Literal["Left and Right", "Left", "Right"] = "Left and Right",
    # The same argument as AREG and ASO, so every tool offers one "Extra
    # outputs" box. Nothing ticked by default: both are the size of the scan.
    outputs: list[Literal["Oriented scan", "Mandible segmentation"]] = [],
    scale_by: Literal["none", "mandible-volume"] = "none",
    scale: float = 0.0,
    *,
    sup=None,
) -> Path:
    """Crop the temporomandibular joints of a CBCT.

    Each scan is oriented onto the Frankfurt horizontal frame, its mandible is
    segmented, and a fixed-size box is cut around each condylar head. The left
    joint is mirrored so every joint has the right-side orientation. The oriented
    scan and the mandible segmentation can be returned as well.

    Args:
        scans: One CBCT (.nrrd, .nii.gz and the other formats ASO reads), or a
            folder of them for a batch.
        output_dir: Where results are written. Per scan: `<scan>_R.nii.gz` and
            `<scan>_L.nii.gz` (the crops), `<scan>.json` (the report) and
            `<scan>_qc.png` (a quick view); only the sides asked for are written.
        sides: Which joint to crop: `Left and Right`, `Left` or `Right`.
        outputs: What to hand back beside the crops, in the oriented frame they
            were cut from. Oriented scan: the whole scan, as
            `<scan>_oriented.nii.gz`. Mandible segmentation: the mandible mask
            (AMASSS), as `<scan>_mandible.nii.gz`.
        scale_by: `none` keeps the box at 28.8 x 33.6 x 24.0 mm. With
            `mandible-volume` the box follows the cube root of the mandible
            volume, the voxel count staying the same.
        scale: A fixed scale factor, overriding `scale_by`; 0 means not given.
            Useful to reuse one patient's first time point for the next.

    Returns:
        The output directory.
    """
    return crop_scans(
        scans=scans, output_dir=output_dir, scale_by=scale_by, scale=scale, sup=sup,
        sides=sides, outputs=outputs,
    )
