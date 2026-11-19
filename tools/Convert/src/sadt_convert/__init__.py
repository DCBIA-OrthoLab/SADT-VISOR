"""Convert 3D scans between file formats. The whole public surface is `run`."""

from pathlib import Path
from typing import Literal

from .convert import convert_tree


def run(
    scans: Path,
    output_dir: Path,
    to: Literal["nii.gz", "nrrd", "gipl.gz"] = "nii.gz",
) -> Path:
    """Convert scans, or DICOM series, to one file format.

    Args:
        scans: One scan, or a folder of scans and DICOM series. Folders are kept as they are.
        output_dir: Where the converted scans are written.
        to: The format to write: NIfTI, NRRD or GIPL.

    Returns:
        The folder holding the converted scans.
    """
    return convert_tree(Path(scans), Path(output_dir), to)
