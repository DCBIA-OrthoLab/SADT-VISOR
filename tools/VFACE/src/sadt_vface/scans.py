"""Scan file naming, copied from the server's file_utils.

Copied rather than shared, like every other helper here: see CONTRIBUTING.md on
why there is no sadt-core package. The compound extensions are the reason this
is not `os.path.splitext` -- `.nii.gz` has to survive as one unit.
"""

import os

SCAN_EXTENSIONS = (".nii.gz", ".nrrd.gz", ".gipl.gz", ".nii", ".nrrd", ".gipl")

# The compressed spelling ITK can WRITE for each scan extension. NIfTI and GIPL
# take an external .gz; NRRD compresses inside the file and ITK has no
# ".nrrd.gz" writer at all, so that spelling maps back down to ".nrrd".
_COMPRESSED_EXTENSIONS = {".nii": ".nii.gz", ".gipl": ".gipl.gz", ".nrrd.gz": ".nrrd"}


def split_scan_extension(filename: str) -> tuple:
    """('scan.nii.gz') -> ('scan', '.nii.gz'), compound extensions preserved."""
    lower = filename.lower()
    for extension in SCAN_EXTENSIONS:
        if lower.endswith(extension):
            return filename[: -len(extension)], filename[-len(extension):]
    return os.path.splitext(filename)


def compressed_extension(extension: str) -> str:
    """The compressed spelling ITK can write for a scan extension."""
    return _COMPRESSED_EXTENSIONS.get(extension.lower(), extension)
