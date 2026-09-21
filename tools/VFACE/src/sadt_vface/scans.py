"""Scan file naming.

Copied from the sibling tools rather than shared -- see CONTRIBUTING.md on why
there is no sadt-core package. The compound extensions are the reason this is
not `os.path.splitext`: `.nii.gz` has to survive as one unit.

The table matters more here than in most tools, because VFACE hands its scans
to six others and reads back what they wrote. A tool that disagreed about
whether `.nrrd` is a volume would drop a cohort silently at whichever step
looked first -- which is what happened upstream when the resample step accepted
only `.nii` and every later step then reported "0 file" on a folder that was
not empty.
"""

import os

SCAN_EXTENSIONS = (".nii.gz", ".nrrd.gz", ".gipl.gz", ".nii", ".nrrd", ".gipl")


def split_scan_extension(filename: str) -> tuple:
    """`'scan.nii.gz'` -> `('scan', '.nii.gz')`, compound extensions preserved."""
    lower = filename.lower()
    for extension in SCAN_EXTENSIONS:
        if lower.endswith(extension):
            return filename[: -len(extension)], filename[-len(extension):]
    return os.path.splitext(filename)


def is_scan_file(filename: str) -> bool:
    return filename.lower().endswith(SCAN_EXTENSIONS)


def find_scans(root: str) -> list:
    """Every volume under `root`, in a stable order.

    Sorted, and the directories sorted in place: readdir order varies between
    filesystems, and an unordered batch makes two runs on the same folder
    produce differently ordered reports -- and, where a step keeps "the first"
    of something, two different answers.
    """
    found = []
    for directory, subdirectories, names in os.walk(root):
        subdirectories.sort()
        for name in sorted(names):
            if not name.startswith(".") and is_scan_file(name):
                found.append(os.path.join(directory, name))
    return found
