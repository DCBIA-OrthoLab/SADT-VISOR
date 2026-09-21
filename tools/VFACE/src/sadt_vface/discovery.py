"""Finding the volumes of a cohort, in a stable order.

`scans.py` beside this one is a vendored COPY and has to stay byte-identical
across every tool that carries it -- it names the extensions a caller's data is
recognised by, and a copy that drifted would accept a scan its neighbour
refuses. So what is VFACE's own lives here instead of being added to it.

Walking a cohort matters more here than in most tools. VFACE hands its scans to
six others and reads back what they wrote, so an unordered walk does not just
reorder a report: wherever a step keeps "the first" of something, two runs on
one folder give two different answers.
"""

import os

from .scans import SCAN_EXTENSIONS


def is_scan_file(filename: str) -> bool:
    return filename.lower().endswith(SCAN_EXTENSIONS)


def find_scans(root: str) -> list:
    """Every volume under `root`, sorted, dotfiles left out.

    The subdirectories are sorted IN PLACE as well as the names: `os.walk`
    visits them in `os.listdir` order, which is arbitrary and differs between
    filesystems.
    """
    found = []
    for directory, subdirectories, names in os.walk(root or ""):
        subdirectories.sort()
        for name in sorted(names):
            if not name.startswith(".") and is_scan_file(name):
                found.append(os.path.join(directory, name))
    return found


def find_by_extension(root: str, extensions) -> list:
    """Every file under `root` whose name ends in one of `extensions`, sorted.

    The same walk for the files that are not volumes: the transforms beside an
    oriented scan, the surfaces a heat map is drawn on.
    """
    extensions = tuple(extension.lower() for extension in extensions)
    found = []
    for directory, subdirectories, names in os.walk(root or ""):
        subdirectories.sort()
        for name in sorted(names):
            if not name.startswith(".") and name.lower().endswith(extensions):
                found.append(os.path.join(directory, name))
    return found
