"""Fixtures for VFACE's suite: real volumes, and a supervisor that is not there.

VFACE drives six other tools, so most of what it does is `sup.run(...)` -- and
those are stood in for by `FakeSup`, which is all a tool can ever see of a
supervisor: `run`, `progress`, `log`, `tmp`, `out`. Nothing is imported across
virtualenvs, here or in the server.

What is NOT stubbed is anything VFACE computes itself: the resample, the
mirroring, the measurements and the classifier all run for real, on volumes
written to disk by SimpleITK.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Volumes
# ---------------------------------------------------------------------------

def write_volume(path, size=(24, 20, 16), spacing=(0.6, 0.6, 0.9),
                 origin=(-10.0, -8.0, -6.0), value=None):
    """A small CBCT, readable by SimpleITK.

    Anisotropic and coarser than the 0.3 mm the resample targets, because that
    is what a real acquisition looks like and what makes the resample do
    something a test can measure.
    """
    import SimpleITK as sitk

    if value is None:
        # A gradient, not a constant: interpolation on a constant volume is
        # indistinguishable from doing nothing at all.
        grids = np.meshgrid(*(np.arange(n, dtype=np.float32) for n in size), indexing="ij")
        value = 100.0 * grids[0] + 10.0 * grids[1] + grids[2] - 1000.0

    image = sitk.GetImageFromArray(np.asarray(value, dtype=np.float32).transpose(2, 1, 0))
    image.SetSpacing(tuple(float(v) for v in spacing))
    image.SetOrigin(tuple(float(v) for v in origin))

    os.makedirs(os.path.dirname(str(path)) or ".", exist_ok=True)
    sitk.WriteImage(image, str(path))
    return str(path)


def read_volume(path):
    import SimpleITK as sitk

    return sitk.ReadImage(str(path))


def cohort(tmp_path, patients=("P001", "P002"), timepoint="T1", **volume):
    """A folder of CBCTs named the way the pipeline expects."""
    for patient in patients:
        write_volume(tmp_path / "t1" / f"{patient}_{timepoint}.nii.gz", **volume)
    return str(tmp_path / "t1")


def tree_of(root):
    return sorted(
        os.path.relpath(os.path.join(directory, name), str(root))
        for directory, _subdirs, names in os.walk(str(root))
        for name in names
    )


# ---------------------------------------------------------------------------
# The supervisor
# ---------------------------------------------------------------------------

class FakeSup:
    """A supervisor, as a tool sees one. Records what it was asked for.

    `outputs` maps a tool name to a callable taking the parameters it was sent
    and returning the directory it "produced", so a test can plant results
    without any of the real tools existing.
    """

    def __init__(self, tmp_path, outputs=None):
        self.out = Path(tmp_path) / "out"
        self.tmp = Path(tmp_path) / "tmp"
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.outputs = outputs or {}
        self.calls = []
        self.messages = []

    def run(self, tool, **params):
        self.calls.append((tool, params))
        maker = self.outputs.get(tool)
        if maker is None:
            raise AssertionError(f"nothing planted for {tool!r} in this test")
        produced = maker(params)
        if isinstance(produced, Exception):
            raise produced
        return Path(produced)

    def progress(self, fraction, message):
        self.messages.append((fraction, message))

    def log(self, message):
        self.messages.append((None, message))

    def asked(self, tool):
        """The parameters of the one call to `tool`."""
        matching = [params for name, params in self.calls if name == tool]
        assert len(matching) == 1, f"{tool} was called {len(matching)} time(s)"
        return matching[0]


@pytest.fixture
def report():
    return {"patients": {}}
