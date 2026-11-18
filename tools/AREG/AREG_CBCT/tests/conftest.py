"""Fixtures for AREG_CBCT's suite: phantoms, masks, and a supervisor.

`itk-elastix` is a wheel and fast enough on a 48^3 phantom that the
registration itself is exercised rather than mocked -- which is what makes the
centre-of-rotation correction, the masked-image correction and the direction of
the written transform testable at all.

It is not free, though: a full registration is a second or two. The tests that
only want to look at what a run PRODUCED share one module-scoped run rather
than each paying for their own.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk


# ---------------------------------------------------------------------------
# Volumes
# ---------------------------------------------------------------------------

def phantom(size=48, seed=0, spacing=0.8, origin=(-140.0, -90.0, 60.0)):
    """A textured volume with an origin far from zero.

    Far from zero on purpose: that is the condition under which elastix's
    centre of rotation matters, and a phantom centred on the origin would let
    the bug this suite pins pass unnoticed.
    """
    rng = np.random.default_rng(seed)
    volume = rng.random((size,) * 3).astype(np.float32) * 120
    zz, yy, xx = np.meshgrid(*[np.arange(size)] * 3, indexing="ij")
    half = size // 2
    volume += 1400 * (
        ((zz - half) ** 2 / 180 + (yy - half + 2) ** 2 / 140 + (xx - half - 2) ** 2 / 160) < 1
    )
    volume += 900 * (
        ((zz - half + 12) ** 2 / 40 + (yy - half - 10) ** 2 / 35 + (xx - half + 12) ** 2 / 30) < 1
    )
    image = sitk.GetImageFromArray(volume)
    image.SetSpacing((spacing,) * 3)
    image.SetOrigin(origin)
    return image


def displaced(image, rotation=(0.04, -0.025, 0.03), translation=(1.2, -1.6, 0.9)):
    """`image` displaced by a known rigid transform, and that transform."""
    truth = sitk.Euler3DTransform()
    size = np.array(image.GetSize()) / 2.0
    truth.SetCenter(image.TransformContinuousIndexToPhysicalPoint(size.tolist()))
    truth.SetRotation(*rotation)
    truth.SetTranslation(translation)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image)
    resampler.SetTransform(truth.GetInverse())
    resampler.SetInterpolator(sitk.sitkLinear)
    return resampler.Execute(image), truth


def write(image, path):
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    sitk.WriteImage(image, str(path), useCompression=True)
    return str(path)


def full_mask(image):
    mask = sitk.GetImageFromArray(np.ones(sitk.GetArrayViewFromImage(image).shape, np.uint8))
    mask.CopyInformation(image)
    return mask


def tree_of(root):
    return sorted(
        os.path.relpath(os.path.join(directory, name), str(root))
        for directory, _subdirs, names in os.walk(str(root))
        for name in names
    )


# ---------------------------------------------------------------------------
# A cohort, and a run over it
# ---------------------------------------------------------------------------

def cohort(root, subjects=("P1",), regions=("CB",), size=48, extension=".nii.gz"):
    """A Semi-Automated request: T1, T2 and one mask per subject per region."""
    for subject in subjects:
        fixed = phantom(size=size, seed=abs(hash(subject)) % 100)
        moving, _truth = displaced(fixed)
        write(fixed, os.path.join(str(root), "T1", f"{subject}_T1_scan{extension}"))
        write(moving, os.path.join(str(root), "T2", f"{subject}_T2_scan{extension}"))
        for region in regions:
            write(
                full_mask(fixed),
                os.path.join(str(root), "masks", f"{subject}_T1_{region}_seg{extension}"),
            )
    return str(root)


@pytest.fixture(scope="module")
def semi_run(tmp_path_factory):
    """One Semi-Automated registration of one subject, shared by module.

    elastix on a 48^3 phantom is a second or two; the tests that only read what
    a run produced have no reason to each pay for their own.
    """
    from sadt_areg_cbct import dispatch
    from sadt_areg_common import catalogs

    root = tmp_path_factory.mktemp("semi")
    cohort(root)
    run = dispatch.register(
        t1_path=os.path.join(str(root), "T1"),
        t2_path=os.path.join(str(root), "T2"),
        t1_masks_path=os.path.join(str(root), "masks"),
        automation=catalogs.AUTOMATION_SEMI,
        regions=["Cranial base"],
        output_dir=os.path.join(str(root), "out"),
    )
    return run, str(root)


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
        self.spans = []
        self.messages = []
        self.logs = []

    def run(self, tool, **params):
        # The caller's span of its own bar. The server's supervisor removes it
        # before the callee sees it, so it is recorded apart, never passed on.
        self.spans.append((tool, params.pop("_progress", None)))
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

    def log(self, message, level="info", user=False):
        self.messages.append((None, message))
        self.logs.append((level, user, message))

    def asked(self, tool):
        """The parameters of the one call to `tool`."""
        matching = [params for name, params in self.calls if name == tool]
        assert len(matching) == 1, f"{tool} was called {len(matching)} time(s)"
        return matching[0]
