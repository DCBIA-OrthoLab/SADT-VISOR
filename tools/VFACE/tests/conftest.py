"""Fixtures for VFACE's suite: real volumes, and a supervisor that is not there.

VFACE drives six other tools, so most of what it does is `sup.run(...)` -- and
those are stood in for by `FakeSup`, which is all a tool can ever see of a
supervisor: `run`, `progress`, `log`, `tmp`, `out`. Nothing is imported across
virtualenvs, here or in the server.

What is NOT stubbed is anything VFACE computes itself: the resample, the
mirroring, the measurements and the classifier all run for real, on volumes
written to disk by SimpleITK.
"""

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
        self.spans = []
        self.messages = []
        self.logs = []

    def run(self, tool, **params):
        # A slot per CALL, which is what the real supervisor gives
        # (`<job>/sup/<NN>_<tool>/output`) and why no caller passes `output_dir`
        # any more. Numbered, so the two orientations and the three
        # registrations land apart without anyone naming a directory.
        # `_progress` is the caller's span of its own bar; the server's
        # supervisor removes it before the callee sees it, so it is recorded
        # apart and never handed to a planted tool.
        self.spans.append((tool, params.pop("_progress", None)))
        self.calls.append((tool, params))
        slot = self.tmp / "sup" / f"{len(self.calls):02d}_{tool}" / "output"
        slot.mkdir(parents=True, exist_ok=True)
        maker = self.outputs.get(tool)
        if maker is None:
            raise AssertionError(f"nothing planted for {tool!r} in this test")
        produced = maker(dict(params, output_dir=str(slot)))
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


@pytest.fixture
def report():
    return {"patients": {}}


# ---------------------------------------------------------------------------
# A whole pipeline, with the six callees standing in for themselves
# ---------------------------------------------------------------------------

ARCH = {
    "Ba": [0.0, -30.0, -20.0], "S": [0.0, -10.0, 0.0], "N": [0.0, 40.0, 10.0],
    "RPo": [-35.0, -25.0, -12.0], "LPo": [35.0, -25.0, -12.0],
    "ROr": [-28.0, 30.0, 8.0], "LOr": [28.0, 30.0, 8.0],
    "ANS": [0.0, 45.0, -25.0], "PNS": [0.0, 5.0, -25.0],
    "RCo": [-48.0, -18.0, -6.0], "LCo": [48.0, -18.0, -6.0],
    "RGo": [-46.0, -6.0, -52.0], "LGo": [46.0, -6.0, -52.0],
    "Me": [0.0, 25.0, -70.0],
}

PATIENTS = ("P1", "P2")


def write_markups(path, points):
    from sadt_vface.landmarks import write_markups as write

    return write(points, str(path))


def write_transform(path, translation=(0.0, 0.0, 0.0)):
    import SimpleITK as sitk

    transform = sitk.AffineTransform(3)
    transform.SetTranslation([float(value) for value in translation])
    os.makedirs(os.path.dirname(str(path)) or ".", exist_ok=True)
    sitk.WriteTransform(transform, str(path))
    return str(path)


def mirrored(points, axis=0):
    """The same landmarks reflected across the mid-sagittal plane, and nudged.

    The nudge is the asymmetry: without it a patient is their own perfect
    mirror and every measurement is zero, which proves nothing about whether
    the numbers travelled.
    """
    flipped = {}
    for label, position in points.items():
        value = list(position)
        value[axis] = -value[axis]
        value[2] += 1.5
        flipped[label] = value
    return flipped


class PipelineSup:
    """A supervisor whose six callees produce what the real ones produce.

    Not a mock of the chain -- the chain runs for real. What is stood in for is
    each tool's own computation: `ASO` writes a scan named as it would be named
    and the transform beside it, `ALI_CBCT` writes the landmark files, and so
    on. Everything VFACE itself does -- the resample, the padding, the
    derivation between frames, the measurements, the features, the
    classification -- runs on what they wrote.
    """

    def __init__(self, tmp_path, landmarks=None, asymmetric=True):
        self.root = Path(tmp_path)
        self.out = self.root / "sup_out"
        self.tmp = self.root / "sup_tmp"
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.landmarks = landmarks or ARCH
        self.asymmetric = asymmetric
        self.calls = []
        self.spans = []
        self.messages = []
        self.logs = []
        # Waypoints and calls in the order they happened: what the server
        # folds into one bar, and so what has to move only forward.
        self.timeline = []

    # -- the five members a tool can see ------------------------------------

    def run(self, tool, **params):
        # The slot the real supervisor allocates per call, rather than one the
        # caller named: no tool passes `output_dir` any more, and numbering by
        # call is what keeps two runs of the same tool apart. `_progress` is
        # recorded apart, as the server's supervisor removes it.
        self.spans.append((tool, params.pop("_progress", None)))
        self.timeline.append((tool, self.spans[-1][1]))
        self.calls.append((tool, params))
        handler = getattr(self, f"_{tool.lower()}")
        destination = self.tmp / "sup" / f"{len(self.calls):02d}_{tool}" / "output"
        destination.mkdir(parents=True, exist_ok=True)
        handler(params, destination)
        return destination

    def progress(self, fraction, message):
        self.messages.append((fraction, message))
        self.timeline.append(("waypoint", (fraction, fraction)))

    def log(self, message, level="info", user=False):
        self.messages.append((None, message))
        self.logs.append((level, user, message))

    def asked(self, tool):
        return [params for name, params in self.calls if name == tool]

    # -- the callees --------------------------------------------------------

    def _aso(self, params, destination):
        """An oriented scan per patient, and the transform that oriented it."""
        suffix = params["output_suffix"]
        for path in _scans_under(params["input"]):
            patient = _patient(os.path.basename(path))
            write_volume(destination / f"{patient}_{suffix}.nii.gz")
            write_transform(destination / f"{patient}_{suffix}.tfm")

    def _amasss(self, params, destination):
        for path in _scans_under(params["scans"]):
            patient = _patient(os.path.basename(path))
            for structure in params["structures"]:
                write_volume(destination / f"{patient}_{structure}_seg.nii.gz")

    def _automatrix(self, params, destination):
        """Files moved by a transform.

        Which transform is the whole question, and `same_transform_for_every_patient`
        is what says: True is the mirror -- one reflection for the cohort -- and
        False is the registration, a matrix per patient. Reflecting on both
        would mirror the landmarks twice, which is the identity, and every
        measurement would come out zero while the run reported success.
        """
        source = Path(params["files"])
        reflecting = params["same_transform_for_every_patient"]
        for path in sorted(source.rglob("*")):
            if not path.is_file():
                continue
            if path.name.endswith(".mrk.json"):
                points = _read_markups(path)
                if reflecting and self.asymmetric:
                    points = mirrored(points)
                stem = path.name[: -len(".mrk.json")]
                suffix = params["output_suffix"]
                write_markups(destination / f"{stem}_{suffix}.mrk.json", points)
            elif path.suffix in (".gz", ".nii", ".nrrd"):
                write_volume(destination / path.name)

    def _areg_cbct(self, params, destination):
        """The registered scan and, beside it, the matrix that registered it."""
        region = params["regions"][0]
        for path in _scans_under(params["t2"]):
            patient = _patient(os.path.basename(path))
            write_volume(destination / region / f"{patient}_{region}_Reg.nii.gz")
            write_transform(destination / region / f"{patient}_{region}_Reg_transform.tfm")

    def _ali_cbct(self, params, destination):
        for path in _scans_under(params["input"]):
            patient = _patient(os.path.basename(path))
            wanted = {label: position for label, position in self.landmarks.items()
                      if label in params["landmarks"]}
            write_markups(destination / f"{patient}_lm_Pred_CB.mrk.json", wanted)

    def _batch_dental_seg(self, params, destination):
        for path in _scans_under(params["scans"]):
            patient = _patient(os.path.basename(path))
            _write_sphere(destination / f"{patient}_Seg.vtk")


def _scans_under(root):
    from sadt_vface.discovery import find_scans

    return find_scans(str(root))


def _patient(name):
    from sadt_vface.landmarks import patient_of

    return patient_of(name)


def _read_markups(path):
    from sadt_vface.landmarks import read_markups

    return read_markups(str(path))


def _write_sphere(path, centre=(0.0, 0.0, 0.0), radius=20.0):
    """A small closed surface, which is what a segmenter hands a heat map."""
    import vtk

    sphere = vtk.vtkSphereSource()
    sphere.SetCenter(*centre)
    sphere.SetRadius(radius)
    sphere.SetThetaResolution(16)
    sphere.SetPhiResolution(16)
    sphere.Update()

    os.makedirs(os.path.dirname(str(path)) or ".", exist_ok=True)
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileVersion(42)
    writer.SetFileName(str(path))
    writer.SetInputData(sphere.GetOutput())
    writer.Write()
    return str(path)


def write_measurement_lists(folder, measurements=None):
    """One measurement list per region, named the way the pipeline matches them."""
    import pandas as pd

    measurements = measurements or [
        ("Distance between 2 points T1 T2", "ROr", "ROr"),
        ("Distance between 2 points T1 T2", "RCo", "RCo"),
        ("Distance between 2 points T1 T2", "RGo", "RGo"),
        ("Distance between 2 points T1 T2", "ANS", "ANS"),
    ]
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        [{"Type of measurement": kind, "Point 1": first, "Point 2 / Line": second}
         for kind, first, second in measurements]
    )
    for name in ("Measurements_CB.xlsx", "Measurements_MAND.xlsx", "Measurements_MAX.xlsx"):
        frame.to_excel(folder / name, index=False)
    return str(folder)


def write_feature_template(path, columns=None):
    """The workbook naming the features a model was trained on."""
    import pandas as pd

    columns = columns or ["CB_ROr_ROr_RL", "MAND_RCo_RCo_IS", "MAX_ANS_ANS_AP"]
    frame = pd.DataFrame(columns=["ID"] + list(columns) + ["Asymmetry", "Mand", "Max"])
    os.makedirs(os.path.dirname(str(path)) or ".", exist_ok=True)
    frame.to_excel(str(path), index=False)
    return str(path)
