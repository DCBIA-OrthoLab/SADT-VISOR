"""A synthetic head: two mirror-symmetric condyles in the oriented frame.

No clinical data is read or written. The scene is built in the ORIENTED frame
(the one ASO leaves a scan in, rotated onto the reference) and then shifted by a
known translation into the scan's own frame, so a test can check that the tool
recovers that translation from the landmarks alone.
"""

import json
import shutil
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

# Where the reference puts each landmark, in LPS as a markups file stores it.
REFERENCE_LPS = {
    "Ba": (0.0, 40.0, -25.0),
    "S": (0.0, -10.0, 10.0),
    "N": (0.0, -85.0, 20.0),
    "RPo": (-50.0, 0.0, 0.0),
    "LPo": (50.0, 0.0, 0.0),
    "ROr": (-30.0, -80.0, 20.0),
    "LOr": (30.0, -80.0, 20.0),
}
# The shift between the scan's own frame and the oriented one: oriented = world + SHIFT.
SHIFT = np.array([3.0, -2.0, 1.0])
# Condyle centres in the oriented frame (RAS), one per side, mirror images.
CONDYLES = {"R": np.array([50.0, 8.0, -5.0]), "L": np.array([-50.0, 8.0, -5.0])}


def reference_ras():
    return {k: np.array(v) * np.array([-1.0, -1.0, 1.0]) for k, v in REFERENCE_LPS.items()}


def _grid(size=160, origin=-80.0):
    """World coordinates (RAS, 1 mm voxels) of a cube, and its affine."""
    axis = origin + np.arange(size, dtype=np.float32)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    affine = np.eye(4)
    affine[:3, 3] = origin
    return np.stack([x, y, z], -1), affine


def write_scene(folder, name, scene_shift=SHIFT, mandible=True):
    """Write `<name>_Or.nii.gz`, its markups and the mandible mask; return the paths."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    world, affine = _grid()
    oriented = world + scene_shift
    volume = np.zeros(world.shape[:3], dtype=np.float32)
    mask = np.zeros(world.shape[:3], dtype=np.uint8)
    for centre in CONDYLES.values():
        distance = np.linalg.norm(oriented - centre, axis=-1)
        volume += 1000.0 * np.exp(-(distance / 6.0) ** 2)
        if mandible:
            mask |= (distance < 6.0).astype(np.uint8)
    if mandible:
        # The body joining the two condyles: the crop keeps the largest connected
        # piece of the mask, as it must on a real mandible, so two floating
        # spheres would lose one side. Its top (z = -10) overlaps the spheres and stays below the heads.
        bar = (np.abs(oriented[..., 1] - 8.0) < 3.0) & (np.abs(oriented[..., 2] + 14.0) < 4.0) \
            & (np.abs(oriented[..., 0]) < 50.0)
        mask |= bar.astype(np.uint8)
    scan = folder / f"{name}_Or.nii.gz"
    nib.save(nib.Nifti1Image(volume, affine), scan)
    mask_path = folder / f"{name}_Or_Pred_MAND.nii.gz"
    nib.save(nib.Nifti1Image(mask, affine), mask_path)

    # The scan's own landmarks: the reference minus the shift, LPS like Slicer writes them.
    points = [
        {"label": label, "position": list((np.array(ras) - scene_shift) * np.array([-1.0, -1.0, 1.0]))}
        for label, ras in reference_ras().items()
    ]
    markups = folder / f"{name}_lm_Or.mrk.json"
    markups.write_text(json.dumps(
        {"markups": [{"coordinateSystem": "LPS", "controlPoints": points}]}))
    return scan, markups, mask_path


def aso_report(names, status="ok", with_reference=True, reasons=None, extension=".nii.gz"):
    """`names` may carry a subfolder (`T1/case`): ASO keys a case by its relative path and
    lists what it produced relative to the folder the report sits in."""
    reasons = reasons or {}
    report = {
        "output_suffix": "Or",
        "frame": "Frankfurt horizontal",
        "cases": {
            name: {
                "status": status,
                **({"reason": reasons[name]} if name in reasons else {}),
                "landmarks_used": sorted(REFERENCE_LPS),
                "landmarks_dropped": {},
                "produced": (
                    [f"{name}_Or{extension}", f"{name}_lm_Or.mrk.json"] if status == "ok" else []
                ),
            }
            for name in names
        },
    }
    if with_reference:
        report["reference_positions"] = {k: list(v) for k, v in REFERENCE_LPS.items()}
    return report


class FakeSup:
    """A supervisor as a tool sees one: records its calls, plants what the tools would write."""

    def __init__(self, tmp_path, names=("a",), aso_status="ok", with_reference=True, mandible=True,
                 reasons=None, aso_extension=".nii.gz", convert_fails=False):
        self.reasons = reasons
        self.aso_extension = aso_extension
        self.convert_fails = convert_fails
        self.root = Path(tmp_path) / "sup"
        self.names = list(names)
        self.aso_status = aso_status
        self.with_reference = with_reference
        self.mandible = mandible
        self.calls = []
        self.messages = []

    def run(self, tool, **params):
        params.pop("_progress", None)
        self.calls.append((tool, params))
        folder = self.root / f"{len(self.calls):02d}_{tool}"
        if tool == "ASO":
            for name in self.names:
                if self.aso_status == "ok":
                    subfolder, stem = os.path.split(name)
                    scan, _, _ = write_scene(folder / subfolder, stem, mandible=False)
                    if self.aso_extension != ".nii.gz":
                        # ASO keeps the input's family; the bytes do not matter to the fake.
                        scan.rename(scan.with_name(f"{stem}_Or{self.aso_extension}"))
            (folder).mkdir(parents=True, exist_ok=True)
            (folder / "ASO_report.json").write_text(
                json.dumps(aso_report(self.names, self.aso_status, self.with_reference,
                                      self.reasons, self.aso_extension)))
            return folder
        if tool == "AMASSS":
            folder.mkdir(parents=True, exist_ok=True)
            if self.mandible:
                # One mask per scan it was handed, named after the scan as AMASSS does.
                for staged in sorted(Path(params["scans"]).iterdir()):
                    write_scene(folder / "SegOut", staged.name[: -len("_Or.nii.gz")])
            return folder
        if tool == "Convert":
            folder.mkdir(parents=True, exist_ok=True)
            if not self.convert_fails:
                for staged in sorted(Path(params["scans"]).iterdir()):
                    stem = staged.name.split(".")[0]
                    shutil.copyfile(staged, folder / f"{stem}.nii.gz")
            return folder
        raise AssertionError(f"unexpected tool {tool!r}")

    def progress(self, fraction, message):
        self.messages.append((fraction, message))

    def log(self, message, level="info", user=False):
        self.messages.append((None, message))

    def asked(self, tool):
        found = [params for name, params in self.calls if name == tool]
        assert len(found) == 1, f"{tool} was called {len(found)} time(s)"
        return found[0]


@pytest.fixture
def sup(tmp_path):
    return FakeSup(tmp_path)
