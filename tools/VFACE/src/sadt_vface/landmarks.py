"""Landmarks: whose they are, where they are, and how to move them.

Three things VFACE does itself, because none of them is another tool's job:

* **it gives a scan room before the landmark search.** ALI border-pads the
  image it samples, but its agent refuses any position outside the UNPADDED
  size, so a landmark a voxel or two from the border sends it into a bounds
  bounce and, after three restarts, it gives up. Me sits 1.2 mm above the floor
  of these CBCTs and the very same model finds it at once when there is room
  around it. Fixing the mismatch belongs in ALI and would serve every module
  that calls it; this is the safe half of the answer.
* **it derives one frame's landmarks from the other's.** Both orientations come
  from the same centred scan, so a point's position in the maxillary frame is
  `inv(target) . source` applied to its position in the cranial base frame.
  Searching a second time costs minutes per patient AND makes the same
  anatomical point land in two slightly different places depending on which run
  found it.
* **it reads what ALI wrote**, which is the only thing that says which
  landmarks belong to which patient.
"""

import json
import logging
import os

from .errors import ToolInputError
from .scans import find_scans, split_scan_extension

logger = logging.getLogger(__name__)

MARKUPS_EXTENSION = ".mrk.json"

# How much room the landmark search is given, in millimetres. Enough to clear
# the agent's field of view at the coarsest scale it searches.
DEFAULT_MARGIN_MM = 30

# Tokens that a file name carries and a PATIENT does not: what a step appended
# to say what it did. The identifier is everything before the first of them.
#
# Matched as whole tokens, never as substrings. Upstream cuts with
# `basename.split("_CB")[0]`, which also cuts `P1_CBrown` -- the same substring
# defect this repository has fixed three times over (`"cb" in basename` making
# every scan a cranial base; `vtk_name in json_name` pairing patient 1 with
# patient 10).
DECORATION_TOKENS = (
    "scan", "or", "mand", "md", "max", "mx", "cb", "lm", "t1", "t2", "cl",
    "pred", "seg", "reg", "mir", "derived",
    # AREG names its outputs after the region it registered on, and it uses the
    # DISPLAY names -- `P1_Cranial base_Reg.nii.gz`, with a space in it. Left
    # out, everything downstream of a registration reads that patient as
    # "P1_Cranial base" and stops pairing with anything the steps before wrote.
    "cranial base", "mandible", "maxilla",
)

# ALI's landmark groups, as it names the files it writes. Carried here rather
# than read out of ALI: the two live in different virtualenvs, and upstream
# reaches into `ALI_CBCT/ALI_CBCT_utils/constants.py` with `ast` to avoid
# importing torch -- which is a coupling to another tool's SOURCE LAYOUT, not
# even to its API.
#
# It is safe as a copy because it names VFACE's OWN output files and nothing
# else: the landmark files ALI writes are read back by glob, one patient's
# points merged across whatever files hold them, so a group renamed there
# changes no behaviour here.
GROUP_LABELS = {
    "CB": ("Ba", "S", "N", "RPo", "LPo", "RFZyg", "LFZyg", "C2", "C3", "C4"),
    "U": ("RInfOr", "LInfOr", "LMZyg", "RMZyg", "RNC", "LNC", "RPF", "LPF",
          "PNS", "ANS", "A", "UR3O", "UR1O", "UL3O", "UR6DB", "UR6MB", "UL6MB",
          "UL6DB", "IF", "ROr", "LOr"),
    "L": ("RCo", "RGo", "Me", "Gn", "Pog", "PogL", "B", "LGo", "LCo", "LR1O",
          "LL6MB", "LL6DB", "LR6MB", "LR6DB", "LAF", "LAE", "RAF", "RAE",
          "LMCo", "LLCo", "RMCo", "RLCo", "RMeF", "LMeF", "RSig", "RPRa",
          "RARa", "LSig", "LARa", "LPRa"),
}

_GROUP_OF = {label: group for group, labels in GROUP_LABELS.items() for label in labels}


def patient_of(filename: str) -> str:
    """The patient a file belongs to: everything before the first decoration.

    `P_0001_T1_CB_Or_lm_Pred_CB.mrk.json` and `P_0001_T1_CB_Or.nii.gz` are one
    patient, and that is the whole contract by which VFACE pairs a landmark
    file with the scan it was found on, and a scan with the transform that
    oriented it.
    """
    stem = split_scan_extension(os.path.basename(filename))[0]
    if stem.lower().endswith(".mrk"):
        stem = stem[: -len(".mrk")]
    stem = os.path.splitext(stem)[0]

    parts = []
    for token in stem.replace("-", "_").split("_"):
        if token.lower() in DECORATION_TOKENS:
            break
        parts.append(token)
    return "_".join(parts) or stem


def group_of(label: str) -> str:
    """Which of ALI's groups a landmark belongs to; `U` for one it does not name."""
    return _GROUP_OF.get(label, "U")


def read_markups(path: str) -> dict:
    """`{label: [x, y, z]}` from one Slicer markups file."""
    with open(path, encoding="utf-8") as handle:
        document = json.load(handle)

    points = {}
    for markup in document.get("markups", []):
        for control_point in markup.get("controlPoints", []):
            label = control_point.get("label")
            position = control_point.get("position")
            if label and position is not None and len(position) >= 3:
                points[label] = [float(value) for value in position[:3]]
    return points


def read_cohort(root: str) -> dict:
    """`{patient: {label: position}}` for every markups file under `root`.

    One patient's points are MERGED across whatever files hold them: ALI writes
    one file per group, so a patient's cranial base and lower points arrive
    separately and a measurement needs both. Read by glob rather than by group
    name, so a group renamed in ALI changes nothing here.
    """
    found: dict = {}
    for directory, subdirectories, names in os.walk(root or ""):
        subdirectories.sort()
        for name in sorted(names):
            if name.startswith(".") or not name.lower().endswith(MARKUPS_EXTENSION):
                continue
            path = os.path.join(directory, name)
            try:
                points = read_markups(path)
            except Exception as exc:  # noqa: BLE001 - one file must not cost the batch
                logger.warning("VFACE could not read one landmark file: %s",
                               type(exc).__name__)
                continue
            if points:
                found.setdefault(patient_of(name), {}).update(points)
    return found


def write_markups(points: dict, path: str) -> str:
    """One Slicer markups file, the shape ALI writes.

    `visibility` is True, and that is not decoration: a markups file written
    with it False loads into Slicer, builds the node and draws NOTHING. It was
    invisible in both of ALI's engines until somebody opened a result outside
    the module.
    """
    document = {
        "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/"
                   "Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#",
        "markups": [
            {
                "type": "Fiducial",
                "coordinateSystem": "LPS",
                "controlPoints": [
                    {
                        "id": str(index),
                        "label": label,
                        "description": "",
                        "associatedNodeID": "",
                        "position": [float(value) for value in position],
                        "orientation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                        "selected": True,
                        "locked": True,
                        "visibility": True,
                        "positionStatus": "defined",
                    }
                    for index, (label, position) in enumerate(sorted(points.items()), start=1)
                ],
            }
        ],
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=4)
    return path


def pad_scans(scans_dir: str, output_dir: str, margin_mm: int = DEFAULT_MARGIN_MM,
              report: dict = None) -> str:
    """Copy each scan with empty space around it, so the search can reach the edge.

    Physical coordinates are preserved -- the origin moves with the padding --
    so the landmarks come back in the original scan's space. Measured upstream
    against an unpadded run, points away from the border moved by at most one
    voxel.

    The padding value is the scan's own minimum rather than zero, so the
    intensity rescaling the landmark tool applies is unchanged by the room it
    was given.
    """
    import SimpleITK as sitk

    found = find_scans(scans_dir)
    if not found:
        raise ToolInputError(
            f"No scan to give room to under {os.path.basename(scans_dir)}."
        )

    os.makedirs(output_dir, exist_ok=True)
    for path in found:
        destination = os.path.join(output_dir, os.path.relpath(path, scans_dir))
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        try:
            image = sitk.ReadImage(path)
            pad = [max(1, int(round(margin_mm / spacing))) for spacing in image.GetSpacing()]
            background = float(sitk.GetArrayFromImage(image).min())
            sitk.WriteImage(sitk.ConstantPad(image, pad, pad, background), destination)
        except Exception as exc:  # noqa: BLE001 - an unpadded scan is still worth searching
            logger.warning("VFACE could not pad one scan, using it as it is: %s",
                           type(exc).__name__)
            if report is not None:
                report.setdefault("not_padded", {})[
                    os.path.relpath(path, scans_dir)
                ] = f"{type(exc).__name__}: {exc}"
            import shutil

            shutil.copy(path, destination)

    logger.info("VFACE: %d scan(s) given %d mm of room for the landmark search",
                len(found), margin_mm)
    return output_dir


def transforms_by_patient(root: str) -> dict:
    """`{patient: path}` for the `.tfm` files written beside the oriented scans."""
    found = {}
    for directory, subdirectories, names in os.walk(root or ""):
        subdirectories.sort()
        for name in sorted(names):
            if name.lower().endswith(".tfm"):
                found.setdefault(patient_of(name), os.path.join(directory, name))
    return found


def stems_by_patient(root: str) -> dict:
    """`{patient: scan basename without its extension}`, to name outputs as ALI does."""
    stems = {}
    for path in find_scans(root):
        name = os.path.basename(path)
        stems.setdefault(patient_of(name), split_scan_extension(name)[0])
    return stems


def derive_into_frame(source_landmarks: str, source_transforms: str,
                      target_transforms: str, target_scans: str,
                      keep, output_dir: str, report: dict = None) -> str:
    """Express landmarks found in one oriented frame in another oriented frame.

    Both orientations come from the same centred scan, so a landmark's position
    in the target frame is `inv(target) . source` applied to its position in
    the source frame. A rigid transform is exact and instant where a second
    search is minutes per patient -- and it keeps one anatomical point from
    landing in two slightly different places depending on which run found it.

    It also means there is only ONE set of landmarks to check: a correction
    made to the cranial base points flows into the maxillary ones.
    """
    import SimpleITK as sitk

    source = read_cohort(source_landmarks)
    if not source:
        raise ToolInputError(
            f"No landmark file to derive from under {os.path.basename(source_landmarks)}."
        )

    from_source = transforms_by_patient(source_transforms)
    to_target = transforms_by_patient(target_transforms)
    stems = stems_by_patient(target_scans)
    wanted = list(keep)
    os.makedirs(output_dir, exist_ok=True)

    written = []
    for patient, points in sorted(source.items()):
        if patient not in from_source or patient not in to_target:
            logger.warning("VFACE: one patient has no orientation transform, its "
                           "landmarks cannot be derived")
            if report is not None:
                report.setdefault("not_derived", {})[patient] = (
                    "no orientation transform for "
                    + ("the source frame" if patient not in from_source else "the target frame")
                )
            continue
        try:
            to_centred = sitk.ReadTransform(from_source[patient])
            from_centred = sitk.ReadTransform(to_target[patient]).GetInverse()
        except Exception as exc:  # noqa: BLE001 - one patient must not cost the cohort
            logger.warning("VFACE could not read one orientation transform: %s",
                           type(exc).__name__)
            if report is not None:
                report.setdefault("not_derived", {})[patient] = (
                    f"{type(exc).__name__}: {exc}"
                )
            continue

        by_group: dict = {}
        for label in wanted:
            if label not in points:
                continue
            moved = from_centred.TransformPoint(to_centred.TransformPoint(points[label]))
            by_group.setdefault(group_of(label), {})[label] = list(moved)

        if not by_group:
            if report is not None:
                report.setdefault("not_derived", {})[patient] = (
                    f"none of the {len(wanted)} landmark(s) asked for were found"
                )
            continue

        stem = stems.get(patient, f"{patient}_derived")
        for group, group_points in sorted(by_group.items()):
            written.append(write_markups(
                group_points,
                os.path.join(output_dir, f"{stem}_lm_Pred_{group}{MARKUPS_EXTENSION}"),
            ))

    if not written:
        # Counted on what was WRITTEN. A guard counting the patients walked past
        # would pass on a cohort where not one of them could be derived.
        raise ToolInputError(
            f"No patient's landmarks could be expressed in the target frame, out "
            f"of {len(source)} with landmarks. The per-patient reasons are in the "
            "run report."
        )
    logger.info("VFACE: landmarks derived into the target frame for %d patient(s)",
                len(source) - len((report or {}).get("not_derived", {})))
    return output_dir
