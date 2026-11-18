"""`scene.json`: which files of a result make one scene, in one space.

A viewer otherwise has to guess what belongs together from file names, and
every tool names its files its own way (VISU's `index.py` lists four
conventions). The tool that wrote the files knows; this is where it says so.

One file per scene folder, every path relative to that folder. A path may
climb out of it (`../`) only to a file of the same result that several scenes
share -- the oriented T1 every region of a CBCT run is registered onto -- so
the result's folder can be moved or downloaded as a whole:

    {
      "version": 1,
      "space": "the CBCT",            # what every file below is placed in
      "files": [
        {"path": "case_Reg.vtk", "kind": "model", "role": "registered"},
        {"path": "case_Reg_lm.mrk.json", "kind": "markups", "role": "moved"},
        ...
      ]
    }

`kind` is one of KINDS, the vocabulary a viewer loads by. `role` says what
the file is to the result, for a viewer that wants to colour or order them.

Standard library only, like the rest of this package; the landmark move below
is the one piece of arithmetic it needs, and a 4x4 product does not need numpy.
"""

import json
import os

SCENE_NAME = "scene.json"
VERSION = 1

VOLUME, MODEL, MARKUPS, TRANSFORM = "volume", "model", "markups", "transform"
KINDS = (VOLUME, MODEL, MARKUPS, TRANSFORM)

_KIND_BY_EXTENSION = (
    (".mrk.json", MARKUPS), (".json", MARKUPS),
    (".nii.gz", VOLUME), (".nii", VOLUME), (".nrrd", VOLUME), (".gipl.gz", VOLUME),
    (".vtk", MODEL), (".vtp", MODEL), (".stl", MODEL), (".obj", MODEL),
    (".tfm", TRANSFORM), (".h5", TRANSFORM),
)


def kind_of(path: str) -> str:
    """The KINDS entry a file's extension says, longest extension first."""
    lowered = path.lower()
    for extension, kind in _KIND_BY_EXTENSION:
        if lowered.endswith(extension):
            return kind
    raise ValueError(f"'{os.path.basename(path)}' is not a file a scene can hold.")


def write(folder: str, space: str, files, name: str = SCENE_NAME) -> str:
    """Write `folder/<name>` for `files`, `(path, role)` pairs.

    `name` is `scene.json` for a folder that holds one scene, and
    `<prefix>_scene.json` where several patients share a folder: a viewer
    reads every file whose name ends in `scene.json`.

    Paths may be absolute or relative to the working directory; they are
    stored relative to `folder`. A file named twice is listed once, at its
    first place, and one that does not exist is refused: a scene that names a
    missing file is a viewer error waiting to happen.
    """
    listed, seen = [], set()
    for path, role in files:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"A scene cannot list '{os.path.basename(path)}': it was not written.")
        relative = os.path.relpath(path, folder)
        if relative in seen:
            continue
        seen.add(relative)
        listed.append({"path": relative.replace(os.sep, "/"), "kind": kind_of(path), "role": role})
    destination = os.path.join(folder, name)
    with open(destination, "w", encoding="utf-8") as handle:
        json.dump({"version": VERSION, "space": space, "files": listed}, handle, indent=2)
    return destination


def move_landmarks(source: str, matrix, destination: str) -> str:
    """Write `source`'s landmarks moved by the 4x4 `matrix`, in `source`'s format.

    Only the positions change: a markups file keeps its coordinate system,
    labels and display settings, a plain `{label: [x, y, z]}` file stays one.
    The matrix is applied to the raw coordinates as written -- the ones a
    registration reads -- so the points move exactly as the mesh they were
    placed on did, whatever convention the file is in.
    """
    with open(source, encoding="utf-8") as handle:
        payload = json.load(handle)

    def moved(position):
        x, y, z = (float(value) for value in position)
        return [float(row[0]) * x + float(row[1]) * y + float(row[2]) * z + float(row[3])
                for row in (matrix[0], matrix[1], matrix[2])]

    markups = payload.get("markups") if isinstance(payload, dict) else None
    if markups:
        for markup in markups:
            for control_point in markup.get("controlPoints", []):
                if control_point.get("position"):
                    control_point["position"] = moved(control_point["position"])
    else:
        payload = {
            label: moved(position)
            if isinstance(position, (list, tuple)) and len(position) == 3 else position
            for label, position in payload.items()
        }
    os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
    with open(destination, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return destination


def landmark_extension(path: str) -> str:
    """`.mrk.json` or `.json`, whichever a landmark file carries."""
    return ".mrk.json" if path.lower().endswith(".mrk.json") else os.path.splitext(path)[1]
