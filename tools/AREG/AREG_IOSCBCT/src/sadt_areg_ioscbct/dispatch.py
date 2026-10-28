"""Everything AREG_IOSCBCT does around the registration itself.

Three modes, and they differ only in where the landmarks come from:

    Registration       you supply both sets. Nothing is predicted, no other
                       tool is called, and this needs no GPU at all.
    Semi-Automated     the intraoral meshes are labelled and the landmarks
                       predicted; the CBCT is taken as it is.
    Fully-Automated    the CBCT is oriented first, then both sides predicted.

That progression is why this tool has no engine of its own: each step it does
not do itself is a `sup.run()` into a tool that has one.
"""

import collections
import json
import logging
import os
import shutil
import time

import numpy as np

from sadt_areg_common import catalogs, pairing, scene
from sadt_areg_common.errors import ToolInputError

from . import geometry, pipeline, progress, tools, where

# Prefixed with the patient and arch the batch loop is on: see `where`.
logger = where.attach(logging.getLogger(__name__))

MODALITY = catalogs.MODALITY_IOSCBCT
REPORT_NAME = "AREG_report.json"
WORK_DIRNAME = ".areg_work"


def _read_mesh(path: str):
    """The intraoral mesh with its point data, as pyvista reads it.

    pyvista rather than raw vtk, for two things the registration needs and a
    bare `vtkPolyDataReader` does not hand over as usefully: the `Universal_ID`
    tooth labels, which say which points are crowns, and `compute_normals`,
    which the point-to-plane step measures along. It also transforms the mesh
    with its arrays attached, so the registered file keeps the labels the
    unregistered one carried.
    """
    import pyvista as pv

    return pv.read(path)


def _write_mesh(mesh, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Binary, not ASCII: it round-trips float32 exactly, while ASCII prints six
    # significant digits, and it parses far faster.
    mesh.save(path, binary=True)


def _cbct_surface(scan_path: str, landmark_sets: dict):
    """The CBCT contoured as a surface, plus its verdict on the landmarks.

    Once per patient, not once per arch: contouring a 2.8 million point surface
    is the expensive half of the run, and both arches query the same one. The
    level is read off all of that patient's CBCT landmarks together -- it
    describes the scan, not one jaw, and twelve points make a steadier median
    than six.

    Returns `(None, {}, reason)` when the volume cannot be read. The caller then
    registers on the landmarks alone and the report says the ICP did not run,
    which is worse than refining but better than stopping -- and it is written
    down either way rather than looking like a run that simply had no ICP.
    """
    from . import volume

    merged = {}
    for points in landmark_sets.values():
        merged.update(points)
    try:
        surface, on_enamel = volume.read(scan_path, merged)
    except Exception as exc:  # noqa: BLE001 - a landmark-only run is still a run
        # The class name, not `str(exc)`, in the report: readers carry the
        # server's own paths in their messages, and the report goes back to the
        # client. The log line has the message too -- the server redacts paths
        # there -- and `where` puts the patient's position in front of it.
        logger.warning(
            "CBCT contouring failed (%s: %s); registering on the landmarks alone, "
            "without ICP", type(exc).__name__, exc)
        logger.debug("CBCT contouring traceback", exc_info=True)
        return None, {}, f"the CBCT could not be contoured ({type(exc).__name__})"
    return surface, on_enamel, None


def _landmark_files(directory: str) -> list:
    """Every landmark file under `directory`, in a stable order."""
    paths = []
    if not directory or not os.path.isdir(directory):
        return paths
    for root, directories, files in os.walk(directory):
        directories.sort()
        for name in sorted(files):
            if name.lower().endswith(pipeline.LANDMARK_EXTENSIONS):
                paths.append(os.path.join(root, name))
    return paths


def _landmarks_by_jaw(directory: str, side: str = "landmark") -> dict:
    """`{relative path: {label: position}}` for every landmark file in a folder.

    Keyed by the path relative to the folder, not by the base name: ALI writes
    one file per scan and MIRRORS the input's tree, so two sites' `P1_U_lm_
    Pred.mrk.json` are two files. Keyed by base name the second silently
    replaced the first, which then made `len(candidates) == 1` -- the "one file
    covers every jaw" fallback -- fire on a folder that held two.

    A file that cannot be read is skipped with a warning rather than taking the
    batch with it: one truncated JSON among forty used to abort the run with a
    bare JSONDecodeError and no word of which side it was on. `side` ("intraoral"
    or "CBCT") is what that warning names, with the file's position -- never its
    name.
    """
    found = {}
    paths = _landmark_files(directory)
    for index, path in enumerate(paths, start=1):
        try:
            points = pipeline.read_landmarks(path)
        except Exception as exc:  # noqa: BLE001 - one bad file must not cost the batch
            logger.warning("%s landmark file %d of %d unreadable (%s: %s)",
                           side, index, len(paths), type(exc).__name__, exc)
            continue
        if points:
            found[os.path.relpath(path, directory)] = points
    return found


def _for_patient(candidates: dict, patient: str, sole_patient: bool) -> dict:
    """The landmark files that belong to one patient.

    Matching on the jaw token ALONE is what made a two-patient batch register
    the second patient's mesh against the FIRST patient's landmarks: every
    ALI_IOS file carries a `_U` or `_L` token, `sorted()` puts `P1...` before
    `P2...`, and the first jaw match won. Nothing about the result looks wrong
    -- a rigid transform is produced, the mesh is written, the report says
    "ok" -- so the whole batch after patient one is quietly registered onto the
    wrong anatomy.

    A batch holding ONE patient keeps the looser rule. There the two sides
    cannot be confused, and a landmark file named by a convention
    `patient_key` cannot read -- one with no digits in it at all, which is what
    the published reference files look like -- would otherwise stop matching
    anything at all.
    """
    own = {
        name: points
        for name, points in candidates.items()
        if pipeline.patient_key(os.path.basename(name)) == patient
    }
    if own:
        return own
    return candidates if sole_patient else {}


_JAW_TOKENS = {"u", "upper", "l", "lower"}


def _arch(mesh_path: str):
    """"U" or "L" from the mesh's jaw token, or None when its name has none."""
    from sadt_areg_common import pairing

    found = set(pairing.tokens(os.path.basename(mesh_path)))
    if found & {"u", "upper"}:
        return "U"
    if found & {"l", "lower"}:
        return "L"
    return None


def _needs_review(detail: dict) -> str:
    """Why a written registration should not be trusted as is, or "".

    The ICP grades itself on the surface only, and on a smooth occlusal
    surface a fit that slid along the arch scores as well as the right one.
    The landmarks are the independent check: still further apart after the
    ICP than the pre-alignment would ever accept, the arch is not where its
    landmarks say it is, whatever the surface fit reports.
    """
    after = (detail.get("icp") or {}).get("landmark_rms_after_mm")
    if after is None or after <= geometry.MAX_LANDMARK_RESIDUAL_MM:
        return ""
    return (f"the landmarks are still {after:.1f} mm from their CBCT counterparts "
            f"after the ICP (more than {geometry.MAX_LANDMARK_RESIDUAL_MM:.0f} mm)")


def _most_common(failures: list) -> tuple:
    """`(first exception of the commonest kind, how many share it)`.

    Grouped on class AND message, so "share only 1 landmark(s)" on five arches
    and an ICP refusal on one read as two kinds, the larger first.
    """
    kinds = collections.Counter((type(exc).__name__, str(exc)) for exc in failures)
    (name, message), count = kinds.most_common(1)[0]
    first = next(exc for exc in failures
                 if (type(exc).__name__, str(exc)) == (name, message))
    return first, count


def _match_landmarks(mesh_path: str, candidates: dict) -> dict:
    """The landmark set belonging to this mesh.

    Two shapes, because the two sides genuinely differ and only one of them can
    name a jaw:

    - **per-jaw files**, which is what ALI_IOS writes (`..._U_lm_Pred.mrk.json`)
      and what upstream's own RegTestFiles carry on both sides. Matched on the
      jaw token, never on sort order: pairing by position is how an upper mesh
      gets registered against a lower arch's points.
    - **one file for everything**, which is what ALI_CBCT writes. A CBCT covers
      both arches in one volume, so its landmark file has no jaw to name. The
      labels themselves carry it -- `UR1O` against `LR1O` -- and
      `shared_landmarks` intersects, so the upper mesh takes the upper points
      out of the same file the lower mesh takes the lower ones from.

    So a jaw match wins when there is one, and a single unlabelled file is
    accepted as covering every jaw rather than refused.
    """
    name = _matched_file(mesh_path, candidates)
    return candidates[name] if name is not None else {}


def _matched_file(mesh_path: str, candidates: dict):
    """The key in `candidates` of the file `_match_landmarks` reads, or None.

    Kept apart so the same-space outputs can copy or move THAT file, the one
    the registration actually used, rather than finding it again by a second
    rule that could disagree.
    """
    from sadt_areg_common import pairing

    mesh_tokens = set(pairing.tokens(os.path.basename(mesh_path)))
    for name in sorted(candidates):
        if mesh_tokens & set(pairing.tokens(os.path.basename(name))) & _JAW_TOKENS:
            return name

    unlabelled = [
        name for name in sorted(candidates)
        if not (set(pairing.tokens(os.path.basename(name))) & _JAW_TOKENS)
    ]
    if len(unlabelled) == 1:
        return unlabelled[0]
    if len(candidates) == 1:
        return next(iter(candidates))
    return None


# The frame the CBCT is oriented into, in ASO's own words. ASO owns the
# reference bundles that define its frames and resolves the one this names from
# its own data folder; this engine holds no copy.
#
# Frankfurt rather than Occlusal, and this is a CHOICE a caller can override by
# naming a reference bundle in `cbct_reference`: the two planes carry DISJOINT
# landmark sets, Frankfurt is the frame the shipped test data is oriented into,
# and it is the anatomical convention a CBCT is read in. What it must not be is
# a question asked of a clinician who has no way to know which one the rest of
# the chain expects. ASO's default landmark selection is this frame's set, so
# nothing else needs naming.
_ORIENTATION_FRAME = "Frankfurt horizontal"


# How a run shares its bar, as fractions of the whole, per supervised call.
# The calls come first because they run first -- in that order: ASO orients the
# CBCT, Crown_Seg labels the crowns, ALI_IOS and then ALI_CBCT place the
# landmarks -- and the registration takes what is left. ASO's CBCT mode runs
# ALI_CBCT inside it, and an ALI_CBCT agent is a full two-scale walk of the
# volume, so those two get the larger slices; the intraoral networks run on a
# mesh and cost less. A weighting, not a measurement: what is exact is the
# counter in each message.
ORIENT_SHARE = 0.25
CROWN_SHARE = 0.1
IOS_LANDMARK_SHARE = 0.1
CBCT_LANDMARK_SHARE = 0.2


def _spans(automation: str, predict_ios: bool, predict_cbct: bool) -> dict:
    """{step: (start, end)} for the steps this run makes, in the order it makes
    them, ending with "register" on whatever is left.

    The waypoints used to be fixed numbers in `tools.py` -- 0.5 for ASO, 0.1
    for ALI_CBCT -- while the calls ran ASO first and ALI_CBCT last, so the bar
    went backwards twice in a fully-automated run. Deriving them from the call
    order is what makes that impossible; a step a mode skips takes no slice.
    """
    steps = []
    if automation != catalogs.AUTOMATION_REGISTRATION:
        if automation == catalogs.AUTOMATION_FULLY:
            steps.append(("orient", ORIENT_SHARE))
        steps.append(("crowns", CROWN_SHARE))
        if predict_ios:
            steps.append(("ios_landmarks", IOS_LANDMARK_SHARE))
        if predict_cbct:
            steps.append(("cbct_landmarks", CBCT_LANDMARK_SHARE))
    spans, position = {}, 0.0
    for name, share in steps:
        spans[name] = (round(position, 6), round(position + share, 6))
        position += share
    spans["register"] = (round(position, 6), 1.0)
    return spans


def _carry_cbct(cbct_path: str, folder: str, output_dir: str,
                with_transform: bool = True) -> dict:
    """Copy the CBCT the meshes were registered onto into the patient's folder.

    In the Fully-Automated mode that is ASO's oriented scan, not the caller's:
    every arch is written in its frame, and it lived only in the job's work
    directory. Opened over the original scan, the meshes were off by the whole
    rotation ASO applied, with nothing in the results to put them back on. Its
    transform (original scan to oriented one, ASO's `<scan>_transform.tfm`)
    comes too, for a reader who wants the meshes on the original scan instead.

    `with_transform` is False for a scan nobody oriented -- the caller's own,
    copied for the same-space outputs -- which has no such transform beside it.

    Returns the report fields naming what was copied, relative to `output_dir`.
    """
    os.makedirs(folder, exist_ok=True)
    copied = {}
    stem = pairing.split_scan_extension(os.path.basename(cbct_path))[0]
    transform = os.path.join(os.path.dirname(cbct_path), f"{stem}_transform.tfm")
    sources = [("oriented_cbct" if with_transform else "cbct_copy", cbct_path)]
    if with_transform:
        sources.append(("cbct_transform", transform))
    for field, source in sources:
        if not os.path.isfile(source):
            logger.warning("No %s to copy beside the registered meshes", field.replace("_", " "))
            continue
        destination = os.path.join(folder, os.path.basename(source))
        shutil.copy2(source, destination)
        copied[field] = os.path.relpath(destination, output_dir)
    return copied


def _write_transform(matrix: np.ndarray, path: str) -> str:
    """The registration as a `.tfm` Slicer opens, written as AREG_IOS writes
    its own (`sadt_areg_ios.icp.write_transform`).

    `matrix` moves the mesh's points onto the CBCT. A `.tfm` is consumed the
    other way round -- it maps a point of the result back to where to sample
    it -- so the inverse is what is written. The `.npy` beside it stays the
    point-moving matrix, as before.
    """
    import SimpleITK as sitk

    inverted = np.linalg.inv(matrix)
    transform = sitk.AffineTransform(3)
    transform.SetMatrix(inverted[:3, :3].flatten().tolist())
    transform.SetTranslation(inverted[:3, 3].tolist())
    sitk.WriteTransform(transform, path)
    return path


def register(ios_dir: str, cbct_dir: str, ios_landmark_dir: str, cbct_landmark_dir: str,
             output_dir: str, suffix: str, report: dict, max_dist: float,
             progress_span: tuple = (0.0, 1.0), produced_by: dict = None,
             carry_cbct: bool = False, outputs=()) -> None:
    """The registration proper, once every landmark exists.

    `progress_span` is the slice of the run's progress bar this phase fills.
    It is the whole bar in the Registration mode, which predicts nothing, and
    what the supervised calls left (see `_spans`) in the two modes that make
    them -- otherwise a run that called no other tool would report itself as
    more than half done before it had registered anything.

    `produced_by` maps "ios", "cbct", "ios_landmarks" and "cbct_landmarks" to
    the supervised tool that wrote that folder; one left out is the caller's
    own argument of that name. It decides two things about a refusal: which
    source the message names, and its class -- an empty folder the caller sent
    is theirs to fix (ToolInputError), one a tool returned is a fault on this
    side (RuntimeError).

    `carry_cbct` copies each patient's CBCT, and the transform beside it, into
    that patient's folder: see `_carry_cbct`.

    `outputs` are the `catalogs.OUTPUT_*` the caller ticked, all written in
    the CBCT's space beside the meshes: the landmarks (each mesh's moved by
    its matrix, the CBCT's as they are), the CBCT itself, and each matrix as
    a `.tfm`. Whatever is written, each patient's `scene.json` lists it.
    """
    outputs = set(outputs or ())
    produced_by = produced_by or {}

    def source(role):
        tool = produced_by.get(role)
        return f"{tool}'s output" if tool else f"'{role}'"

    def error_for(*roles):
        return RuntimeError if any(produced_by.get(r) for r in roles) else ToolInputError

    # Before the first file is read: walking and parsing every landmark file
    # takes a while on a large batch, and the last stage a watcher saw was the
    # previous tool's.
    progress.emit(progress_span[0], "reading landmarks")
    paired, unpaired = pipeline.discover(
        ios_dir, cbct_dir,
        ios_source=source("ios"), cbct_source=source("cbct"),
        ios_error=error_for("ios"), cbct_error=error_for("cbct"),
    )
    report["unpaired"] = unpaired

    ios_landmarks = _landmarks_by_jaw(ios_landmark_dir, "intraoral")
    cbct_landmarks = _landmarks_by_jaw(cbct_landmark_dir, "CBCT")
    for found, role, side, directory in (
        (ios_landmarks, "ios_landmarks", "intraoral", ios_landmark_dir),
        (cbct_landmarks, "cbct_landmarks", "CBCT", cbct_landmark_dir),
    ):
        if not found:
            files = len(_landmark_files(directory))
            raise error_for(role)(
                f"No {side} landmark could be read from {source(role)}: "
                f"{files} landmark file(s) (.json or .mrk.json) found, none "
                "readable with points in it."
            )

    sole_patient = len(paired) == 1
    failures = []
    to_review = 0
    total_meshes = sum(len(data["ios"]) for data in paired.values())
    for index, (patient, data) in enumerate(paired.items(), start=1):
        # Per patient, not per mesh: the inner loop is one or two arches, and
        # the counter is what a watcher can act on. The patient key is built
        # from the caller's file names and never travels in a message.
        progress.report(index, len(paired), "patient",
                        start=progress_span[0], end=progress_span[1])
        position = f"patient {index}/{len(paired)}"
        folder = os.path.join(output_dir, data["name"])
        entry = {"cbct": os.path.basename(data["cbct"]), "meshes": {}}
        # Narrowed to this patient BEFORE the jaw is looked at: see _for_patient.
        own_ios = _for_patient(ios_landmarks, patient, sole_patient)
        own_cbct = _for_patient(cbct_landmarks, patient, sole_patient)
        # The CBCT landmark files the registration read, for the same-space
        # outputs to copy once per patient whichever mesh used them.
        used_cbct = set()
        with where.at(position):
            cbct_surface, on_enamel, surface_error = _cbct_surface(data["cbct"], own_cbct)
        if surface_error:
            entry["cbct_surface_error"] = surface_error
        for mesh_index, mesh_path in enumerate(data["ios"], start=1):
            name = os.path.basename(mesh_path)
            arch = _arch(mesh_path)
            here = (f"{position}, arch {arch}" if arch
                    else f"{position}, mesh {mesh_index}/{len(data['ios'])}")
            try:
                with where.at(here):
                    moving = _match_landmarks(mesh_path, own_ios)
                    fixed = _match_landmarks(mesh_path, own_cbct)
                    if not moving or not fixed:
                        role = "ios_landmarks" if not moving else "cbct_landmarks"
                        raise error_for(role)(
                            "No landmark file matches this patient and this mesh's "
                            f"jaw on {'the intraoral' if not moving else 'the CBCT'} "
                            f"side, in {source(role)}."
                        )
                    mesh = _read_mesh(mesh_path)
                    matrix, detail = pipeline.register_one(
                        mesh, moving, fixed, cbct_surface, on_enamel, max_dist=max_dist
                    )
                    destination = os.path.join(
                        folder, f"{os.path.splitext(name)[0]}_{suffix}.vtk"
                    )
                    _write_mesh(mesh.transform(matrix, inplace=False), destination)
                    # splitext, not `destination.replace(".vtk", ...)`: str.replace
                    # rewrites EVERY occurrence, so a mesh whose own stem carries
                    # `.vtk` produced a mangled matrix name beside a correct mesh.
                    np.save(os.path.splitext(destination)[0] + "_matrix.npy", matrix)
                    extra = _mesh_outputs(
                        mesh_path, destination, matrix, own_ios, own_cbct,
                        ios_landmark_dir, output_dir, used_cbct, outputs,
                    )
                review = _needs_review(detail)
                if review:
                    logger.warning("%s: written, but to be checked before use: %s",
                                   here, review)
                    to_review += 1
                entry["meshes"][name] = dict(
                    detail, status="review" if review else "ok",
                    output=os.path.relpath(destination, output_dir),
                    **({"review": review} if review else {}),
                    **extra,
                )
            except Exception as exc:  # noqa: BLE001 - one mesh must not cost the batch
                # Position, class and message: the report naming the mesh is
                # deleted with the job when the run fails, so this line is what
                # an operator has left.
                logger.warning("%s: registration failed (%s: %s)",
                               here, type(exc).__name__, exc)
                logger.debug("Registration traceback", exc_info=True)
                failures.append(exc)
                entry["meshes"][name] = {"status": "failed", "error": str(exc)}
        statuses = [m["status"] for m in entry["meshes"].values()]
        entry["status"] = ("ok" if "ok" in statuses
                           else "review" if "review" in statuses else "failed")
        if entry["status"] != "failed":
            if carry_cbct or catalogs.OUTPUT_CBCT_SCAN in outputs:
                entry.update(_carry_cbct(data["cbct"], folder, output_dir,
                                         with_transform=carry_cbct))
            if catalogs.OUTPUT_LANDMARKS in outputs:
                entry["cbct_landmarks"] = _copy_cbct_landmarks(
                    used_cbct, cbct_landmark_dir, folder, output_dir)
            _write_scene(folder, entry, output_dir, oriented=carry_cbct)
        # Under the name the files carry, never the matching key: see
        # pipeline.patient_name.
        report["patients"][data["name"]] = entry

    report["meshes_registered"] = total_meshes - len(failures)
    report["meshes_failed"] = len(failures)
    report["meshes_to_review"] = to_review
    if failures and len(failures) == total_meshes:
        first, count = _most_common(failures)
        # Self-contained: the report is deleted with a failed job, so pointing
        # at it sends the operator nowhere. Kept as the caller's error when
        # every arch failed on the caller's own input; anything else is ours.
        every_input = all(isinstance(exc, ToolInputError) for exc in failures)
        raise (ToolInputError if every_input else RuntimeError)(
            f"0 of {total_meshes} mesh(es) registered; most common failure: "
            f"{type(first).__name__}: {first} ({count} of {total_meshes})"
        ) from first


def _mesh_outputs(mesh_path: str, destination: str, matrix: np.ndarray,
                  own_ios: dict, own_cbct: dict, ios_landmark_dir: str,
                  output_dir: str, used_cbct: set, outputs: set) -> dict:
    """What one mesh adds beside its registered copy. Returns report fields.

    Records in `used_cbct` the CBCT landmark file this mesh was registered on,
    the same file `_match_landmarks` read, for the patient's copy of them.
    """
    written = {}
    stem = os.path.splitext(destination)[0]
    fixed = _matched_file(mesh_path, own_cbct)
    if fixed is not None:
        used_cbct.add(fixed)
    if catalogs.OUTPUT_LANDMARKS in outputs:
        moving = _matched_file(mesh_path, own_ios)
        if moving is not None:
            source = os.path.join(ios_landmark_dir, moving)
            landmarks = scene.move_landmarks(
                source, matrix, stem + "_lm" + scene.landmark_extension(source))
            written["landmarks"] = os.path.relpath(landmarks, output_dir)
    if catalogs.OUTPUT_TRANSFORMS in outputs:
        written["transform"] = os.path.relpath(
            _write_transform(matrix, stem + "_transform.tfm"), output_dir)
    return written


def _copy_cbct_landmarks(used: set, cbct_landmark_dir: str, folder: str,
                         output_dir: str) -> list:
    """Copy the CBCT landmark files the registration used, as they are: they
    were placed on the very CBCT the meshes are now in, so they need no move."""
    copied = []
    for name in sorted(used):
        destination = os.path.join(folder, os.path.basename(name))
        shutil.copy2(os.path.join(cbct_landmark_dir, name), destination)
        copied.append(os.path.relpath(destination, output_dir))
    return copied


def _write_scene(folder: str, entry: dict, output_dir: str, oriented: bool) -> None:
    """The patient's `scene.json`: every file written for it, all in the
    CBCT's space (see `sadt_areg_common.scene`)."""
    def here(path):
        return os.path.join(output_dir, path)

    files = []
    for field, role in (("oriented_cbct", "reference"), ("cbct_copy", "reference"),
                        ("cbct_transform", "orientation")):
        if entry.get(field):
            files.append((here(entry[field]), role))
    files += [(here(path), "reference landmarks") for path in entry.get("cbct_landmarks", [])]
    for mesh in entry["meshes"].values():
        if mesh["status"] == "failed":
            continue
        files.append((here(mesh["output"]), "registered"))
        if mesh.get("landmarks"):
            files.append((here(mesh["landmarks"]), "moved landmarks"))
        if mesh.get("transform"):
            files.append((here(mesh["transform"]), "registration"))
    scene.write(folder, "the oriented CBCT" if oriented else "the CBCT", files)


def _count(directory, keep) -> int:
    """How many files under `directory` `keep(name)` accepts; 0 when absent."""
    if not directory or not os.path.isdir(str(directory)):
        return 0
    return sum(1 for _root, _dirs, files in os.walk(str(directory))
               for name in files if keep(name))


def _is_mesh(name: str) -> bool:
    return name.lower().endswith(pipeline.SURFACE_EXTENSIONS)


def _is_landmark(name: str) -> bool:
    return name.lower().endswith(pipeline.LANDMARK_EXTENSIONS)


def _is_scan(name: str) -> bool:
    from sadt_areg_common import pairing

    return pairing.is_scan_file(name)


def _compare(tool: str, what: str, sent: int, returned: int) -> None:
    """Say so when a supervised tool hands back fewer files than it was sent.

    The tool itself ran to the end, so nothing raised: the shortfall shows up
    only later, as patients the registration silently has nothing for. Counted
    here, where it is still clear whose output it was. Fewer is a warning, not
    a refusal -- the rest of the batch is still worth registering, and an
    empty return is refused downstream, naming the tool.
    """
    if returned < sent:
        logger.warning("%s returned %d %s for %d sent", tool, returned, what, sent)
    else:
        logger.info("%s returned %d %s for %d sent", tool, returned, what, sent)


def derive_automation(automation: str, ios_landmarks, cbct_landmarks,
                     orient_cbct_first: bool = True) -> tuple:
    """The mode this request really is. Returns `(mode, source)`.

    `source` is "from the data" or "requested". See the note at the call site
    for why only two of the three modes can be read off a folder.
    """
    if automation and automation != catalogs.AUTOMATION_AUTO:
        return automation, "requested"
    if ios_landmarks and cbct_landmarks:
        return catalogs.AUTOMATION_REGISTRATION, "from the data"
    return (catalogs.AUTOMATION_FULLY if orient_cbct_first
            else catalogs.AUTOMATION_SEMI), "from the data"


def main(ios, cbct, output_dir, automation=None, ios_landmarks=None, cbct_landmarks=None,
         cbct_reference=None, landmark_model=None, ios_landmark_model=None,
         crown_model=None, max_dist=None, output_suffix="Reg",
         orient_cbct_first=True, outputs=(), sup=None):
    """Validate, fetch whatever the mode does not supply, then register."""
    started_at = time.monotonic()
    # Two of the three modes are written in the request; the third is a choice.
    #
    # `Registration` is: both landmark sets supplied means predict nothing, there
    # being nothing left to predict. What separates `Fully-Automated` from
    # `Semi-Automated` is NOT in the data -- it is "orient the CBCT first", which
    # nothing in a folder can answer -- so that one is asked as
    # `orient_cbct_first`, a box that says what it does, instead of being hidden
    # inside a three-valued mode nobody could map onto their own files.
    #
    # `cbct_reference` cannot stand in for it either: it is an override, and
    # left empty ASO orients into `_ORIENTATION_FRAME` with its own bundle.
    automation, automation_source = derive_automation(
        automation, ios_landmarks, cbct_landmarks, orient_cbct_first
    )
    allowed = catalogs.AUTOMATION_BY_MODALITY[MODALITY]
    if automation not in allowed:
        raise ToolInputError(
            f"'{automation}' is not a mode {MODALITY} has. It offers: {', '.join(allowed)}."
        )

    suffix = (output_suffix or "Reg").strip() or "Reg"
    if os.sep in suffix or (os.altsep and os.altsep in suffix):
        raise ToolInputError("'output_suffix' is a name fragment, not a path.")

    output_dir = os.path.abspath(str(output_dir))
    os.makedirs(output_dir, exist_ok=True)
    work_dir = os.path.join(output_dir, WORK_DIRNAME)
    os.makedirs(work_dir, exist_ok=True)

    report = {
        "modality": MODALITY,
        "automation": automation,
        # Which of the two it was: a mode name in a report does not say whether
        # anybody chose it.
        "automation_source": automation_source,
        "output_suffix": suffix,
        "patients": {},
    }

    try:
        ios_root, cbct_root = str(ios), str(cbct)
        ios_lm = str(ios_landmarks) if ios_landmarks else None
        cbct_lm = str(cbct_landmarks) if cbct_landmarks else None

        spans = _spans(automation, predict_ios=not ios_lm, predict_cbct=not cbct_lm)
        # Which folders a supervised tool wrote, for `register` to name in a
        # refusal -- and to tell the caller's fault from this side's.
        produced_by = {}

        if automation != catalogs.AUTOMATION_REGISTRATION:
            # Everything the caller did not supply is fetched from the tool that
            # produces it. Checked up front so a request that cannot work comes
            # back in a second rather than after the first prediction.
            for name in ("Crown_Seg", "ALI_IOS", "ALI_CBCT"):
                tools.require(sup, name, f"{automation} IOSCBCT registration")
            if automation == catalogs.AUTOMATION_FULLY:
                tools.require(sup, "ASO", "Fully-Automated IOSCBCT registration")
                # The frame, not a bundle: there is one frame the rest of this
                # chain expects, and ASO resolves the reference that defines it.
                scans_sent = _count(cbct_root, _is_scan)
                cbct_root = tools.orient_cbct(
                    sup, cbct_root, str(cbct_reference or ""), landmark_model,
                    span=spans["orient"], frame=_ORIENTATION_FRAME,
                )
                _compare("ASO", "oriented CBCT(s)", scans_sent, _count(cbct_root, _is_scan))
                produced_by["cbct"] = "ASO"

            meshes_sent = _count(ios_root, _is_mesh)
            labelled = tools.label_crowns(sup, ios_root, crown_model or "", span=spans["crowns"])
            meshes_labelled = _count(labelled, _is_mesh)
            _compare("Crown_Seg", "labelled mesh(es)", meshes_sent, meshes_labelled)
            produced_by["ios"] = "Crown_Seg"
            if not ios_lm:
                ios_lm = tools.predict_ios_landmarks(
                    sup, labelled, ios_landmark_model or "", span=spans["ios_landmarks"]
                )
                # One landmark file per mesh is what ALI_IOS writes.
                _compare("ALI_IOS", "landmark file(s)", meshes_labelled,
                         _count(ios_lm, _is_landmark))
                produced_by["ios_landmarks"] = "ALI_IOS"
            if not cbct_lm:
                scans = _count(cbct_root, _is_scan)
                cbct_lm = tools.predict_cbct_landmarks(
                    sup, cbct_root, landmark_model or "", span=spans["cbct_landmarks"]
                )
                # And one per scan from ALI_CBCT.
                _compare("ALI_CBCT", "landmark file(s)", scans, _count(cbct_lm, _is_landmark))
                produced_by["cbct_landmarks"] = "ALI_CBCT"
            ios_root = labelled

        if not ios_lm or not cbct_lm:
            raise ToolInputError(
                "The Registration mode takes the landmarks already computed: send "
                "both 'ios_landmarks' and 'cbct_landmarks', or use a mode that "
                "predicts them."
            )

        register(
            ios_dir=ios_root, cbct_dir=cbct_root,
            ios_landmark_dir=ios_lm, cbct_landmark_dir=cbct_lm,
            output_dir=output_dir, suffix=suffix, report=report,
            max_dist=float(max_dist) if max_dist else geometry.ICP_MAX_DIST_MM,
            # Whatever the supervised calls left; the whole bar in the
            # Registration mode, which makes none of them.
            progress_span=spans["register"],
            produced_by=produced_by,
            # The meshes are registered onto the CBCT ASO oriented, a volume
            # the caller never had: without it beside them, they sit nowhere
            # near the scan the caller opens them over.
            carry_cbct=produced_by.get("cbct") == "ASO",
            outputs=outputs,
        )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    report["duration_seconds"] = round(time.monotonic() - started_at, 2)
    with open(os.path.join(output_dir, REPORT_NAME), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    failed = report.get("meshes_failed", 0)
    total = report.get("meshes_registered", 0) + failed
    to_review = report.get("meshes_to_review", 0)
    unpaired = len(report.get("unpaired", {}))
    # WARNING when partial: a batch that registered 30 of 40 arches finishes
    # "successfully", and this line is where the other ten show -- along with
    # the arches written but far from their landmarks, and the patients never
    # registered at all because one modality was missing. "4 of 4 registered"
    # once summed up a batch of six patients of which two were paired, both
    # wrong.
    logger.log(
        logging.WARNING if failed or to_review or unpaired else logging.INFO,
        "AREG_IOSCBCT: %d of %d mesh(es) registered, %d failed, %d to review, over "
        "%d patient(s) in %.1fs%s", total - failed, total, failed, to_review,
        len(report["patients"]), report["duration_seconds"],
        f"; {unpaired} patient(s) not registered, one modality missing"
        if unpaired else "")
    return output_dir
