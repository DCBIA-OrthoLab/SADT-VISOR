"""Crown segmentation on intraoral surface scans, via shapeaxi.

Labels every tooth of a mesh with its Universal number, as a point-data array
on a copy of the surface. That array is the precondition for ALI's IOS landmark
identification and for the IOS modes of ASO, AREG and FlexReg, which is why
this is its own tool rather than a helper inside any of them -- and, since the
split, why the server sequences CrownSeg before ALI instead of ALI importing
this module.

Nothing here is ported. The Slicer modules ran the `dentalmodelseg` executable
out of Slicer's own bin directory, and that executable is only the
console-script entry point of the `shapeaxi` PyPI package
(`dentalmodelseg = shapeaxi.dental_model_seg:cml`). There is no Slicer binary
to shell out to, so `shapeaxi.dental_model_seg.main` is called directly with
the namespace its own `cml()` would have built.

What the move out of the server changed: scratch space lives under the caller's
`output_dir`, `segment_crowns()` returns the run report rather than a
`CrownSegRun`, zip extraction is gone (the server unpacks archives before
`run()` is called), the model is a required argument rather than a name looked
up in the data store, and the GPU semaphore is gone because each call is its
own process.
"""

import contextlib
import io
import json
import logging
import os
from pathlib import Path
import shutil
import time
from argparse import Namespace

from .errors import ToolInputError, ToolUnavailableError

logger = logging.getLogger(__name__)


# The tool's own name, derived rather than written down. `describe.py` publishes
# `tool_dir.name` as the schema's `name`, so taking it from the same place is
# what keeps a report and the schema from disagreeing -- which they already did:
# this file said "CrownSeg" and "BatchDentalSeg" long after the folders were
# renamed to the underscored convention, and nothing noticed because nothing
# reads the field yet.
#
# `parents[2]` is <tool>/src/<package>/thisfile.py -> <tool>.
TOOL_NAME = Path(__file__).resolve().parents[2].name

# Extensions shapeaxi's reader handles and this tool therefore discovers.
SURFACE_EXTENSIONS = (".vtk", ".stl")

# Point-data array names that already carry per-tooth labels. A mesh holding
# any of them is segmented and is passed through untouched -- re-running the
# network on it would cost minutes and change nothing.
LABEL_ARRAY_NAMES = ("PredictedID", "UniversalID", "Universal_ID", "FDI_ID")

DEFAULT_ARRAY_NAME = "Universal_ID"
FDI_ARRAY_NAME = "FDI_ID"
DEFAULT_SUFFIX = "Seg"

# The array a run writes to, named after the numbering that went into it.
#
# **The name has to follow the values, and this is not tidiness.** shapeaxi's
# `ConvertFDI` converts the integers in place and writes them back under the
# SAME array name, so asking for FDI used to produce an array literally called
# `Universal_ID` holding FDI numbers. The two systems overlap across almost
# their whole range -- Universal 1..32 against FDI 11..18, 21..28, 31..38,
# 41..48 -- so a consumer reading that array does not fail, it reads a
# different tooth: FDI 18 is the upper right third molar where Universal 18 is
# the lower left second molar, and FDI 33 (lower left canine) is the value
# shapeaxi reserves for GUM in Universal.
#
# Naming the FDI array differently is what turns that into a refusal. Every
# tool that consumes these meshes -- ALI's IOS landmarks, ASO, AREG and
# FlexReg -- looks for `Universal_ID`, `PredictedID` or `UniversalID`, so an
# FDI mesh now carries no name they know and they say so, instead of
# registering a patient onto the wrong teeth. That is the right answer: none of
# them speaks FDI, and every legacy caller passed `fdi: 0`.
_ARRAY_NAMES = {"Universal": DEFAULT_ARRAY_NAME, "FDI": FDI_ARRAY_NAME}


def array_name_for(numbering: str) -> str:
    """The array name that goes with a numbering system."""
    return _ARRAY_NAMES.get(numbering, DEFAULT_ARRAY_NAME)

WORK_DIRNAME = ".crownseg_work"

# The published crown-segmentation checkpoint, and the token every one of them
# is named with. Fly-by-CNN publishes its weights under the training run that
# produced them -- `<date>_val-loss<number>.pth` -- and that name is what
# `scripts/data-manifest.yml` stages, byte for byte, under every tool that
# needs it.
PUBLISHED_CHECKPOINT = "07-21-22_val-loss0.169.pth"
CROWN_CHECKPOINT_MARKER = "val-loss"

_INSTALL_HINT = (
    "CrownSeg's engine is an optional extra. Install it with "
    "`uv sync --extra segmentation` in tools/Crown_Seg. A deployment image "
    "builds it with `--all-extras`; if this appears there, the image was built "
    "without them."
)


def _restore_moved_class(saxi_nets) -> bool:
    """Work around an upstream bug: shapeaxi 2.0.x cannot find its own class.

    `DentalModelSeg` moved from `shapeaxi.saxi_nets` to
    `shapeaxi.saxi_nets_lightning`, but `shapeaxi/dental_model_seg.py` -- the
    module behind shapeaxi's own `dentalmodelseg` console script, and the only
    supported way in -- still reads it off `saxi_nets`. Every 2.0.x release
    (2.0.0, 2.0.1, 2.0.2) has this; 1.x referenced the right module. So
    `dental_model_seg.main()` raises AttributeError before touching a mesh, and
    the pre-port tool fails in exactly the same way on the deployed image.

    Two lines rather than reimplementing `main()`: this keeps the entry point
    shapeaxi supports, and the `hasattr` guard makes the workaround vanish by
    itself the day upstream puts the name back. Report it at
    DCBIA-OrthoLab/ShapeAXI; delete this function once a release carries the fix.
    """
    if hasattr(saxi_nets, "DentalModelSeg"):
        return False

    from shapeaxi import saxi_nets_lightning

    target = getattr(saxi_nets_lightning, "DentalModelSeg", None)
    if target is None:
        raise ToolUnavailableError(
            "This shapeaxi exposes DentalModelSeg in neither saxi_nets nor "
            "saxi_nets_lightning; the pinned version is 2.0.2."
        )
    saxi_nets.DentalModelSeg = target
    logger.warning(
        "shapeaxi %s does not expose DentalModelSeg where its own "
        "dental_model_seg module looks for it; pointing it at "
        "saxi_nets_lightning (see _restore_moved_class)",
        _shapeaxi_version(),
    )
    return True


def _shapeaxi_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("shapeaxi")
    except PackageNotFoundError:  # pragma: no cover - installed by definition here
        return "unknown"


def _import_dental_model_seg():
    """Import shapeaxi's segmentation module, lazily.

    Deferred so that importing this package -- which `scripts/describe.py` does
    on every CI run to publish the schema -- costs nothing. It also means a
    venv without the extra still loads and reports the schema; only an actual
    segmentation fails, naming what is missing.
    """
    try:
        from shapeaxi import dental_model_seg, saxi_nets
    except ImportError as exc:
        raise ToolUnavailableError(
            "{} (missing: {})".format(_INSTALL_HINT, exc.name or "shapeaxi")
        ) from exc

    _restore_moved_class(saxi_nets)
    return dental_model_seg


def resolve_device(requested: str = None) -> str:
    """The device string shapeaxi accepts: exactly "cpu" or "cuda:0"."""
    import torch

    wanted = (requested or "cpu").strip().lower()
    if wanted.startswith("cuda") and torch.cuda.is_available():
        return "cuda:0"
    if wanted.startswith("cuda"):
        logger.warning("device=%s requested but CUDA is unavailable; falling back to CPU", wanted)
    return "cpu"


# ---------------------------------------------------------------------------
# Model bundle
# ---------------------------------------------------------------------------

def _is_crown_checkpoint(name: str) -> bool:
    """Whether this file name is a crown-segmentation checkpoint.

    **The name is the only thing that can answer this, and that is measured
    rather than assumed.** The obvious alternative -- open the checkpoint and
    look at what it holds -- does not separate it from its neighbours: this
    network and ALI_IOS's landmark networks are both monai UNets, and their
    state dicts start with the very same key
    (`model.0.conv.unit0.conv.weight`), so telling them apart would mean
    comparing tensor shapes. In the folder this is actually pointed at,
    `DATA/ALI/models/`, that is 244 files and 12 GB to read to find 6.6 MB.

    So it matches the token every published crown checkpoint carries. `_` and
    `-` are folded together because the two spellings are both in the wild, and
    neither ALI bundle can collide with it: theirs are `Upper_O_model.pth`,
    `Lower_MG_v6.pth` and `<landmark>_Net_<scale>.pth`.

    A checkpoint named by some other rule is not recognised, and that is the
    known cost: `find_checkpoint` then says what it looked for, and pointing
    `model` straight at the `.pth` still works.
    """
    if not name.lower().endswith(".pth"):
        return False
    return CROWN_CHECKPOINT_MARKER in name.lower().replace("_", "-")


def find_checkpoint(model_path: str) -> str:
    """The checkpoint `model` names -- given the file itself, or a folder of them.

    `model` used to be a single `.pth` and nothing else. It still is when a
    caller passes one, which is what keeps a direct call and every existing
    request working unchanged. What is new is the DIRECTORY: the server hands a
    hosted-model argument the whole of `DATA/<tool>/models/` when the caller
    named no bundle, and a tool reaching this one through the supervisor passes
    on the directory IT was handed -- ALI_IOS segmenting an unlabelled mesh
    mid-run is exactly that. Neither knows where the file sits inside, so this
    finds it.

    Walked recursively, so a checkpoint filed under a bundle folder of its own
    is found as readily as one sitting at the top.
    """
    model_path = os.fspath(model_path)
    if os.path.isfile(model_path):
        return model_path
    if not os.path.isdir(model_path):
        raise ToolInputError(f"Crown-segmentation checkpoint not found: {model_path}")

    # Relative to the folder, never absolute: these messages reach the client
    # verbatim and the server's own paths are not its business.
    candidates = sorted(
        os.path.relpath(os.path.join(root, name), model_path)
        for root, _dirs, files in os.walk(model_path)
        for name in files
        if _is_crown_checkpoint(name)
    )
    bundle = os.path.basename(model_path.rstrip(os.sep))

    if len(candidates) == 1:
        return os.path.join(model_path, candidates[0])
    if candidates:
        # Two vintages under one folder. Picking here -- the first, the newest,
        # the largest -- would leave which weights ran unrecorded, and which
        # model vintage ran must never be a surprise.
        raise ToolInputError(
            "'{}' holds {} crown-segmentation checkpoints ({}). Point 'model' at "
            "the one to use -- choosing here would leave which weights ran "
            "unrecorded.".format(bundle, len(candidates), ", ".join(candidates))
        )
    raise ToolInputError(
        "'{}' holds no crown-segmentation checkpoint: a .pth whose name carries "
        "the '{}' token every published one is named with, e.g. '{}'. Fetch it "
        "with `setup-models.sh` -- the manifest stages it under every tool that "
        "needs it -- or point 'model' straight at the .pth file.".format(
            bundle, CROWN_CHECKPOINT_MARKER, PUBLISHED_CHECKPOINT)
    )


# ---------------------------------------------------------------------------
# Input discovery
# ---------------------------------------------------------------------------

def is_segmented(mesh_path: str) -> bool:
    """True when the mesh already carries a per-point tooth-label array.

    Reads only the point-data array NAMES, never their values: this decides
    whether to spend minutes of GPU time, and nothing about the patient.
    """
    import vtk

    extension = os.path.splitext(mesh_path)[1].lower()
    if extension == ".stl":
        # An STL has no point data at all by construction, so it can never be
        # segmented. Reading it anyway keeps one code path.
        reader = vtk.vtkSTLReader()
    elif extension == ".vtk":
        reader = vtk.vtkPolyDataReader()
    else:
        return False

    reader.SetFileName(mesh_path)
    reader.Update()
    point_data = reader.GetOutput().GetPointData()
    names = {point_data.GetArrayName(index) for index in range(point_data.GetNumberOfArrays())}
    return bool(names & set(LABEL_ARRAY_NAMES))


def discover_meshes(input_path: str) -> list:
    """Resolve the `meshes` argument into a list of surface files.

    One mesh, or a folder of them for a batch. Folder scanning is recursive,
    and the tree is what the output mirrors.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if os.path.isfile(input_path):
        if not input_path.lower().endswith(SURFACE_EXTENSIONS):
            raise ToolInputError(
                f"'{os.path.basename(input_path)}' is not a surface mesh. Expected one of: "
                f"{', '.join(SURFACE_EXTENSIONS)}."
            )
        return [input_path]

    found = [
        os.path.join(root, name)
        for root, _dirs, files in os.walk(input_path)
        for name in sorted(files)
        if name.lower().endswith(SURFACE_EXTENSIONS)
    ]
    if not found:
        raise ToolInputError(
            f"No surface mesh found in the input. Supported extensions: "
            f"{', '.join(SURFACE_EXTENSIONS)}."
        )
    return sorted(found)


def _input_root(input_path: str, meshes: list) -> str:
    """The directory output paths are made relative to, so a batch keeps its
    tree instead of being flattened into one folder where two patients named
    scan.vtk overwrite each other."""
    if os.path.isdir(input_path):
        return input_path
    return os.path.dirname(meshes[0])


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def _run_shapeaxi(csv_path: str, output_dir: str, model_path: str, input_root: str,
                  array_name: str, suffix: str, device: str, fdi: bool,
                  num_workers: int) -> None:
    """Call shapeaxi's own `main()` with the namespace its `cml()` would build.

    stdout is swallowed on purpose. shapeaxi prints "Saving results to <path>"
    for every mesh, and that path carries the patient's own file name, which
    must not reach a log. The exception it raises on a real failure is
    untouched.
    """
    dental_model_seg = _import_dental_model_seg()

    args = Namespace(
        surf=None,
        csv=csv_path,
        model=model_path,
        suffix=suffix,
        out=output_dir,
        # Must be >= 1: shapeaxi builds its loader with persistent_workers=True,
        # which PyTorch rejects at 0.
        num_workers=max(1, int(num_workers)),
        crown_segmentation=0,
        array_name=array_name,
        fdi=1 if fdi else 0,
        overwrite=0,
        device=device,
        vtk_folder=input_root,
    )

    with contextlib.redirect_stdout(io.StringIO()):
        dental_model_seg.main(args)


def _predicted_path(output_dir: str, csv_stem: str, suffix: str, mesh: str,
                    input_root: str) -> str:
    """Where shapeaxi's csv branch writes the segmented copy of `mesh`.

    Mirrors `save_data_vtk_from_csv`: the input path minus its extension, plus
    `_<suffix>.vtk`, with `vtk_folder` stripped off the front, filed under
    `<out>/<csv stem>_<suffix>/`. Recomputed rather than parsed back out of
    shapeaxi's stdout, which is where the file names this tool must not log
    would have had to travel.
    """
    without_extension = os.path.splitext(mesh)[0] + f"_{suffix}.vtk"
    relative = without_extension.replace(input_root, "").lstrip(os.sep)
    return os.path.normpath(os.path.join(output_dir, f"{csv_stem}_{suffix}", relative))


def segment_crowns(
    input_path: str,
    model_path: str,
    output_dir: str,
    array_name: str = DEFAULT_ARRAY_NAME,
    suffix: str = DEFAULT_SUFFIX,
    fdi: bool = False,
    skip_segmented: bool = True,
    device: str = "cuda",
    num_workers: int = 2,
) -> dict:
    """Label the teeth of one mesh or a whole folder of them.

    `skip_segmented=True` passes an already-labelled mesh through unchanged
    instead of re-running the network on it, which is what makes a mixed batch
    of raw and pre-segmented meshes one call.
    """
    started_at = time.monotonic()

    array_name = (array_name or DEFAULT_ARRAY_NAME).strip() or DEFAULT_ARRAY_NAME
    suffix = (suffix or DEFAULT_SUFFIX).strip() or DEFAULT_SUFFIX

    output_dir = os.fspath(output_dir)

    # `model` may be the checkpoint itself or a folder to find it in; see
    # find_checkpoint. Resolved HERE, before the scratch directory exists: a
    # request the tool refuses must leave the caller's output directory exactly
    # as it found it, and a 422 that has already written a folder into the
    # response's own directory is a refusal that made a mess.
    checkpoint = find_checkpoint(model_path)

    meshes = discover_meshes(os.fspath(input_path))
    input_root = _input_root(os.fspath(input_path), meshes)

    work_dir = os.path.join(output_dir, WORK_DIRNAME)
    os.makedirs(work_dir, exist_ok=True)
    try:
        return _segment(
            meshes=meshes,
            input_root=input_root,
            model_path=checkpoint,
            output_dir=output_dir,
            work_dir=work_dir,
            array_name=array_name,
            suffix=suffix,
            fdi=fdi,
            skip_segmented=skip_segmented,
            device=device,
            num_workers=num_workers,
            started_at=started_at,
        )
    finally:
        # `finally`, not "after the run and again in an except": the scratch
        # directory used to survive every error path that was not shapeaxi's
        # own, and it holds a csv listing the patient's file paths.
        shutil.rmtree(work_dir, ignore_errors=True)


def _segment(meshes, input_root, model_path, output_dir, work_dir, array_name, suffix,
             fdi, skip_segmented, device, num_workers, started_at) -> dict:
    """The run itself, once the arguments are known to be usable.

    Split out so `segment_crowns` can own the scratch directory's lifetime in
    one `try/finally` instead of repeating the cleanup on each way out.
    """
    already_segmented, to_segment = [], []
    for mesh in meshes:
        (already_segmented if skip_segmented and is_segmented(mesh) else to_segment).append(mesh)

    # Probed HERE, before the split decides whether the engine is ever reached.
    # The engine is imported inside `_run_shapeaxi`, which only runs when
    # something needs segmenting -- so a batch that happens to be entirely
    # pre-labelled returned a clean report on a deployment where the
    # `segmentation` extra is not installed at all, and the same deployment
    # failed the moment one raw mesh appeared. A tool that can serve some
    # batches and not others is not available; it just has not been asked the
    # right question yet.
    #
    # Not raised here: a pre-labelled batch is still legitimately served, and
    # refusing it would break the Crown_Seg -> ALI chain for callers who
    # segmented elsewhere. It is RECORDED, so the report says what this run
    # could and could not have done. The real fix is server-side -- the
    # registry must refuse to serve a tool whose declared extras are absent,
    # the same way it refuses a tool whose schema no longer matches its source.
    try:
        _import_dental_model_seg()
        engine_available, engine_error = True, None
    except Exception as exc:  # ToolUnavailableError, or anything its import raises
        engine_available, engine_error = False, f"{type(exc).__name__}: {exc}"

    records = {}
    produced = []

    # A mesh that already carries labels is copied, not re-predicted. Copied
    # rather than referenced in place because the caller is handed one output
    # tree, and because the input may live on a read-only mount.
    for mesh in already_segmented:
        relative = os.path.relpath(mesh, input_root)
        base = os.path.splitext(os.path.basename(mesh))[0]
        destination = os.path.join(
            output_dir, os.path.dirname(relative), f"{base}_{suffix}.vtk"
        )
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        _write_as_vtk(mesh, destination)
        # `engine_unavailable` when this deployment could not have segmented
        # anything anyway. The name is the server's existing vocabulary for
        # "this installation cannot do that" -- ToolUnavailableError, answered
        # as 501 -- so the status reads with it rather than beside it.
        # `already_segmented` means "it did not need doing";
        # `engine_unavailable` means "it did not need doing, AND could not have
        # been done". `engine_error` holds the reason.
        #
        # Putting it in the status rather than only in a field further down is
        # the right PLACE, not a guarantee it is seen. `ALI_IOS` DOES read
        # this report -- it is how it tells a mesh that was labelled from one
        # that could not be -- so the vocabulary here is a contract, not a
        # note for a human.
        records[relative] = {
            "status": "already_segmented" if engine_available else "engine_unavailable",
            "input": relative,
            "produced": [destination],
        }
        produced.append(destination)

    if to_segment:
        device = resolve_device(device)

        # Written into the work dir. The Slicer module wrote this csv into the
        # extension's own source folder, which breaks a read-only install and
        # leaves a file behind pointing at the patient's data.
        csv_path = os.path.join(work_dir, "crownseg_input.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as handle:
            handle.write("surf\n")
            for mesh in to_segment:
                handle.write(f"{mesh}\n")

        logger.info("CrownSeg: segmenting %d mesh(es) on %s", len(to_segment), device)
        _run_shapeaxi(
            csv_path=csv_path,
            output_dir=output_dir,
            model_path=model_path,
            input_root=input_root,
            array_name=array_name,
            suffix=suffix,
            device=device,
            fdi=fdi,
            num_workers=num_workers,
        )

        csv_stem = os.path.splitext(os.path.basename(csv_path))[0]
        for mesh in to_segment:
            relative = os.path.relpath(mesh, input_root)
            predicted = _predicted_path(output_dir, csv_stem, suffix, mesh, input_root)
            if os.path.isfile(predicted):
                records[relative] = {"status": "segmented", "input": relative,
                                     "produced": [predicted]}
                produced.append(predicted)
            else:
                # One mesh shapeaxi could not write must not cost the batch.
                records[relative] = {
                    "status": "failed",
                    "input": relative,
                    "produced": [],
                    "error": "the segmentation produced no output for this mesh",
                }

    if not produced:
        raise RuntimeError("CrownSeg produced no segmented mesh for any input.")

    report = {
        "tool": TOOL_NAME,
        # WHICH weights ran, by name. `model` may now be a folder holding
        # several, so the report has to say which one was picked out of it --
        # the argument no longer answers that on its own.
        "checkpoint": os.path.basename(model_path),
        "array_name": array_name,
        "suffix": suffix,
        "numbering": "FDI" if fdi else "Universal",
        "device": device if to_segment else None,
        "cases": records,
        # Absolute paths of every mesh that now carries tooth labels, whether
        # this run produced them or found them already labelled. This is what a
        # caller sequencing CrownSeg before ALI reads.
        "segmented_meshes": produced,
        # So a caller can tell "nothing needed segmenting" from "nothing COULD
        # have been segmented". Without it the two produce an identical report.
        "engine_available": engine_available,
        "engine_error": engine_error,
        "summary": {
            "total": len(meshes),
            "segmented": sum(1 for r in records.values() if r["status"] == "segmented"),
            # Counted off the records rather than off `already_segmented`, so
            # these two never contradict the per-mesh statuses above: a
            # degraded run reports 0 already_segmented and N
            # engine_unavailable, not N of both.
            "already_segmented": sum(
                1 for r in records.values() if r["status"] == "already_segmented"
            ),
            "engine_unavailable": sum(1 for r in records.values() if r["status"] == "engine_unavailable"),
            "failed": sum(1 for r in records.values() if r["status"] == "failed"),
        },
        "duration_seconds": round(time.monotonic() - started_at, 2),
    }

    with open(os.path.join(output_dir, "run_report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    return report


def _write_as_vtk(source: str, destination: str) -> None:
    """Copy a mesh to `destination` as .vtk polydata.

    A pre-segmented input can be an .stl, and shapeaxi's outputs are always
    .vtk; normalizing here means the caller gets one format whichever branch a
    mesh took, instead of an .stl that every downstream reader then has to
    special-case. (The Slicer module copied it verbatim, and its landmark CLI
    then discovered only .vtk -- so those meshes were silently never used.)
    """
    import vtk

    extension = os.path.splitext(source)[1].lower()
    reader = vtk.vtkSTLReader() if extension == ".stl" else vtk.vtkPolyDataReader()
    reader.SetFileName(source)
    reader.Update()

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(destination)
    writer.SetInputData(reader.GetOutput())
    writer.Write()
