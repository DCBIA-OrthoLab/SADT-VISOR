"""nnUNet v2 inference, isolated from the rest of the AMASSS pipeline.

Three things this module gets right that the original Slicer CLI did not:

1. No `nnUNet_results` environment variable. The CLI set it before spawning
   `nnUNetv2_predict`, and `os.environ` is process-global: two overlapping
   AMASSS runs would overwrite each other's model path.
   `initialize_from_trained_model_folder` takes an explicit path instead.
2. No output-file polling. The CLI killed the predictor once the output file
   stopped growing for three seconds, which could interrupt nnUNet
   mid-postprocessing. The Python API simply returns when it is done.
3. The resampling runs on the GPU too (see `_enable_gpu_resampling`), which is
   where the run time actually went -- the network was an eighth of it.

The server-side port also held a `threading.BoundedSemaphore` here, because
every tool shared one process. A tool is now its own process invoked by the
runner, so serialising GPU work is the server's job and the semaphore is gone.

torch and nnunetv2 are still imported lazily even though the lockfile
guarantees them: `scripts/describe.py` imports this package on every CI run to
publish the schema, and that must not pay for a CUDA stack.
"""

import glob
import inspect
import logging
import os

from . import progress
from .errors import ModelNotFoundError

logger = logging.getLogger(__name__)

CHECKPOINT_NAME = "checkpoint_final.pth"
PLANS_FOLDER_PATTERN = "*__nnUNetPlans__3d_fullres"


def resolve_device(requested: str) -> str:
    """Return the device to actually use, falling back to CPU when needed."""
    import torch

    wanted = (requested or "cpu").strip().lower()
    if wanted.startswith("cuda") and not torch.cuda.is_available():
        logger.warning(
            "device=%s requested but CUDA is unavailable; falling back to CPU", requested
        )
        # The operator's concern: the result is the same, only many times
        # slower, and the fix is in the deployment.
        progress.log(
            "a GPU was requested but none is visible; segmenting on the CPU",
            "warning",
        )
        return "cpu"
    return wanted


def find_model_folder(model_root: str, structure_code: str):
    """Locate the trained nnUNet folder for one structure, or None.

    Layout expected under the model bundle:
        <model_root>/<CODE>/**/<Dataset...>__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth

    A candidate is only accepted once its fold_0 checkpoint is confirmed
    present, so a half-copied bundle degrades to "this structure is
    unavailable" (reported to the caller) rather than crashing the run.
    """
    structure_root = os.path.join(model_root, structure_code)
    if not os.path.isdir(structure_root):
        return None

    pattern = os.path.join(structure_root, "**", PLANS_FOLDER_PATTERN)
    for candidate in sorted(glob.glob(pattern, recursive=True)):
        if os.path.isfile(os.path.join(candidate, "fold_0", CHECKPOINT_NAME)):
            return candidate

    # Also accept the plans folder being the structure folder itself.
    if os.path.isfile(os.path.join(structure_root, "fold_0", CHECKPOINT_NAME)):
        return structure_root
    return None


def why_no_model(model_root: str, structure_code: str) -> str:
    """Why `find_model_folder` found nothing for this structure, in words.

    In words and not as a path, because the operator reads it through a
    redaction that turns every path into `<path>`: "MAND: <path>" says nothing,
    "MAND: folder missing" says what to copy.
    """
    structure_root = os.path.join(model_root, structure_code)
    if not os.path.isdir(structure_root):
        return "folder missing"
    pattern = os.path.join(structure_root, "**", PLANS_FOLDER_PATTERN)
    if not glob.glob(pattern, recursive=True):
        return "no 3d_fullres plans folder"
    return "no fold-0 final checkpoint under the 3d_fullres plans folder"


def _last_line(exc: BaseException) -> str:
    """The last non-empty line of an exception's message, or its class name.

    nnUNet's errors are long and end on their cause, and the server cuts a
    reason at about 300 characters -- so the cause is what is kept.
    """
    lines = [line.strip() for line in str(exc).splitlines() if line.strip()]
    return lines[-1] if lines else type(exc).__name__


def _build_predictor(device: str, tile_step_size: float):
    """Instantiate an nnUNetPredictor, tolerating nnUNet's renamed kwargs.

    nnUNet 2.x renamed `perform_everything_on_gpu` to
    `perform_everything_on_device` mid-series; passing whichever the installed
    version declares keeps this working across the range.
    """
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    options = {
        # The one knob here that changes the segmentation, so it is an argument
        # rather than something tuned in place. See run()'s docstring.
        "tile_step_size": float(tile_step_size),
        "use_gaussian": True,
        # Equivalent to the CLI's --disable_tta: no test-time mirroring.
        "use_mirroring": False,
        "device": torch.device(device),
        "verbose": False,
        "verbose_preprocessing": False,
        "allow_tqdm": False,
    }
    accepted = set(inspect.signature(nnUNetPredictor.__init__).parameters)
    for name in ("perform_everything_on_device", "perform_everything_on_gpu"):
        if name in accepted:
            options[name] = device.startswith("cuda")
            break

    predictor = nnUNetPredictor(**{k: v for k, v in options.items() if k in accepted})
    # AFTER the constructor, which is what sets it: `nnUNetPredictor.__init__`
    # turns cuDNN autotuning ON, and autotuning picks a convolution by TIMING
    # candidates -- so the algorithm, and with it the rounding, depends on how
    # busy the card was. Measured on this pipeline before anything overlapped:
    # seven runs of one untouched scan produced THREE different masks, 205-265
    # voxels apart. With structures now predicted side by side that stops being
    # a rare coincidence and becomes the normal case.
    #
    # It costs Dice 0.99992 against the autotuned configuration -- roughly 270x
    # smaller than the 0.978 the GPU resampling already costs on the same
    # structure -- and buys a property the tool did not have: the same scan
    # segments to the same mask whatever else the machine is doing.
    torch.backends.cudnn.benchmark = False
    return predictor


# The resampler nnUNet's own plans name by default, and the only one we are
# willing to substitute. A bundle asking for anything else (no_resampling, a
# custom function) was configured that way deliberately and its geometry is not
# ours to reinterpret.
_STOCK_RESAMPLER = "resample_data_or_seg_to_shape"
_RESAMPLING_KEYS = ("resampling_fn_data", "resampling_fn_probabilities")


def _enable_gpu_resampling(predictor, device: str) -> bool:
    """Point this predictor's resamplers at the GPU. Returns whether it applied.

    Resampling, not inference, is what makes AMASSS slow: nnUNet's defaults are
    scipy splines on one core and outweigh the network by roughly seven to one.
    nnUNet ships torch equivalents, so there is nothing to reimplement, only to
    select.

    Selected by NAME: nnUNet resolves both resampling functions out of the
    configuration dict via `recursive_find_resampling_fn_by_name`, so rewriting
    the two names redirects both ends. No monkeypatching.

    Mutating that dict is safe because PlansManager hands out a `deepcopy`: it
    touches neither the shared plans nor a concurrent run, and the
    `torch.device` put in here never reaches the `plans.json` nnUNet writes
    beside its output (which `json.dump` could not serialize).
    """
    if not device.startswith("cuda"):
        return False

    try:
        # Imported to prove it EXISTS, then named rather than called: nnUNet
        # resolves both resampling functions out of the configuration dict by
        # name. Using `__name__` below keeps the probe and the name it writes
        # from being two literals that can drift apart.
        from nnunetv2.preprocessing.resampling.resample_torch import (
            resample_torch_fornnunet,
        )
    except ImportError:
        logger.info("This nnUNet has no torch resampler; keeping the scipy one")
        return False

    import torch

    configuration_manager = predictor.configuration_manager
    configuration = configuration_manager.configuration

    if any(configuration.get(key) != _STOCK_RESAMPLER for key in _RESAMPLING_KEYS):
        logger.info("Model plans request a non-default resampler; leaving it alone")
        return False

    for key in _RESAMPLING_KEYS:
        configuration[key] = resample_torch_fornnunet.__name__
        # 'linear' is order 1, already what the plans ask for on the
        # probabilities. The input data drops from order 3 to order 1 (torch
        # has no 3D cubic interpolation): that is the whole numerical
        # difference, and it is what `gpu_resampling=False` turns off.
        configuration[f"{key}_kwargs"] = {
            "is_seg": False,
            "device": torch.device(device),
            "mode": "linear",
        }

    # Both are `@property @lru_cache`, so a value read before this point would
    # otherwise outlive the swap.
    manager_class = type(configuration_manager)
    for key in _RESAMPLING_KEYS:
        getattr(manager_class, key).fget.cache_clear()

    return True


def predict_folder(
    model_folder: str,
    input_dir: str,
    output_dir: str,
    device: str,
    tile_step_size: float,
    gpu_resampling: bool,
) -> None:
    """Segment every `*_0000.nii.gz` in `input_dir`, writing masks to `output_dir`.

    A whole folder per call is deliberate: the model is loaded once per
    structure rather than once per (scan x structure), which on a batch run is
    the difference between N*S and S checkpoint loads.
    """
    os.makedirs(output_dir, exist_ok=True)

    try:
        predictor = _build_predictor(device, tile_step_size)
        # Explicit path: no nnUNet_results env var, hence no cross-run race.
        predictor.initialize_from_trained_model_folder(
            model_folder,
            use_folds=(0,),
            checkpoint_name=CHECKPOINT_NAME,
        )
    except Exception as exc:
        raise RuntimeError(
            f"loading the nnUNet model failed ({type(exc).__name__}: {_last_line(exc)})"
        ) from exc

    on_gpu = bool(gpu_resampling) and _enable_gpu_resampling(predictor, device)
    logger.info(
        "nnUNet predicting on %s (tile_step_size=%s, gpu_resampling=%s)",
        device, tile_step_size, on_gpu,
    )

    in_process = on_gpu or not device.startswith("cuda")
    try:
        if in_process:
            _predict_in_process(predictor, input_dir, output_dir)
        else:
            _predict_with_workers(predictor, input_dir, output_dir)
    except Exception as exc:
        raise RuntimeError(f"nnUNet prediction failed on {device} ({_cause(exc)})") from exc


def _predict_in_process(predictor, input_dir: str, output_dir: str) -> None:
    """Every stage in this process.

    With GPU resamplers because `predict_from_files` fans preprocessing and
    export out to SPAWNED processes, each of which would need its own CUDA
    context to run them -- trading away the CPU/GPU overlap, a smaller loss
    than the resampling win. On a CPU device because there is no overlap to
    keep: the workers would compete with the network for the same cores, and a
    worker that dies -- almost always the kernel reclaiming RAM -- surfaces
    only as "Background workers died", its cause printed to a stderr no
    operator sees. Both paths run the same preprocessor, network call and
    export on the same float32 data, so the masks are the same.
    """
    predictor.predict_from_files_sequential(
        input_dir,
        output_dir,
        save_probabilities=False,
        overwrite=True,
    )


def _predict_with_workers(predictor, input_dir: str, output_dir: str) -> None:
    """The card with scipy resamplers: nnUNet's own worker processes.

    The workers are what overlap one scan's resampling with the next one's
    inference, and that resampling is most of a run, so they stay -- with
    their death translated by `_cause`.
    """
    predictor.predict_from_files(
        input_dir,
        output_dir,
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=2,
        num_processes_segmentation_export=2,
    )


def _cause(exc: BaseException) -> str:
    """"Type: last line", or what a dead nnUNet worker actually means.

    nnUNet reports a worker process that died as "Background workers died" or
    "Segmentation export worker died", and the real error went to that
    worker's stderr, which nothing forwards. The usual cause is the kernel
    killing it for RAM, so that is said instead of the line that says nothing.
    """
    text = str(exc)
    if "worker" in text.lower() and "died" in text.lower():
        return (f"{type(exc).__name__}: an nnUNet worker process died, most often for "
                "lack of RAM; its own error went to stderr only")
    return f"{type(exc).__name__}: {_last_line(exc)}"


__all__ = [
    "CHECKPOINT_NAME",
    "ModelNotFoundError",
    "find_model_folder",
    "predict_folder",
    "resolve_device",
    "why_no_model",
]
