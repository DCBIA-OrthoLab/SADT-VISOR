"""nnUNet v2 inference for BatchDentalSeg, isolated from the pipeline.

The upstream widget drove nnUNet through `SlicerNNUNetLib.Parameter` and a
`QProcess`, which is why most of that file is process management: killing a
crashed inference tree, reclaiming stray workers, a RAM watchdog. None of it
applies here -- the Python API returns when it is done, and the server bounds
concurrency.

The resampling runs on the GPU too (see `_enable_gpu_resampling`), which is
where the run time actually goes -- the network is a fraction of it. AMASSS
measured that first; the four bundles here were measured separately, because
their models are not AMASSS's and a numerical cost measured on one network says
nothing about another. See README.md, "GPU resampling".

AMASSS carries a near-identical module and they stay separate copies. In the
server that was because `registry.py` imported every tool at startup, so one
tool's missing dependency would take both out of the registry; here it is
because the two packages share no environment at all and may pin different
nnUNet versions. See CONTRIBUTING.md.

torch and nnunetv2 are imported lazily even though the lockfile guarantees
them: `scripts/describe.py` imports this package on every CI run to publish the
schema, and that must not pay for a CUDA stack.
"""

import inspect
import logging
import os

from . import low_memory
from .errors import ModelNotFoundError

logger = logging.getLogger(__name__)

CHECKPOINT_NAME = "checkpoint_final.pth"

# A bundle is the directory holding these three, whatever it is called and
# however deeply the archive nested it. Discovered rather than assumed: the
# four bundles do not share one layout -- three are three flat files plus a
# fold_0/, and DentalSegmentator arrives as a zip with its own Dataset<n>/
# tree inside.
_REQUIRED_FILES = ("dataset.json", "plans.json")
_FOLD_DIR = "fold_0"


def resolve_device(requested: str = None) -> str:
    """The device to actually use, falling back to CPU when CUDA is absent."""
    import torch

    wanted = (requested or "cpu").strip().lower()
    if wanted.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("device=%s requested but CUDA is unavailable; falling back to CPU", wanted)
        return "cpu"
    return wanted


def find_model_folder(model_root: str):
    """The nnUNet bundle inside `model_root`, or None.

    Walks for the directory holding dataset.json, plans.json and
    fold_0/checkpoint_final.pth, all three confirmed before a candidate is
    accepted. A half-downloaded bundle therefore reports "this model is not
    installed" rather than failing inside nnUNet's loader.
    """
    if not os.path.isdir(model_root):
        return None

    for directory, _dirs, _files in os.walk(model_root):
        if not all(os.path.isfile(os.path.join(directory, name)) for name in _REQUIRED_FILES):
            continue
        if os.path.isfile(os.path.join(directory, _FOLD_DIR, CHECKPOINT_NAME)):
            return directory
    return None


def _build_predictor(device: str, tile_step_size: float):
    """Instantiate an nnUNetPredictor, tolerating nnUNet's renamed kwargs.

    nnUNet 2.x renamed `perform_everything_on_gpu` to
    `perform_everything_on_device` mid-series; passing whichever the installed
    version declares keeps this working across the range.
    """
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    options = {
        "tile_step_size": float(tile_step_size),
        "use_gaussian": True,
        # No test-time mirroring, matching upstream's inference settings.
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

    return nnUNetPredictor(**{key: value for key, value in options.items() if key in accepted})


# The resampler nnUNet's own plans name by default, and the only one we are
# willing to substitute. All four bundles ask for it; one asking for anything
# else (no_resampling, a custom function) was configured that way deliberately
# and its geometry is not ours to reinterpret.
_STOCK_RESAMPLER = "resample_data_or_seg_to_shape"
# The two ends that cost: the input volume on the way to the model's grid, and
# the logits on the way back. nnUNet has a third, `resampling_fn_seg` for the
# nonzero mask it crops by, and that one is deliberately NOT swapped -- it is
# 4.5 s of a 58 s run, it is the only one of the three that is `is_seg=True`,
# and matching AMASSS exactly is what keeps the numerical change to the one
# thing that was measured.
_RESAMPLING_KEYS = ("resampling_fn_data", "resampling_fn_probabilities")
# What we substitute, written as the string nnUNet will resolve it back by --
# NOT read off the imported function's `__name__`. The two are the same thing
# right up until something has wrapped that module attribute, and then the
# plans carry a name that resolves to nothing.
_TORCH_RESAMPLER = "resample_torch_fornnunet"


def _enable_gpu_resampling(predictor, device: str) -> bool:
    """Point this predictor's resamplers at the GPU. Returns whether it applied.

    Resampling, not inference, is where a run goes: nnUNet's defaults are scipy
    splines pinned to one core, and on a 512x512x365 CBCT at 0.33 mm they
    outweigh the network by 4x (Naso), 8-9x (DentalSegmentator, Pediatric) and
    23x (UniversalLab, whose 55 output classes make one resampling 190 s).
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

    import torch

    # The module, and then the name in it: nnUNet resolves the name out of this
    # very module, so checking it here is checking the thing that will happen.
    try:
        from nnunetv2.preprocessing.resampling import resample_torch
    except ImportError:
        logger.info("This nnUNet has no torch resampler; keeping the scipy one")
        return False

    if not hasattr(resample_torch, _TORCH_RESAMPLER):
        logger.info("This nnUNet has no %s; keeping the scipy resampler", _TORCH_RESAMPLER)
        return False

    configuration_manager = predictor.configuration_manager
    configuration = configuration_manager.configuration

    if any(configuration.get(key) != _STOCK_RESAMPLER for key in _RESAMPLING_KEYS):
        logger.info("Model plans request a non-default resampler; leaving it alone")
        return False

    for key in _RESAMPLING_KEYS:
        configuration[key] = _TORCH_RESAMPLER
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

    # And then the probabilities end is narrowed further: same arithmetic,
    # one class at a time, so the card never holds the whole block. See
    # low_memory.py for why that is exact rather than approximate.
    low_memory.install(predictor)
    return True


def predict_folder(model_folder: str, input_dir: str, output_dir: str, device: str,
                   tile_step_size: float = 0.5, gpu_resampling: bool = True) -> None:
    """Segment every `*_0000.nii.gz` in `input_dir`, writing masks to `output_dir`.

    A whole folder per call, so the checkpoint is loaded once for the batch
    rather than once per scan.
    """
    os.makedirs(output_dir, exist_ok=True)

    predictor = _build_predictor(device, tile_step_size)
    # An explicit path, never nnUNet's `nnUNet_results` environment variable:
    # os.environ is process-global and the variable would outlive the call.
    predictor.initialize_from_trained_model_folder(
        model_folder,
        use_folds=(0,),
        checkpoint_name=CHECKPOINT_NAME,
    )

    on_gpu = bool(gpu_resampling) and _enable_gpu_resampling(predictor, device)

    if on_gpu:
        # `predict_from_files` fans preprocessing and export out to SPAWNED
        # processes, each of which would need its own CUDA context to run a
        # GPU resampler. The GPU path therefore runs everything in this
        # process, trading away the CPU/GPU overlap on multi-scan batches --
        # a smaller loss than the resampling win.
        predictor.predict_from_files_sequential(
            input_dir,
            output_dir,
            save_probabilities=False,
            overwrite=True,
        )
    else:
        predictor.predict_from_files(
            input_dir,
            output_dir,
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=2,
            num_processes_segmentation_export=2,
        )


__all__ = [
    "CHECKPOINT_NAME",
    "ModelNotFoundError",
    "find_model_folder",
    "predict_folder",
    "resolve_device",
]
