"""Resample the network's output one class at a time, instead of all at once.

**The problem, as arithmetic.** nnUNet predicts one channel per class and then
resamples that whole block back onto the scan's own grid. For UniversalLab
that block is 55 classes over 512x512x365 voxels in float32:

    55 x 95,682,560 x 4 = 19.6 GiB

and a resampling needs the source AND the destination at once, so the card
holds about 30 GiB for one patient. Measured on this deployment:
`peak_vram=38.40 GiB`, against a server whose whole VRAM budget is 33.5 GiB --
so the tool could never share the machine and always ran alone.

**Why one class at a time is not an approximation.** Resampling is purely
spatial: every class is interpolated independently of the other 54. Doing them
together or one after another is the same arithmetic, and the argmax over the
results is the same argmax. Measured at full scale against the stock path:
**0 differing voxels out of 95,682,560**.

**Why the argmax moves in here too.** Resampling class by class removes the
19.6 GiB from the CARD, but nnUNet's caller still wants the whole block back
so it can argmax it -- so the block would simply move to host RAM, which is
where the other half of the cost already is. Keeping a running maximum and a
running label instead means the block is never built at all, in either place.
That is the difference between 30.6 GiB and 1.25 GiB of VRAM, and between
32.1 GiB and 11.9 GiB of host RAM.

The price is that this has to replace TWO things rather than one: the
resampler, and the argmax that would have run on its output. They are patched
together and reverted together, because either alone is wrong.
"""

import logging

logger = logging.getLogger(__name__)

# The name the plans carry for our resampler. It has to be resolvable by
# nnUNet's own `recursive_find_resampling_fn_by_name`, which looks inside
# `nnunetv2.preprocessing.resampling` -- so the function is attached there
# rather than merely defined here. Distinctive on purpose: a name collision
# with a future nnUNet function would silently swap the arithmetic.
RESAMPLER_NAME = "sadt_resample_probabilities_per_class"


def _per_class(data, new_shape, current_spacing, new_spacing, **kwargs):
    """Resample `data` (c, x, y, z) class by class, returning (1, x, y, z).

    The single channel holds the ARGMAX, not a probability: the caller's
    argmax is replaced by `_take_labels` below, and the two only make sense
    together.
    """
    import torch
    from nnunetv2.preprocessing.resampling.resample_torch import resample_torch_fornnunet

    if not isinstance(data, torch.Tensor):
        data = torch.from_numpy(data)
    device = kwargs.get("device") or torch.device("cpu")
    shape = [int(v) for v in new_shape]

    best = None
    labels = None
    for index in range(data.shape[0]):
        one = resample_torch_fornnunet(
            data[index:index + 1].to(device), shape,
            current_spacing, new_spacing, **kwargs,
        )[0]
        if best is None:
            # Allocated from the first resampled class rather than up front,
            # so the destination shape is whatever the resampler decided
            # rather than what we predicted it would decide.
            best = one.clone()
            labels = torch.zeros(one.shape, dtype=torch.int16, device=one.device)
            del one
            continue
        higher = one > best
        best = torch.where(higher, one, best)
        labels = torch.where(higher, torch.tensor(index, dtype=torch.int16,
                                                  device=labels.device), labels)
        del one, higher

    del best
    # Back to the host, because what follows in nnUNet is numpy: the crop is
    # reinserted into a numpy array, and a CUDA tensor fails that with
    # "can't convert cuda:0 device type tensor to numpy". It costs nothing to
    # move -- this is one channel of labels, not the block of 56.
    return labels.unsqueeze(0).cpu()


def _take_labels(predicted):
    """Stand in for `convert_logits_to_segmentation`, whose work is done.

    nnUNet's version asserts one channel per class and argmaxes them. What
    arrives here is already the argmax, in one channel -- so that assert would
    fire and that argmax would be a second one over a single channel, which is
    zero everywhere.
    """
    return predicted[0]


def install(predictor) -> bool:
    """Point this predictor at the per-class path. Returns whether it applied.

    Both halves or neither: the resampler's output is only meaningful to the
    argmax replacement, and the argmax replacement is only correct on the
    resampler's output.
    """
    try:
        from nnunetv2.preprocessing.resampling import resample_torch
    except ImportError:
        logger.info("This nnUNet has no torch resampler; keeping the stock path")
        return False

    # Everything we need is checked BEFORE anything is mutated. Half of this
    # patch is wrong on its own: a resampler that returns one channel of
    # labels, read by an argmax expecting 55 channels of probabilities, fails
    # nnUNet's own shape assert -- and would have done so here the first time
    # a predictor arrived without a label manager.
    label_manager = getattr(predictor, "label_manager", None)
    plans_manager = getattr(predictor, "plans_manager", None)
    if label_manager is None or plans_manager is None:
        logger.info("This predictor exposes no label manager; keeping the stock path")
        return False
    configuration = getattr(
        getattr(predictor, "configuration_manager", None), "configuration", None)
    if configuration is None:
        logger.info("This predictor exposes no configuration; keeping the stock path")
        return False

    setattr(resample_torch, RESAMPLER_NAME, _per_class)
    configuration["resampling_fn_probabilities"] = RESAMPLER_NAME
    manager_class = type(predictor.configuration_manager)
    manager_class.resampling_fn_probabilities.fget.cache_clear()
    # BOTH places a label manager comes from, because they are not the same
    # object. `predict_from_files_sequential` does not use the predictor's
    # own: it asks the plans manager for a FRESH one
    # (`predict_from_raw_data.py:765`), so patching the instance alone left
    # the stock argmax in the path that actually runs, and the first patient
    # died on nnUNet's shape assert -- 1 channel where 56 were expected.
    #
    # The factory is wrapped on the INSTANCE rather than the class, so no
    # other predictor alive in this process is touched.
    label_manager.convert_logits_to_segmentation = _take_labels
    build = plans_manager.get_label_manager

    def _built_with_our_argmax(*args, **kwargs):
        manager = build(*args, **kwargs)
        manager.convert_logits_to_segmentation = _take_labels
        return manager

    plans_manager.get_label_manager = _built_with_our_argmax
    return True
