"""The two torch helpers both ALI engines need, and nothing else.

Duplicated from the CBCT engine's `brain.py` rather than shared, and the split
between the two is the point: `brain.py` also holds the deep-RL agent and the
monai DenseNet it walks the volume with. This engine uses monai too -- a 2D
UNet, in `engine.py` -- but a different network entirely, so importing
`brain.py` for two eleven-line guards would pull in the agent as well.

`resolve_device`'s own docstring already said "shared by both engines" before
the split -- what it was NOT was shared through a package. These are eleven
lines of guard; a copy costs nothing and an import would couple two virtualenvs
pinned to different torch versions.
"""

import logging

from .errors import ToolUnavailableError

logger = logging.getLogger(__name__)

# Names THIS tool, not the one it was copied from: an intraoral run that cannot
# import torch should be told where to fix it.
_INSTALL_HINT = (
    "ALI's intraoral engine needs torch and pytorch3d. Run `uv sync` in "
    "tools/ALI/ALI_IOS."
)

def import_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on the deployment
        raise ToolUnavailableError(f"{_INSTALL_HINT} (missing: torch)") from exc
    return torch


def resolve_device(requested: str = None) -> str:
    """The device to actually use, falling back to CPU when CUDA is absent.

    Decided once, from the caller's `device` argument, rather than by
    `torch.cuda.is_available()` deep in the code -- which the original did
    independently in five modules, so a run asked for CPU still used a card
    that happened to be present. Shared by both engines.
    """
    torch = import_torch()
    wanted = (requested or "cpu").strip().lower()
    if wanted.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("DEVICE=%s requested but CUDA is unavailable; falling back to CPU", wanted)
        return "cpu"
    if wanted == "mps" and not torch.backends.mps.is_available():
        logger.warning("device=mps requested but MPS is unavailable; falling back to CPU")
        return "cpu"
    return wanted
