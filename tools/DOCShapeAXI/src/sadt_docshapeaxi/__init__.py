"""DOCShapeAXI: classify a 3D anatomical shape, and explain it on the surface.

Ported from SlicerAutomatedDentalTools' `DOCShapeAXI` module and
`DOCShapeAXI_CLI`.
"""

import json
import logging
import os
from pathlib import Path
from typing import Literal

logger = logging.getLogger("DOCShapeAXI")

__all__ = ["run"]

# The share of the bar the prediction pass takes when an explanation follows
# it. See `run`.
PREDICTION_SHARE = 0.25


def run(
    surfaces: Path,
    model: Path,
    output_dir: Path,
    explain: bool = True,
    output_suffix: str = "_pred",
    device: Literal["cuda", "cpu"] = "cuda",
) -> Path:
    """Classify each 3D surface, and paint the reason for it onto the mesh.

    Args:
        surfaces: A `.vtk` surface, or a folder of them (searched recursively).
        model: The checkpoint to use. Its name says which anatomy is being
            graded, on what scale, and with which network -- upstream asked for
            all four separately, so a caller could pair a two-class model with
            a four-class scale and be handed a severity grade the network was
            never trained to produce.
        output_dir: Where the explained surfaces and the report are written.
        explain: Also compute the GradCAM attribution and write it onto each
            surface as a point array. Off, only the predictions are produced,
            which is several times faster.
        output_suffix: Appended to each written surface's stem.
        device: `cuda` or `cpu`. Falls back to `cpu` with a warning when no
            CUDA device is visible.

    Returns:
        `{"outputs": {...}, "predictions": [...], "report": path}` -- the
        `outputs` mapping is what the server turns into the response.
    """
    from . import catalog, engine, pipeline, progress

    # The runner hands a tool `pathlib.Path` for every `path` argument, while
    # everything here works in strings. Coerced once, at the door.
    surfaces = os.fspath(surfaces)
    model = os.fspath(model)
    output_dir = os.fspath(output_dir)
    analysis = catalog.analysis_for(os.path.basename(model))

    if not os.path.exists(model):
        # Named by argument and analysis first: the file name that follows is
        # redacted on its way to an operator's log.
        raise FileNotFoundError(
            f"'model': the {analysis.anatomy} {analysis.task} checkpoint was "
            f"not found ('{os.path.basename(model)}')."
        )

    found = pipeline.discover_surfaces(surfaces)
    if not found:
        raise ValueError(
            f"No .vtk surface was found under '{os.path.basename(surfaces)}'. "
            f"DOCShapeAXI reads surfaces, not volumes."
        )
    # Both before the checkpoint is loaded, which is the slow part: a batch
    # that cannot finish is refused while refusing it is cheap.
    pipeline.check_surfaces(found)
    if explain:
        pipeline.check_unique_names(found)

    device = pipeline.resolve_device(device)
    os.makedirs(output_dir, exist_ok=True)

    # shapeaxi's dataset joins its file column onto a mount point, so the two
    # have to share a root. A single file is mounted on its own directory.
    mount_point = surfaces if os.path.isdir(surfaces) else os.path.dirname(found[0])

    logger.info(
        "DOCShapeAXI: %d surface(s), %s, %s on %s",
        len(found), analysis.anatomy, analysis.task, device,
    )

    # How the bar is shared between the two passes. Both are counted per
    # surface; what is chosen here is only where one ends and the next begins.
    # The explanation computes a GradCAM per class on top of the same forward
    # pass, which is what makes it several times slower than the prediction,
    # so the prediction is given the first quarter when both run.
    split = PREDICTION_SHARE if explain else 1.0

    # The checkpoint load is one opaque call: announced at the start of the
    # bar rather than given a share of it nothing here measures.
    progress.emit(0.0, "loading the model")
    network = pipeline.load_network(
        model, analysis.network, device, anatomy=analysis.anatomy
    )
    values = engine.predict(
        network, analysis, found, mount_point, device, span=(0.0, split)
    )

    predictions = []
    for path, value in zip(found, values):
        predictions.append({
            "surface": os.path.relpath(path, mount_point),
            # A regression checkpoint predicts a value, not a class. Upstream
            # ran argmax on both and so reported class 0 for every regression
            # subject -- there is only one column to take the argmax of.
            ("score" if analysis.is_regression else "class"):
                value if analysis.is_regression else int(value),
        })

    written = []
    if explain:
        written = engine.explain(
            network, analysis, found, mount_point, device, output_dir,
            span=(split, 1.0),
        )
        written = _rename_with_suffix(written, output_suffix)

    report_path = os.path.join(output_dir, "DOCShapeAXI_report.json")
    report = {
        "model": os.path.basename(model),
        "anatomy": analysis.anatomy,
        "task": analysis.task,
        "classes": analysis.classes,
        "network": analysis.network,
        "device": device,
        "explained": bool(explain),
        "surfaces": len(found),
        "predictions": predictions,
    }
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    outputs = {"report": report_path}
    for path in written:
        outputs[os.path.splitext(os.path.basename(path))[0]] = path

    # Every surface is accounted for or the run has already raised, so this
    # is the operator's confirmation of how much was done, not a tally of
    # partial success.
    logger.info(
        "%d of %d surfaces classified, %d explained", len(predictions),
        len(found), len(written),
    )
    return {"outputs": outputs, "predictions": predictions, "report": report_path}


def _rename_with_suffix(paths: list, suffix: str) -> list:
    """Append `suffix` to each written surface's stem.

    Done after the fact rather than inside the writer because shapeaxi names
    the file it writes; an empty suffix leaves the name alone.
    """
    if not suffix:
        return paths

    renamed = []
    for path in paths:
        stem, extension = os.path.splitext(os.path.basename(path))
        destination = os.path.join(os.path.dirname(path), f"{stem}{suffix}{extension}")
        os.replace(path, destination)
        renamed.append(destination)
    return renamed
