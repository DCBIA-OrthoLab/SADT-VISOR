"""The two passes: predict a class per surface, then explain it on the mesh."""

import logging
import os

from . import progress
from .pipeline import GRADCAM_IMAGE_SIZE, scale_attribution

logger = logging.getLogger("DOCShapeAXI")


def _dataset(model, surfaces, mount_point, device):
    """A shapeaxi dataset over the surfaces, built the way upstream builds it.

    shapeaxi takes its file list through a DataFrame, so one is made in memory.
    Upstream wrote a CSV into the OUTPUT folder and read it back -- and opened
    it with mode 'a', so a second run appended to the first run's list.
    """
    import pandas as pd
    from shapeaxi.saxi_dataset import SaxiDataset
    from shapeaxi.saxi_transforms import EvalTransform

    frame = pd.DataFrame({model.hparams.surf_column: [
        os.path.relpath(path, mount_point) for path in surfaces
    ]})
    scale_factor = getattr(model.hparams, "scale_factor", None)
    return SaxiDataset(
        frame,
        transform=EvalTransform(scale_factor),
        CN=True,
        surf_column=model.hparams.surf_column,
        mount_point=mount_point,
        class_column=None,
        scalar_column=None,
    )


def predict(model, analysis, surfaces, mount_point, device, span=(0.0, 1.0)) -> list:
    """One prediction per surface, in the order given.

    `span` is the slice of the run's bar this pass occupies; the surfaces are
    counted across it as they are taken off the loader.
    """
    import torch
    from torch.utils.data import DataLoader

    dataset = _dataset(model, surfaces, mount_point, device)
    loader = DataLoader(dataset, batch_size=1, pin_memory=False)
    softmax = torch.nn.Softmax(dim=1)

    predictions = []
    total = len(surfaces)
    with torch.no_grad():
        # `index` is kept current so a failure while the loader reads the
        # next surface is still reported at the right position.
        index = 1
        try:
            for index, (vertices, faces, normals) in enumerate(loader, start=1):
                progress.report(index, total, "classifying surface", *span)
                vertices = vertices.to(device)
                faces = faces.to(device)
                normals = normals.to(device)

                mesh = model.create_mesh(vertices, faces, normals)
                points = model.sample_points_from_meshes(mesh, model.hparams.sample_levels[0])
                views, _ = model.render(mesh)

                output = model(points, views)
                if not analysis.is_regression:
                    # No argmax for a regression checkpoint: its single output
                    # IS the value, and taking an argmax of one column returns
                    # 0 for every subject.
                    output = torch.argmax(softmax(output).detach(), dim=1, keepdim=True)
                predictions.append(float(output.reshape(-1)[0].cpu()))
                index += 1
        except Exception as exc:
            _log_failure(index, total, "classification", exc)
            raise
    return predictions


def _log_failure(index, total, step, exc) -> None:
    """Say which surface a pass died on, by position, before it propagates.

    The run stops at the first failure -- predictions are only meaningful for
    the whole batch -- so this is an ERROR rather than a per-item warning. The
    exception itself is re-raised untouched, so its class and frame still
    reach the server's diagnosis.
    """
    logger.error(
        "surface %d of %d: %s failed (%s: %s)",
        min(index, total), total, step, type(exc).__name__, exc,
    )


def explain(model, analysis, surfaces, mount_point, device, output_dir,
            span=(0.0, 1.0)) -> list:
    """One surface per input, carrying a GradCAM array per class.

    Written ONCE per surface, after every class has been added. Upstream wrote
    the file inside the per-class loop, to the same path each time, so a
    four-class run rewrote the same file four times.

    `span` is the slice of the run's bar this pass occupies, counted per
    surface like `predict`.
    """
    from captum.attr import LayerGradCam
    from torch.utils.data import DataLoader

    dataset = _dataset(model, surfaces, mount_point, device)
    # In-process, like `predict`. Upstream used four worker processes, but
    # loading a surface is cheap next to a GradCAM per class, and a worker's
    # failure reaches the server as a re-raised traceback string cut short,
    # with no log records -- worker processes do not forward them.
    loader = DataLoader(dataset, batch_size=1, num_workers=0, pin_memory=False)

    blocks = getattr(model.convnet.module, "_blocks")
    cam = LayerGradCam(model, blocks[-1], device_ids=[0])

    written = []
    total = len(surfaces)
    index = 0
    try:
        for index, batch in enumerate(loader):
            progress.report(index + 1, total, "explaining surface", *span)
            written.append(_explain_one(
                model, analysis, dataset, index, batch, cam, device, output_dir,
            ))
            index += 1
    except Exception as exc:
        _log_failure(index + 1, total, "explanation", exc)
        raise
    return written


def _explain_one(model, analysis, dataset, index, batch, cam, device, output_dir) -> str:
    """The GradCAM arrays for one surface, written once; returns the path."""
    from shapeaxi import post_process, utils
    from shapeaxi.saxi_gradcam import gradcam_process

    vertices, faces, normals = batch
    vertices = vertices.to(device)
    faces = faces.to(device)
    normals = normals.to(device)

    mesh = model.create_mesh(vertices, faces, normals)
    points = model.sample_points_from_meshes(mesh, model.hparams.sample_levels[0])
    views, per_face = model.render(mesh)

    surface = dataset.getSurf(index)
    source = dataset.getSurfPath(index)

    for class_index in range(analysis.classes):
        attribution = cam.attribute(
            inputs=(points, views), target=class_index, attr_dim_summation=False
        )
        attribution = attribution.sum(dim=1).cpu().detach()
        scaled = scale_attribution(attribution.numpy(), GRADCAM_IMAGE_SIZE)
        projected = gradcam_process(
            _Namespace(device=device, target_class=class_index),
            scaled, faces, per_face, vertices, device=device,
        )
        surface.GetPointData().AddArray(projected)
        post_process.MedianFilter(surface, projected)

    destination = os.path.join(output_dir, os.path.basename(source))
    utils.WriteSurf(surface, destination)
    return destination


class _Namespace:
    """`gradcam_process` reads its arguments off an object; upstream passed the
    whole argparse namespace.

    Two fields are read: `device`, and `target_class`, which names the point
    array it writes (`grad_cam_target_class_2`). Passing the class index is
    what keeps one array per class on the mesh instead of every class
    overwriting a single `grad_cam_max`.
    """

    def __init__(self, device, target_class=None):
        self.device = device
        self.target_class = target_class
