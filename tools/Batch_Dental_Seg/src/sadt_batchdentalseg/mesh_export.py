"""Surface exports for a segmentation: STL, OBJ, VTK, and one merged VTK.

**Why this exists here and not in the client.** The local module did its mesh
exports inside Slicer, through `slicer.modules.segmentations` and a scene node
per label. None of that exists on a server, which is why the exports were the
one part of BatchDentalSeg deliberately left unported. Doing them here instead
means every consumer gets them -- a Slicer panel, a `curl`, another tool
through the supervisor -- rather than only the callers that happen to be
running inside Slicer.

**The geometry.** SimpleITK mask -> `vtkDiscreteMarchingCubes` -> clean ->
windowed-sinc smoothing -> decimation, which is the shape AMASSS's
`vtk_export` already uses on the same kind of data. Marching cubes runs on the
ORIGINAL scan grid, so a CBCT at 0.33 mm yields a triangle per voxel face:
detail no mask actually carries, a mask being accurate to about half a voxel
to begin with, and enough of it to make a cohort's worth of meshes unusable
both to ship and to open. `surface_decimation` is what bounds that, and it is
an argument rather than a constant because it is lossy.

**One file per label, except the merged VTK.** A tooth is a separate object to
anyone who opens these, and STL and OBJ cannot carry a label array to separate
them again afterwards. The merged VTK is the exception: it is one file holding
every surface with a `Label` cell array, which is what the local module wrote
and what downstream tooling reads.
"""

import logging
import os
import tempfile

logger = logging.getLogger(__name__)

# The NIfTI label volume this tool has always written. Named here beside the
# mesh formats because it is one entry of the same argument: a caller picks
# what comes out, and the volume is one of the things that can come out.
NIFTI = "NIFTI"
MERGED_VTK = "VTK (merged)"

# Order is what a panel renders, so it runs cheapest-first and puts the two
# formats a clinician opens in a mesh viewer next to each other.
FORMATS = (NIFTI, "STL", "OBJ", "VTK", MERGED_VTK)

# What each mesh format is written by. `vtkOBJWriter` has no binary mode --
# the format has none -- so an OBJ of a cohort is the largest thing here.
_PER_LABEL_WRITERS = {
    "STL": ("vtkSTLWriter", ".stl", True),
    "OBJ": ("vtkOBJWriter", ".obj", False),
    "VTK": ("vtkPolyDataWriter", ".vtk", True),
}


def wanted_meshes(formats) -> list:
    """Those of `formats` that are surfaces, in `FORMATS` order."""
    chosen = set(formats or ())
    return [name for name in FORMATS if name != NIFTI and name in chosen]


def _surface(mask, reference, smoothing: int, decimation: int):
    """A cleaned, smoothed, decimated surface for one binary mask."""
    import numpy as np
    import SimpleITK as sitk
    import vtk

    binary = sitk.GetImageFromArray(mask.astype(np.uint8))
    binary.CopyInformation(reference)
    # Through a file because `vtkNrrdReader` is what carries the image's
    # spacing, origin and direction into VTK's own coordinates. Unique per
    # call: surfaces are built one label at a time today and a fixed name
    # would corrupt them silently the first time they are not.
    handle, temp_nrrd = tempfile.mkstemp(suffix=".nrrd")
    os.close(handle)
    try:
        sitk.WriteImage(binary, temp_nrrd)
        reader = vtk.vtkNrrdReader()
        reader.SetFileName(temp_nrrd)
        reader.Update()

        marching_cubes = vtk.vtkDiscreteMarchingCubes()
        marching_cubes.SetInputConnection(reader.GetOutputPort())
        marching_cubes.GenerateValues(1, 1, 1)

        clean = vtk.vtkCleanPolyData()
        clean.SetInputConnection(marching_cubes.GetOutputPort())

        # Windowed-sinc rather than the plain Laplacian smoother: it does not
        # shrink the mesh towards its centre, and a tooth losing a tenth of a
        # millimetre off every surface is a tooth that no longer touches its
        # neighbour. Same filter and same pass band the local module used.
        smoother = vtk.vtkWindowedSincPolyDataFilter()
        smoother.SetInputConnection(clean.GetOutputPort())
        smoother.SetNumberOfIterations(max(0, int(smoothing)))
        smoother.SetPassBand(0.05)
        smoother.BoundarySmoothingOn()
        # OFF, and this one line was the whole difference in how the result
        # looked. Feature-edge smoothing PRESERVES edges sharper than the
        # feature angle -- and on a raw marching-cubes mesh every voxel step
        # is such an edge, so the filter was carefully protecting the exact
        # staircase it was there to remove. Measured on one real canine at
        # 0.33 mm: the mean angle between adjacent facets goes 12.9 -> 3.9
        # degrees, and the surface moves CLOSER to the one the module this
        # replaces produces, 0.051 -> 0.028 mm. Smoother and more faithful at
        # once, which is what says it was a mistake rather than a trade.
        #
        # It was copied from that module, where it is applied to a surface
        # Slicer has already smoothed and there is no staircase left for it
        # to protect. Harmless there, backwards here.
        smoother.SetFeatureEdgeSmoothing(False)
        smoother.NonManifoldSmoothingOn()
        smoother.NormalizeCoordinatesOn()
        smoother.Update()
        polydata = smoother.GetOutput()
    finally:
        # A mask can be a few hundred megabytes; a cohort would otherwise keep
        # one of these alive per surface until the request is cleaned up.
        try:
            os.remove(temp_nrrd)
        except OSError:
            pass

    reduction = min(max(int(decimation), 0), 99) / 100.0
    if reduction > 0 and polydata.GetNumberOfCells() > 0:
        decimator = vtk.vtkDecimatePro()
        decimator.SetInputData(polydata)
        decimator.SetTargetReduction(reduction)
        # Thin structures -- a root, a cortical plate -- are punctured without
        # this, and a punctured tooth is not obviously wrong when you look at it.
        decimator.PreserveTopologyOn()
        decimator.SetFeatureAngle(60)
        decimator.Update()
        polydata = decimator.GetOutput()

    # No normals are written, deliberately. A reader that shades a surface
    # computes them anyway, and the two reasons for baking them in did not
    # survive being measured: marching cubes already returns a consistently
    # wound mesh -- 100 percent of the facets of a sphere face outwards
    # before any filter runs -- and the apparent smoothing gain came from
    # `SplittingOn` duplicating points, which hid a third of the shared edges
    # from the metric rather than improving anything. Same triangles, same
    # vertices, and the file carries nothing a viewer cannot derive.
    return polydata


def _labelled(polydata, label: int):
    """The same surface carrying `label` on every cell, as `Label`."""
    import numpy as np
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk

    values = numpy_to_vtk(
        np.full(polydata.GetNumberOfCells(), int(label), dtype=np.int32), deep=True
    )
    values.SetName("Label")
    polydata.GetCellData().AddArray(values)
    polydata.GetCellData().SetScalars(values)
    return polydata


def _write(polydata, writer_name: str, destination: str, binary: bool) -> str:
    import vtk

    os.makedirs(os.path.dirname(destination), exist_ok=True)
    writer = getattr(vtk, writer_name)()
    writer.SetFileName(destination)
    writer.SetInputData(polydata)
    if binary:
        # These writers default to ASCII, which prints every coordinate as a
        # decimal string. On AMASSS that default was what made the responses
        # enormous -- 848.5 MB for one merged surface against 6.4 MB for every
        # segmentation in the same run. Binary is also the MORE accurate of
        # the two: it round-trips the float32 vertices exactly, where ASCII
        # keeps about six significant digits.
        writer.SetFileTypeToBinary()
    writer.Write()
    return destination


def write(labels, model, base: str, output_dir: str, suffix: str,
          formats, smoothing: int = 30, decimation: int = 0) -> list:
    """Write every surface format in `formats`. Returns what it wrote.

    `labels` is the multi-label volume as SimpleITK read it, so the surfaces
    land on the input scan's own geometry. Only the labels PRESENT in this
    scan are written, for the same reason `_split_segments` does the same: a
    UniversalLab run would otherwise write 55 empty meshes per patient, and an
    empty mesh cannot be told from a structure the model failed on.
    """
    chosen = wanted_meshes(formats)
    if not chosen:
        return []

    import numpy as np
    import SimpleITK as sitk

    array = sitk.GetArrayViewFromImage(labels)
    present = set(int(value) for value in np.unique(array) if value != 0)
    if not present:
        logger.warning("No label in this scan, so no surface was written")
        return []

    per_label = [name for name in chosen if name in _PER_LABEL_WRITERS]
    merged_wanted = MERGED_VTK in chosen

    written = []
    merged = None
    if merged_wanted:
        import vtk
        merged = vtk.vtkAppendPolyData()

    for name, value in model.labels.items():
        if value not in present:
            continue
        # Built ONCE and written in as many formats as were asked for:
        # marching cubes over a CBCT is the expensive half, and ticking STL
        # and OBJ together must not pay for it twice.
        surface = _labelled(
            _surface(array == value, labels, smoothing, decimation), value
        )
        safe_name = name.replace(" ", "-").replace("/", "-")
        for fmt in per_label:
            writer_name, extension, binary = _PER_LABEL_WRITERS[fmt]
            written.append(_write(
                surface, writer_name,
                os.path.join(output_dir, f"{base}_{suffix}_{safe_name}{extension}"),
                binary,
            ))
        if merged is not None:
            merged.AddInputData(surface)

    if merged is not None:
        merged.Update()
        if merged.GetOutput().GetNumberOfCells():
            written.append(_write(
                merged.GetOutput(), "vtkPolyDataWriter",
                os.path.join(output_dir, f"{base}_{suffix}_merged.vtk"), True,
            ))
    return written
