"""Put every scan of a cohort on one voxel grid, centred.

**This is MRI2CBCT's code, and it is here on purpose.** VFACE's full pipeline
begins by calling `MRI2CBCT_RESAMPLE_CBCT_MRI`, and MRI2CBCT is not a tool in
this repository. What VFACE asks it for is not MRI2CBCT, though -- it is a
generic resample that happens to live there. Every argument VFACE sends turns
off everything that makes MRI2CBCT what it is:

    input_folder_MRI  = "None"      the MRI branches never run; the CLI tests
    input_folder_Seg  = "None"      `os.path.isdir("None")`, which is False
    resample_size     = "None"      so the volume keeps its own grid size
    spacing           = 0.3 mm      isotropic
    center            = "True"

and inside `resample_fn`, `mri=0` skips the direction-aware centring,
`rightSide=0` the left/right mirror, `iso_spacing=False` the max-spacing
branch, `fit_spacing=False` the spacing derived from a target size, and
`linear=True` the nearest-neighbour path segmentations take. Nothing of the
MRI-to-CBCT approximation, the registration, the TMJ and LR crops, the
percentile normalisation or the condyle segmentation is reachable from here.

So this is a copy, which is what CONTRIBUTING.md says two tools needing the
same code usually get: a copy costs a divergence, a coupling costs an entire
class of failure. **The divergence is real and worth naming.** Resampling
interpolates, so it changes voxel values, and a result therefore depends on
this copy rather than on MRI2CBCT. If MRI2CBCT is ever ported here, the two can
drift apart without anything failing. PROVENANCE.md records that.

One thing is NOT copied. Upstream derives each output path with
`file_path.replace(os.path.dirname(file_path), output_folder)`, which flattens
the tree: two site subdirectories holding a scan of the same name write to one
file, and the cohort silently loses a patient. The relative path is kept here,
as every other tool in this repository keeps it.
"""

import logging
import os

from .errors import ToolInputError
from .discovery import find_scans

logger = logging.getLogger(__name__)

# Upstream's value, the one VFACE sends. Isotropic, and finer than most CBCTs
# are acquired at, so this is an upsample for nearly every cohort -- which is
# the point: the later steps compare volumes voxel for voxel and need them on
# one grid.
DEFAULT_SPACING = (0.3, 0.3, 0.3)


def resample_image(image, spacing=DEFAULT_SPACING, centre: bool = True):
    """One volume on the given spacing, its own grid size kept.

    The size is deliberately NOT recomputed from the new spacing. Upstream
    passes the file's own size through and says why: forcing a CBCT to a target
    size crops the full head down to it and discards most of the volume. At a
    finer spacing and the same size the field of view shrinks around the
    centre, which is what centring is for.

    The default pixel value is the volume's own minimum rather than 0: CBCT air
    sits well below zero, and padding with 0 writes a shell of soft-tissue
    intensity around the head that every later threshold then sees.
    """
    import numpy as np
    import SimpleITK as sitk

    size = list(image.GetSize())
    origin = np.array(image.GetOrigin(), dtype=float)
    output_spacing = [float(value) for value in spacing]

    if centre:
        # The physical box the output covers, against the one the input did.
        # Half the difference moves the origin so the two share a centre.
        output_extent = np.array(size, dtype=float) * np.array(output_spacing)
        input_extent = np.array(size, dtype=float) * np.array(image.GetSpacing(), dtype=float)
        origin = origin - (output_extent - input_extent) / 2.0

    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputSpacing(output_spacing)
    resampler.SetSize(size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin([float(value) for value in origin])
    resampler.SetDefaultPixelValue(
        float(np.min(sitk.GetArrayViewFromImage(image)))
    )
    return resampler.Execute(image)


def resample_cohort(scans_dir: str, output_dir: str, spacing=DEFAULT_SPACING,
                    centre: bool = True, report: dict = None) -> str:
    """Every volume under `scans_dir`, resampled into `output_dir`.

    The tree is mirrored, not flattened. Returns `output_dir`.

    One unreadable volume does not cost the cohort: it is recorded and the run
    goes on, which is what every tool here does. A cohort where NOTHING could be
    resampled is refused, because the steps after this one would otherwise
    report "0 file" on a folder the caller knows is not empty -- and name the
    wrong step while doing it.
    """
    import SimpleITK as sitk

    found = find_scans(scans_dir)
    if not found:
        raise ToolInputError(
            f"No CBCT volume found to resample. Expected one of "
            f"{', '.join(('.nii', '.nii.gz', '.nrrd', '.nrrd.gz', '.gipl', '.gipl.gz'))}."
        )

    written = []
    for index, path in enumerate(found, start=1):
        destination = os.path.join(output_dir, os.path.relpath(path, scans_dir))
        try:
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            sitk.WriteImage(resample_image(sitk.ReadImage(path), spacing, centre), destination)
            written.append(destination)
        except Exception as exc:  # noqa: BLE001 - one scan must not cost the cohort
            logger.exception("VFACE could not resample one volume")
            if report is not None:
                report.setdefault("not_resampled", {})[
                    os.path.relpath(path, scans_dir)
                ] = f"{type(exc).__name__}: {exc}"

    if not written:
        # Counted on what was WRITTEN, not on what the walk found: a guard that
        # counted the files it walked past would pass on a cohort where every
        # single one failed to read.
        raise ToolInputError(
            f"None of the {len(found)} volume(s) found could be resampled. The "
            "per-scan errors are in the run report."
        )
    logger.info("VFACE: %d of %d volume(s) resampled to %s mm",
                len(written), len(found), "x".join(str(value) for value in spacing))
    return output_dir
