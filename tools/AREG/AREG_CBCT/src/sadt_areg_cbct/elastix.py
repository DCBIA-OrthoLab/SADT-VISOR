"""The voxel-based rigid registration itself: elastix, and the conversion of
its result into a SimpleITK transform.

Ported from `AREG_CBCT/AREG_CBCT_utils/utils.py`
(`make_rigid_param_map_deterministic` / `ElastixReg` / `MaskedImage` /
`MatrixRetrieval` / `ComputeFinalMatrix`).

The correction that matters is in `retrieve_transform`. elastix reports a rigid
result as three Euler angles, a translation, AND a centre of rotation: the
transform is `y = R(x - c) + c + t`. `MatrixRetrieval` dropped `c`, building a
`Euler3DTransform` centred on the origin -- a different transform by exactly
`(I - R)c`, i.e. proportional to how far the fixed image sits from the physical
origin. Measured against a known ground truth on a phantom whose origin is at
(-140, -90, 60) mm: **8.371 mm** with the centre dropped, **0.025 mm** with it
honoured. When `c` IS the origin the two agree to the float, so this fixes the
bad cases without touching the good ones -- which is why it stayed invisible on
oriented data, ASO having recentred it.
"""

import logging
import os
import re
import tempfile

import numpy as np
import SimpleITK as sitk

from sadt_areg_common.errors import ToolUnavailableError

logger = logging.getLogger(__name__)

_INSTALL_HINT = (
    "CBCT registration needs the itk-elastix package. Install it with "
    "`pip install -r requirements.txt` and recreate the container "
    "(`docker compose up -d --force-recreate inference`)."
)


class RegistrationError(Exception):
    """One patient could not be registered. Reported, and the batch goes on.

    `cause` says whose problem it is, because a run where nothing registered
    has to come back as the caller's fault or the server's, and only the place
    that raised knows which:

    * "input" -- the caller's data or arguments (a label the mask does not
      hold, a scan that cannot be read);
    * "mask" -- the mask does not fit its scan, which is the caller's fault
      when they sent the mask and the segmentation step's when it made it;
    * "engine" -- the registration itself.
    """

    def __init__(self, message: str, cause: str = "engine"):
        super().__init__(message)
        self.cause = cause


def check_dependencies() -> None:
    """Import the whole stack once, before any scan is read.

    A dependency belongs to the server, not to one input: discovering it in the
    per-patient loop makes a 40-patient batch fail 40 times identically, each
    only after that patient's mask has been built.
    """
    _import_elastix()


def _import_elastix():
    try:
        import itk
    except ImportError as exc:  # pragma: no cover - depends on the deployment
        raise ToolUnavailableError(f"{_INSTALL_HINT} (missing: itk)") from exc
    if not hasattr(itk, "ElastixRegistrationMethod"):  # pragma: no cover
        raise ToolUnavailableError(
            f"{_INSTALL_HINT} (the itk package is installed but carries no elastix module)"
        )
    return itk


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

# The tuned rigid map, unchanged from the original except for two lines.
#
# REMOVED -- `ImagePyramidSchedule = 8,8, 4,4, 2,2`: six values for a THREE-
# dimensional image over THREE resolutions, where elastix wants nine. It does
# not error, it discards a mismatched schedule and uses its default, so the
# line never had any effect. Verified three ways: the original six-value
# schedule and no schedule at all give bit-identical results (0.025 mm from
# ground truth), a corrected nine-value 8/4/2 schedule gives a different one
# (0.113 mm). Deleting it keeps the behaviour every published result was
# produced with; "fixing" it would change a validated pipeline for no gain.
#
# ADDED -- `AutomaticTransformInitializationMethod = GeometricalCenter`,
# already elastix's default. Spelled out because it is what makes the T2
# pre-centring step removable: the original resampled every T2 onto a centred
# grid first, costing an interpolation pass per patient and writing a .tfm
# expressed in a space the caller never received.
_RIGID_PARAMETERS = {
    # Determinism. Two runs of the same request on the same data must give the
    # same transform -- this is patient data being resampled.
    "NumberOfThreads": ["1"],
    "UseDirectionCosines": ["true"],
    "ImageSampler": ["Grid"],
    "NewSamplesEveryIteration": ["false"],
    # Multi-resolution pyramid, coarse to fine.
    "NumberOfResolutions": ["3"],
    "FixedImagePyramid": ["FixedSmoothingImagePyramid"],
    "MovingImagePyramid": ["MovingSmoothingImagePyramid"],
    # Metric and interpolation.
    "Metric": ["AdvancedMattesMutualInformation"],
    "NumberOfHistogramBins": ["64"],
    "NormalizeGradient": ["true"],
    "Interpolator": ["LinearInterpolator"],
    # Optimizer.
    "Optimizer": ["ConjugateGradient"],
    "MaximumNumberOfIterations": ["1500"],
    "MaximumStepLength": ["2.0"],
    "MinimumStepLength": ["0.001"],
    "ValueTolerance": ["1e-6"],
    "GradientTolerance": ["1e-6"],
    # Initialization and scale estimation.
    "AutomaticTransformInitialization": ["true"],
    "AutomaticTransformInitializationMethod": ["GeometricalCenter"],
    "AutomaticScalesEstimation": ["true"],
    # Output. `ErodeMask` is kept for fidelity but is inert: it applies to an
    # elastix fixed MASK, and none is set -- the region is imposed by zeroing
    # the fixed image instead (see `apply_mask`).
    "ErodeMask": ["true"],
    "WriteResultImage": ["false"],
    # Ignored with the Grid sampler, harmless.
    "NumberOfSpatialSamples": ["30000"],
}


def rigid_parameter_map():
    itk = _import_elastix()
    parameters = itk.ParameterObject.New()
    parameter_map = parameters.GetDefaultParameterMap("rigid")
    for key, value in _RIGID_PARAMETERS.items():
        parameter_map[key] = value
    parameters.AddParameterMap(parameter_map)
    return parameters


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------

def apply_mask(image: sitk.Image, mask: sitk.Image, label: int = None) -> tuple:
    """Zero everything outside the mask. Returns `(masked image, note)`.

    `note` is None when nothing worth reporting happened, and a sentence
    otherwise -- it ends up in the run report rather than in a log line.

    Two things the original did silently and this does not:

    * it forced `mask.SetOrigin(image.GetOrigin())` unconditionally, papering
      over both the float drift an AMASSS mask picks up and a genuinely
      different geometry -- a mask cropped relative to its scan was applied
      several millimetres off. The geometries are compared first: identical
      sampling means the drift is copied across, anything else fails;
    * a `label` absent from the mask fell through to using the WHOLE mask, so
      asking for label 4 of a mask holding 1 and 2 registered on everything and
      reported success.
    """
    if mask.GetSize() != image.GetSize():
        raise RegistrationError(
            f"the mask is {mask.GetSize()} voxels and the scan is {image.GetSize()}: "
            f"they are not the same sampling of the same patient.",
            cause="mask",
        )
    if not np.allclose(mask.GetSpacing(), image.GetSpacing(), atol=1e-4):
        raise RegistrationError(
            f"the mask's spacing {tuple(round(v, 4) for v in mask.GetSpacing())} differs "
            f"from the scan's {tuple(round(v, 4) for v in image.GetSpacing())}.",
            cause="mask",
        )
    # Same sampling: any remaining origin/direction difference is the float
    # drift a round trip through a segmentation tool leaves behind.
    mask.CopyInformation(image)

    array = sitk.GetArrayViewFromImage(mask)
    note = None
    if label:
        present = np.unique(array)
        if label not in present:
            raise RegistrationError(
                f"the mask holds no label {label} (it has "
                f"{', '.join(str(int(value)) for value in present)}). Set "
                f"'segmentation_label' to 0 to use the whole mask.",
                cause="input",
            )
        binary = sitk.GetImageFromArray((array == label).astype(np.uint8))
        binary.CopyInformation(image)
        mask = binary
    elif len(np.unique(array)) > 2:
        note = (
            "the mask holds several labels and 'segmentation_label' is 0, so the "
            "union of all of them was used"
        )

    return sitk.Mask(image, sitk.Cast(mask, sitk.sitkUInt8)), note


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register(fixed: sitk.Image, moving: sitk.Image) -> sitk.Transform:
    """Rigidly register `moving` onto `fixed`; return the resampling transform.

    The returned transform maps a point of the FIXED image's space to the
    corresponding point of the MOVING image's space, which is the direction
    `sitk.ResampleImageFilter` consumes -- so resampling `moving` with it
    produces the moving image in the fixed image's frame.

    Neither image touches the disk. `MaskedImage` wrote the masked fixed image
    to `<temp_folder>/fixed_image_masked.nii.gz` -- one name shared by every
    patient of a run and every concurrent request, so two overlapping runs
    registered against each other's anatomy. The conversion is in memory.
    """
    itk = _import_elastix()

    # Said BEFORE the call: when elastix fails, what it was asked to do is the
    # half of the diagnosis its own message never carries.
    logger.info(
        "elastix rigid registration: fixed %s voxels at %s mm, moving %s voxels at "
        "%s mm (%s, %s resolutions, up to %s iterations)",
        "x".join(str(v) for v in fixed.GetSize()),
        "x".join(f"{v:.3g}" for v in fixed.GetSpacing()),
        "x".join(str(v) for v in moving.GetSize()),
        "x".join(f"{v:.3g}" for v in moving.GetSpacing()),
        _RIGID_PARAMETERS["Metric"][0],
        _RIGID_PARAMETERS["NumberOfResolutions"][0],
        _RIGID_PARAMETERS["MaximumNumberOfIterations"][0],
    )

    registration = itk.ElastixRegistrationMethod.New(_to_itk(fixed), _to_itk(moving))
    registration.SetParameterObject(rigid_parameter_map())
    registration.SetLogToConsole(False)
    # The log goes to a private temporary directory, removed on return. The
    # exception elastix raises says only "Internal elastix error: See elastix
    # log", so without the log a failure has no cause at all; with it the ITK
    # `Description:` line names one ("fewer than four pixels along direction
    # 0"). The log holds parameters and metric values, never image data, and the
    # directory is unique per call, so concurrent runs cannot share it.
    with tempfile.TemporaryDirectory(prefix="areg_elastix_") as log_dir:
        registration.SetLogToFile(True)
        registration.SetOutputDirectory(log_dir)
        try:
            registration.UpdateLargestPossibleRegion()
        except RuntimeError as exc:
            raise RuntimeError(
                f"elastix rigid registration failed: {_failure_description(exc, log_dir)}"
            ) from exc
    return retrieve_transform(registration.GetTransformParameterObject())


# "ITK ERROR: RecursiveGaussianImageFilter(0x59508e2d5540): " -- the class and
# the object's address say nothing an operator can act on, and the address
# changes every run.
_ITK_PREFIX = re.compile(r"^(ITK ERROR:\s*)?\w+\(0x[0-9a-fA-F]+\):\s*")


def describe(exc: BaseException) -> str:
    """The one meaningful line of an ITK exception.

    ITK's message is the C++ source file and line it was raised at, then the
    object, then the sentence. Only the sentence is kept: the source path is
    the build machine's, and the whole thing would be cut by the log line's
    length long before reaching the cause. A `Description:` line wins when
    there is one, the last non-empty line otherwise.
    """
    lines = [line.strip() for line in str(exc).splitlines() if line.strip()]
    for line in lines:
        if line.startswith("Description:"):
            return _ITK_PREFIX.sub("", line[len("Description:"):].strip())
    return _ITK_PREFIX.sub("", lines[-1]) if lines else type(exc).__name__


def _failure_description(exc: BaseException, log_dir: str) -> str:
    """The `Description:` line of elastix's log, or of the exception itself."""
    try:
        with open(os.path.join(log_dir, "elastix.log"), encoding="utf-8",
                  errors="replace") as handle:
            for line in handle:
                if line.strip().startswith("Description:"):
                    return describe(RuntimeError(line))
    except OSError:
        pass
    return describe(exc)


def retrieve_transform(parameter_object) -> sitk.Transform:
    """The SimpleITK transform equivalent to elastix's result.

    See this module's docstring for why `CenterOfRotationPoint` is read.
    """
    parameters = parameter_object.GetParameterMap(0)
    kind = parameters["Transform"][0]
    values = [float(value) for value in parameters["TransformParameters"]]
    # Absent only for a transform kind that has no centre; falling back to the
    # origin then is exact rather than a guess.
    center = (
        [float(value) for value in parameters["CenterOfRotationPoint"]]
        if "CenterOfRotationPoint" in parameters
        else [0.0, 0.0, 0.0]
    )

    if kind == "EulerTransform":
        transform = sitk.Euler3DTransform()
        transform.SetCenter(center)
        transform.SetRotation(angleX=values[0], angleY=values[1], angleZ=values[2])
        transform.SetTranslation(values[3:6])
        return transform

    if kind == "AffineTransform":
        transform = sitk.AffineTransform(3)
        transform.SetCenter(center)
        transform.SetMatrix(values[:9])
        transform.SetTranslation(values[9:12])
        return transform

    # `ComputeFinalMatrix` used to compose a LIST of transforms by multiplying
    # the rotations and ADDING the translations, which is not composition at
    # all; it was only ever handed one transform, so the error never showed.
    # There is one transform here too, and an unexpected kind says so instead of
    # producing a plausible wrong one.
    raise RegistrationError(
        f"elastix returned a '{kind}', which this tool does not know how to convert. "
        f"Expected EulerTransform (rigid) or AffineTransform."
    )


def _to_itk(image: sitk.Image):
    """A float32 itk.Image sharing this image's geometry, built in memory."""
    itk = _import_elastix()

    array = sitk.GetArrayFromImage(image).astype(np.float32)
    itk_image = itk.GetImageFromArray(np.ascontiguousarray(array))
    itk_image.SetOrigin(tuple(float(value) for value in image.GetOrigin()))
    itk_image.SetSpacing(tuple(float(value) for value in image.GetSpacing()))
    itk_image.SetDirection(
        itk.matrix_from_array(np.asarray(image.GetDirection(), dtype=np.float64).reshape(3, 3))
    )
    return itk_image
