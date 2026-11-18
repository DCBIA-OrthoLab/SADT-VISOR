"""Turning elastix's answer into a SimpleITK transform, and the map it is
asked with.

The correction that matters is in `retrieve_transform`: elastix reports a rigid
result as three Euler angles, a translation AND a centre of rotation, so the
transform is `y = R(x - c) + c + t`. `MatrixRetrieval` dropped `c`, building a
transform centred on the physical origin -- a different transform by exactly
`(I - R)c`, i.e. proportional to how far the fixed image sits from the origin.

`test_run.py::TestElastix` measures that against a known ground truth. What is
here is the conversion itself, driven with elastix's own output shape, plus the
parameter map -- every line of which is either a determinism guarantee or a
deliberate difference from upstream.
"""

import numpy as np
import pytest
import SimpleITK as sitk

from conftest import displaced, full_mask, phantom
from sadt_areg_cbct import elastix


class StubParameterObject:
    """What `GetTransformParameterObject()` hands back, as far as this reads it."""

    def __init__(self, mapping):
        self._mapping = mapping

    def GetParameterMap(self, index):
        assert index == 0
        return self._mapping


def euler(angles=("0.1", "-0.05", "0.02"), translation=("1.0", "2.0", "3.0"),
          center=("-140.0", "-90.0", "60.0")):
    mapping = {
        "Transform": ["EulerTransform"],
        "TransformParameters": list(angles) + list(translation),
    }
    if center is not None:
        mapping["CenterOfRotationPoint"] = list(center)
    return StubParameterObject(mapping)


# ---------------------------------------------------------------------------
# retrieve_transform
# ---------------------------------------------------------------------------

def test_the_centre_of_rotation_reaches_the_transform():
    transform = elastix.retrieve_transform(euler())
    assert transform.GetCenter() == pytest.approx((-140.0, -90.0, 60.0))
    assert transform.GetParameters()[:3] == pytest.approx((0.1, -0.05, 0.02))
    assert transform.GetParameters()[3:] == pytest.approx((1.0, 2.0, 3.0))


def test_dropping_the_centre_would_move_every_point():
    """Not a rounding difference: `(I - R)c` grows with the distance from the
    physical origin, which is why the defect stayed invisible on data ASO had
    already recentred."""
    honoured = elastix.retrieve_transform(euler())

    dropped = sitk.Euler3DTransform()
    dropped.SetRotation(*honoured.GetParameters()[:3])
    dropped.SetTranslation(honoured.GetParameters()[3:6])

    probe = (-140.0, -90.0, 60.0)
    difference = np.linalg.norm(
        np.array(honoured.TransformPoint(probe)) - np.array(dropped.TransformPoint(probe))
    )
    assert difference > 1.0


def test_a_transform_with_no_centre_falls_back_to_the_origin():
    """Exact rather than a guess: a transform kind that has no centre rotates
    about the origin by definition."""
    transform = elastix.retrieve_transform(euler(center=None))
    assert transform.GetCenter() == pytest.approx((0.0, 0.0, 0.0))


def test_an_affine_result_is_converted_too():
    """elastix can be asked for one, and a plausible wrong conversion is worse
    than a refusal."""
    matrix = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    parameters = StubParameterObject({
        "Transform": ["AffineTransform"],
        "TransformParameters": [str(v) for v in matrix + [4.0, 5.0, 6.0]],
        "CenterOfRotationPoint": ["1.0", "2.0", "3.0"],
    })
    transform = elastix.retrieve_transform(parameters)
    assert isinstance(transform, sitk.AffineTransform)
    assert transform.GetCenter() == pytest.approx((1.0, 2.0, 3.0))
    assert transform.GetTranslation() == pytest.approx((4.0, 5.0, 6.0))


def test_an_unexpected_transform_kind_is_named_rather_than_approximated():
    """`ComputeFinalMatrix` used to compose a LIST of transforms by multiplying
    the rotations and ADDING the translations, which is not composition at all.
    It was only ever handed one, so the error never showed."""
    parameters = StubParameterObject({
        "Transform": ["BSplineTransform"],
        "TransformParameters": ["0.0"] * 6,
    })
    with pytest.raises(elastix.RegistrationError) as raised:
        elastix.retrieve_transform(parameters)
    assert "BSplineTransform" in str(raised.value)
    assert "EulerTransform" in str(raised.value)


# ---------------------------------------------------------------------------
# The parameter map
# ---------------------------------------------------------------------------

def test_the_registration_is_asked_to_be_deterministic():
    """Two runs of the same request on the same data must give the same
    transform: this is patient data being resampled."""
    parameters = elastix.rigid_parameter_map().GetParameterMap(0)
    assert parameters["NumberOfThreads"] == ("1",)
    assert parameters["ImageSampler"] == ("Grid",)
    assert parameters["NewSamplesEveryIteration"] == ("false",)


def test_the_mismatched_pyramid_schedule_is_gone():
    """`ImagePyramidSchedule = 8,8, 4,4, 2,2` is six values for a THREE-
    dimensional image over THREE resolutions, where elastix wants nine. It does
    not error -- it discards the schedule and uses its default -- so the line
    never had any effect. Deleting it keeps the behaviour every published
    result was produced with; "fixing" it changes a validated pipeline."""
    assert "ImagePyramidSchedule" not in elastix._RIGID_PARAMETERS
    # Membership, not `.get`: the map elastix hands back is a SWIG
    # `mapstringvectorstring`, whose `get` takes no default and raises
    # `IndexError: key not found` for a missing key.
    assert "ImagePyramidSchedule" not in elastix.rigid_parameter_map().GetParameterMap(0)


def test_the_initialisation_that_makes_the_pre_centring_step_removable():
    """The original resampled every T2 onto a centred grid first, costing an
    interpolation pass per patient and writing a `.tfm` expressed in a space
    the caller never received. elastix aligns the centres itself."""
    parameters = elastix.rigid_parameter_map().GetParameterMap(0)
    assert parameters["AutomaticTransformInitialization"] == ("true",)
    assert parameters["AutomaticTransformInitializationMethod"] == ("GeometricalCenter",)


def test_elastix_is_not_asked_to_write_anything():
    parameters = elastix.rigid_parameter_map().GetParameterMap(0)
    assert parameters["WriteResultImage"] == ("false",)


def test_the_metric_and_the_optimizer_are_the_tuned_ones():
    """These decide what the registration converges to, so they are pinned
    rather than left to elastix's rigid default."""
    parameters = elastix.rigid_parameter_map().GetParameterMap(0)
    assert parameters["Metric"] == ("AdvancedMattesMutualInformation",)
    assert parameters["Optimizer"] == ("ConjugateGradient",)
    assert parameters["MaximumNumberOfIterations"] == ("1500",)
    assert parameters["NumberOfResolutions"] == ("3",)


def test_two_registrations_of_the_same_data_agree_to_the_float():
    """What the determinism settings above are for, measured rather than
    assumed."""
    fixed = phantom(size=32)
    moving, _truth = displaced(fixed)
    masked, _note = elastix.apply_mask(fixed, full_mask(fixed))

    first = elastix.register(masked, moving)
    second = elastix.register(masked, moving)
    assert first.GetParameters() == pytest.approx(second.GetParameters(), abs=0)


def test_the_dependency_check_runs_before_any_scan_is_read():
    """A dependency belongs to the server, not to one input: discovering it in
    the per-patient loop makes a 40-patient batch fail 40 times identically,
    each only after that patient's mask has been built."""
    assert elastix.check_dependencies() is None


# ---------------------------------------------------------------------------
# A failure an operator can read
# ---------------------------------------------------------------------------

def test_an_elastix_failure_names_its_cause_not_the_log_it_hid():
    """elastix's own exception says "Internal elastix error: See elastix log"
    and nothing else -- the cause is the ITK `Description:` line in a log this
    tool used to switch off. It is read back, and only that line is kept."""
    fixed = phantom(size=16)
    tiny = sitk.GetImageFromArray(np.zeros((2, 2, 2), np.float32))

    with pytest.raises(RuntimeError) as raised:
        elastix.register(fixed, tiny)
    message = str(raised.value)
    assert message.startswith("elastix rigid registration failed: ")
    assert "less than 4" in message
    assert "See elastix log" not in message
    assert ".hxx" not in message and "0x" not in message
    assert isinstance(raised.value.__cause__, RuntimeError)


def test_what_elastix_was_asked_is_logged_before_it_runs(caplog):
    fixed = phantom(size=16)
    tiny = sitk.GetImageFromArray(np.zeros((2, 2, 2), np.float32))
    caplog.set_level("INFO", logger="sadt_areg_cbct")

    with pytest.raises(RuntimeError):
        elastix.register(fixed, tiny)
    attempted = [r.getMessage() for r in caplog.records if "elastix rigid" in r.getMessage()]
    assert attempted and "16x16x16 voxels" in attempted[0] and "2x2x2 voxels" in attempted[0]


def test_an_itk_message_is_reduced_to_its_sentence():
    raised = RuntimeError(
        "/build/itkSomething.hxx:389:\n"
        "ITK ERROR: ElastixRegistrationMethod(0x63ec076aa9b0): Internal elastix error"
    )
    assert elastix.describe(raised) == "Internal elastix error"
    assert elastix.describe(RuntimeError(
        "File: x.hxx\nLine: 3\nDescription: ITK ERROR: Filter(0xab): too small\n"
    )) == "too small"
