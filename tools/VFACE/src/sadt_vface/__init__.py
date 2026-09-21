"""VFACE -- facial asymmetry, measured and classified from a CBCT.

The pipeline is in `dispatch.py`; only `run` is public.

**A patient is compared against themselves.** There is no second scan in an
asymmetry assessment: the baseline is oriented, mirrored across the
mid-sagittal plane, registered back onto itself region by region, and what is
measured is how far each landmark has moved from its own reflection. A
longitudinal study is the same chain with a real follow-up in the mirror's
place, which is why one tool does both.

Six other tools do the work VFACE does not: `ASO` orients, `AMASSS` segments,
`AutoMatrix` mirrors and moves, `AREG_CBCT` registers, `ALI_CBCT` places the
landmarks and `Batch_Dental_Seg` makes the surfaces a heat map is drawn on.
VFACE measures, and reads the measurements as a classification.
"""

from pathlib import Path
from typing import Literal


def run(
    t1: Path,
    output_dir: Path,
    mode: Literal[
        "Full pipeline", "File already Oriented", "File already Registered"
    ] = "Full pipeline",
    study: Literal["Asymmetry assessment", "Longitudinal study"] = "Asymmetry assessment",
    outputs: Literal[
        "Measurements and classification", "Measurements and heat maps", "Heat maps"
    ] = "Measurements and classification",
    regions: list[
        Literal["Cranial base", "Mandible", "Maxilla"]
    ] = ["Cranial base", "Mandible", "Maxilla"],
    t2: Path = "",
    measurements: Path = "",
    feature_template: Path = "",
    cranial_base_reference: Path = "",
    maxilla_reference: Path = "",
    mirror_reference: Path = "",
    segmentation_model: Path = "",
    landmark_model: Path = "",
    classifier_model: Path = "",
    surface_model: Path = "",
    *,
    sup=None,
) -> Path:
    """Classify a patient's facial asymmetry, or measure the change between two scans.

    Args:
        t1: The CBCT volumes, one folder, searched recursively. Raw scans for
            the full pipeline; already-oriented ones for the modes that skip
            it, which read the frame each was oriented into off its name.
        output_dir: Where the measurement tables, the classification and
            `VFACE_report.json` are written. Nothing is written outside it.
        mode: How far down the pipeline the scans already are. Full pipeline
            resamples, orients, segments, registers and measures. File already
            Oriented starts at the segmentation; File already Registered
            measures only.
        study: An asymmetry assessment has no second scan -- the patient is
            compared against their own mirror image. A longitudinal study
            compares the baseline with the follow-up in `t2`.
        outputs: Measurements and a classification, heat maps of where the two
            surfaces differ, or both. Measurements need no segmentation of the
            soft tissue and heat maps need no landmarks, so a deployment
            missing one tool can still answer the other half.
        regions: Which regions to measure. Each is worked in the frame it is
            defined in: the mandible against the cranial base, the maxilla
            against the occlusal plane.
        t2: The follow-up CBCTs, for a longitudinal study. Ignored by an
            asymmetry assessment, which makes its own second scan.
        measurements: A folder holding one measurement list per region -- an
            Excel naming, per row, a measurement and the landmarks it is taken
            on. Matched to its region by the file's name.
        feature_template: The workbook whose columns name the features the
            classifier was trained on. It comes from the same training run as
            `classifier_model`, and without both the run stops after the
            measurements.
        cranial_base_reference: The already-oriented case defining the
            Frankfort horizontal and mid-sagittal frame.
        maxilla_reference: The already-oriented case defining the occlusal and
            mid-sagittal frame.
        mirror_reference: The transform that reflects a scan across the
            mid-sagittal plane. What an asymmetry assessment compares against.
        segmentation_model: The bundle that segments the bone each region is
            registered on.
        landmark_model: The bundle that places the landmarks every measurement
            is computed from.
        classifier_model: The bundle holding the three asymmetry models.
        surface_model: The bundle that segments the surfaces a heat map is
            drawn on.

    Returns:
        The output directory, holding the measurement tables, the
        classification and the run report.
    """
    # Imported HERE, not at module level: dispatch pulls pandas, SimpleITK and
    # vtk, and CI imports every tool on every PR to generate its schema.
    from .dispatch import main

    output_dir = Path(output_dir)
    main(
        t1=t1, output_dir=output_dir, mode=mode, study=study, outputs=outputs,
        regions=regions, t2=t2, measurements=measurements,
        feature_template=feature_template,
        cranial_base_reference=cranial_base_reference,
        maxilla_reference=maxilla_reference, mirror_reference=mirror_reference,
        segmentation_model=segmentation_model, landmark_model=landmark_model,
        classifier_model=classifier_model, surface_model=surface_model, sup=sup,
    )
    return output_dir
