"""AREG_IOSCBCT -- register an intraoral scan onto a CBCT of the same patient.

NOT longitudinal, unlike its two siblings: AREG_CBCT and AREG_IOS take a
baseline and a follow-up of one modality, this takes ONE timepoint imaged two
ways. Upstream's own test set says so -- `P001_T2_U.vtk` beside
`P_0001_T2.nii.gz`, both T2 -- which is why the arguments are `ios` and `cbct`
rather than `t1` and `t2`.

**It has no engine of its own, and that is the design.** No torch, no pytorch3d,
no nnUNet: the landmarks come from ALI_CBCT and ALI_IOS, the tooth labels from
Crown_Seg and the orientation from ASO, each in its own virtualenv, reached
through the supervisor. Containing both stacks would mean one environment
holding torch 2.8 for one half and 2.11 for the other -- the exact defect the
AREG and ALI splits removed. What is left here is geometry.
"""

from pathlib import Path
from typing import Literal

from .dispatch import main


def run(
    ios: Path,
    cbct: Path,
    output_dir: Path,
    automation: Literal[
        "Registration", "Semi-Automated", "Fully-Automated"
    ] = "Registration",
    ios_landmarks: Path = "",
    cbct_landmarks: Path = "",
    cbct_reference: Path = "",
    landmark_model: Path = "",
    ios_landmark_model: Path = "",
    crown_model: Path = "",
    max_dist: float = 0.0,
    output_suffix: str = "Reg",
    *,
    sup=None,
    data_root=None,
) -> Path:
    """Register an intraoral scan onto a CBCT of the same patient.

    Args:
        ios: The intraoral surfaces (.vtk/.stl), one folder, searched
            recursively. Upper and lower arches are registered separately and
            matched to their landmarks by the jaw token in the name.
        cbct: The CBCT volumes, one folder. Paired to the intraoral scans on the
            digits of the patient identifier, the two modalities being named by
            different conventions.
        output_dir: Where the registered meshes, their 4x4 matrices and
            `AREG_report.json` are written. Nothing is written outside it.
        automation: Registration takes both landmark sets and predicts nothing --
            no other tool is called and no GPU is needed. Semi-Automated labels
            the crowns and predicts both sets. Fully-Automated orients the CBCT
            first, which needs a reference.
        ios_landmarks: Registration mode. Your own intraoral landmarks.
        cbct_landmarks: Registration mode. Your own CBCT landmarks.
        cbct_reference: The frame the CBCT volumes are oriented onto before
            registering.
        landmark_model: The CBCT landmark bundle, for the modes that predict.
        ios_landmark_model: The intraoral landmark bundle.
        crown_model: The crown-labelling checkpoint, for the modes that label.
        max_dist: How far a point may be from its nearest neighbour and still
            count as an ICP correspondence, in millimetres. 0 uses 1.5, which is
            upstream's.
        output_suffix: Added to each output name, e.g. `scan_Reg.vtk`.

    Returns:
        The output directory.
    """
    output_dir = Path(output_dir)
    main(
        ios=ios, cbct=cbct, output_dir=output_dir, automation=automation,
        ios_landmarks=ios_landmarks, cbct_landmarks=cbct_landmarks,
        cbct_reference=cbct_reference, landmark_model=landmark_model,
        ios_landmark_model=ios_landmark_model, crown_model=crown_model,
        max_dist=max_dist, output_suffix=output_suffix, sup=sup,
        data_root=data_root,
    )
    return output_dir
