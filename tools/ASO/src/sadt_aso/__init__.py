"""ASO -- Automated Standardized Orientation.

Orients CBCT volumes or intra-oral meshes onto a reference frame. The pipeline
is in dispatch.py; only `run` is public.

One tool, two engines, four modes. Fully-automated CBCT needs landmarks it does
not place itself, and it needs them **mid-pipeline** -- after recentring, which
is the space the landmark tool was always run in. So it either receives them
(`landmarks`) or reaches the tool that makes them through the supervisor, at
the point in the run where it has always been called.
"""

from pathlib import Path
from typing import Literal


# The argument holding one entry per case, so a caller may hand this tool
# three patients of forty instead of all of them. Named rather than guessed:
# it is `input` here and `scans` in the segmentation tools, and a server
# picking the wrong one would silently drop patients from a run.
CASE_INPUT = "input"


def run(
    input: Path,
    reference: Path,
    output_dir: Path,
    modality: Literal["CBCT", "IOS"] = "CBCT",
    automation: Literal["Semi-Automated", "Fully-Automated"] = "Semi-Automated",
    landmarks: Path = "",
    landmark_model: Path = "",
    # Spelled out because `Literal` takes literals only -- it cannot be built
    # from catalogs.CBCT_LANDMARKS. That makes this a second declaration of the
    # same set, which is the thing this contract otherwise avoids, so a test
    # asserts the two agree.
    cbct_landmarks: list[
        Literal[
            "Ba", "C2", "C3", "C4", "LFZyg", "LPo", "N", "RFZyg", "RPo", "S", "A", "ANS",
            "IF", "LInfOr", "LMZyg", "LNC", "LOr", "LPF", "PNS", "RInfOr", "RMZyg", "RNC",
            "ROr", "RPF", "UL1O", "UL1R", "UL2O", "UL2R", "UL3O", "UL3R", "UL4O", "UL4R",
            "UL5O", "UL5R", "UL6DB", "UL6MB", "UL6MP", "UL6O", "UL6R", "UL7O", "UL7R",
            "UR1O", "UR1R", "UR2O", "UR2R", "UR3O", "UR3R", "UR4O", "UR4R", "UR5O", "UR5R",
            "UR6DB", "UR6MB", "UR6MP", "UR6O", "UR6R", "UR7O", "UR7R", "B", "Gn", "LAE",
            "LAF", "LARa", "LCo", "LGo", "LLCo", "LMCo", "LMeF", "LPRa", "LSig", "Me",
            "Pog", "PogL", "LL1O", "LL1R", "LL2O", "LL2R", "LL3O", "LL3R", "LL4O", "LL4R",
            "LL5O", "LL5R", "LL6DB", "LL6MB", "LL6O", "LL6R", "LL7O", "LL7R", "LR1O",
            "LR1R", "LR2O", "LR2R", "LR3O", "LR3R", "LR4O", "LR4R", "LR5O", "LR5R", "LR6DB",
            "LR6MB", "LR6O", "LR6R", "LR7O", "LR7R", "RAE", "RAF", "RARa", "RCo", "RGo",
            "RLCo", "RMCo", "RMeF", "RPRa", "RSig",
        ]
    ] = ["Ba", "S", "N", "RPo", "LPo", "ROr", "LOr"],
    ios_teeth: list[
        Literal[
            "UR8", "UR7", "UR6", "UR5", "UR4", "UR3", "UR2", "UR1", "UL1", "UL2", "UL3",
            "UL4", "UL5", "UL6", "UL7", "UL8", "LL8", "LL7", "LL6", "LL5", "LL4", "LL3",
            "LL2", "LL1", "LR1", "LR2", "LR3", "LR4", "LR5", "LR6", "LR7", "LR8",
        ]
    ] = ["UR6", "UL1", "UL6", "LL6", "LR1", "LR6"],
    ios_landmark_types: list[
        Literal["O", "MB", "DB", "CB", "CL", "OIP", "R", "RIP"]
    ] = ["O"],
    ios_jaws: list[Literal["Upper", "Lower"]] = ["Upper", "Lower"],
    ios_occlusion: Literal[
        "Orient each jaw independently", "Upper drives Lower", "Lower drives Upper"
    ] = "Orient each jaw independently",
    dicom_input: bool = False,
    output_suffix: str = "Or",
    max_triplets: int = 2500,
    seed: int = 0,
    *,
    sup=None,
) -> Path:
    """Orient CBCT scans or intra-oral scans onto a standard reference frame.

    Args:
        input: One scan (.nii/.nii.gz/.nrrd/.nrrd.gz/.gipl/.gipl.gz), one
            intra-oral mesh (.vtk/.stl), or a folder of either for a batch.
            Folders are searched recursively and the output keeps their tree.
            In Semi-Automated mode the landmark files (.mrk.json) travel beside
            the scans, paired by name.
        reference: The already-oriented case defining the target frame -- its
            landmark file for CBCT, its landmarks and meshes for IOS.
        output_dir: Where results are written -- per patient, the oriented scan,
            its landmarks and the transform (.tfm) -- plus `ASO_report.json`.
            Nothing is written outside it.
        modality: CBCT volumes or intra-oral surface scans. Never inferred from
            the file extension: a folder can hold either, and guessing wrong
            means orienting a patient against the wrong reference and calling
            it a success.
        automation: Semi-Automated registers on landmarks you send.
            Fully-Automated predicts CBCT landmarks first, and orients IOS
            meshes from the tooth labels they already carry.
        landmarks: Optional folder of landmark files (.mrk.json) to use instead
            of the ones beside the scans -- which is what makes Fully-Automated
            CBCT work with no supervisor: run the landmark tool yourself and
            pass its output here. Paired to scans by name, like the ones in the
            input tree.
        landmark_model: The model bundle the landmark tool predicts with.
            Required by Fully-Automated CBCT when the landmarks are not
            supplied, and ignored otherwise. Singular, and that matters: the
            server publishes an argument named `model`, `*_model` or
            `*_reference` as a name picked from the weights it hosts, and
            anything else as a file a clinician may upload. `landmark_models`
            missed that rule by one letter and would have asked for a 4.7 GB
            bundle from a laptop.
        cbct_landmarks: Which landmarks to register on; at least 3, and they
            must exist in the reference. The default seven are the points both
            published reference bundles are built on.
        ios_teeth: Which teeth to register on, 3 or 4 spread across each arch.
        ios_landmark_types: Which point on each tooth, combined with the teeth
            above into `<tooth><type>` keys, e.g. UR6 x O -> UR6O. Used for the
            arches that came with landmarks beside the mesh.
        ios_jaws: Which jaws to orient.
        ios_occlusion: Whether each jaw is oriented on its own, or one jaw's
            transform is applied to the other so the occlusion is kept.
        dicom_input: Force DICOM conversion. Detected from the input on its own,
            so this is only for a caller who knows better than the detector.
        output_suffix: Added to each output name, e.g. `patient1_Or.nii.gz`.
        max_triplets: Landmark triplets the coarse alignment searches before
            running ICP. Lower is faster and less thorough; it changes the
            result, so the run report records it.
        seed: Seed for that search, so a run is reproducible.

    Returns:
        The output directory, holding the oriented cases and the run report.
    """
    # Imported HERE, not at module level: dispatch pulls the two engines, which
    # pull SimpleITK, VTK and numpy. describe.py imports this module on every CI
    # run to publish the schema, and that must not cost an imaging stack -- the
    # rule covers the whole import chain, not just this file.
    from .dispatch import orient

    output_dir = Path(output_dir)
    orient(
        input_path=str(input),
        reference_path=str(reference),
        output_dir=str(output_dir),
        modality=modality,
        automation=automation,
        cbct_landmarks=cbct_landmarks,
        ios_teeth=ios_teeth,
        ios_landmark_types=ios_landmark_types,
        ios_jaws=ios_jaws,
        ios_occlusion=ios_occlusion,
        landmarks_path=str(landmarks) if landmarks else "",
        landmark_model=str(landmark_model) if landmark_model else "",
        dicom_input=dicom_input,
        output_suffix=output_suffix,
        max_triplets=max_triplets,
        seed=seed,
        sup=sup,
    )
    return output_dir
