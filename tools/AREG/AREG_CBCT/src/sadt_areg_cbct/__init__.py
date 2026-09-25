"""AREG_CBCT -- register a follow-up CBCT onto its baseline.

elastix, rigid, restricted to the anatomy that has not changed between the two
timepoints: the cranial base, the mandible or the maxilla, taken as masks.

Split out of the former single `AREG`, which served both modalities from one
schema and one virtualenv. The reason is the same as ALI's: the intraoral
engine needs pytorch3d and therefore torch 2.11, this one needs neither, and
while they shared an environment neither could be pinned without the other.
They now share only `sadt_areg_common` -- the patient-key convention, the
catalogs, and the scan-extension table -- which has no dependencies at all.

The masks this registers on, and the orientation both timepoints must share,
come from other tools reached through the supervisor; see the CBCT half of
`tools.py`. `AREG_CBCT -> ASO -> ALI_CBCT` is the deepest chain in the family.
"""

from pathlib import Path
from typing import Literal

from sadt_areg_common import pairing

from .dispatch import main


# What each path argument can read: the client's file dialog and the server's
# upload check both narrow from here. DERIVED from the table the pairing
# registers against, never retyped.
#
# `t1_masks` takes volumes too -- a mask IS a volume, one label per voxel --
# and DICOM is deliberately absent: a series is a folder of `.dcm`, which the
# `dicom_input` switch handles rather than an extension filter.
ACCEPTS = {
    "t1": pairing.SCAN_EXTENSIONS,
    "t2": pairing.SCAN_EXTENSIONS,
    "t1_masks": pairing.SCAN_EXTENSIONS,
}


def run(
    t1: Path,
    t2: Path,
    output_dir: Path,
    automation: Literal[
        "Semi-Automated", "Fully-Automated", "Oriented + Fully-Automated"
    ] = "Fully-Automated",
    # Spelled out because `Literal` takes literals only -- it cannot be built
    # from catalogs.REGION_CHOICES. That makes this a second declaration of the
    # same set, which is the thing this contract otherwise avoids, so a test
    # asserts the two agree.
    regions: list[
        Literal["Cranial base", "Mandible", "Maxilla"]
    ] = ["Cranial base"],
    t1_masks: Path = "",
    # The second group of boxes the original module shows, and it is NOT the
    # regions above: those decide what the registration is masked to, these
    # decide what comes back to look at. Spelled out for the same reason
    # `regions` is -- `Literal` cannot be built from the catalog.
    segmentations: list[
        Literal[
            "Cranial base", "Cervical vertebra", "Mandible", "Maxilla",
            "Skin", "Upper airway",
        ]
    ] = [],
    segmentation_model: Path = "",
    segmentation_label: int = 0,
    reference: Path = "",
    landmark_model: Path = "",
    dicom_input: bool = False,
    output_suffix: str = "Reg",
    *,
    sup=None,
    data_root=None,
) -> Path:
    """Register a follow-up CBCT onto its baseline, so the two can be compared.

    Args:
        t1: The baseline scans -- one volume or a folder of them, searched
            recursively. A DICOM series is converted when `dicom_input` is set.
        t2: The follow-up scans, paired to T1 by patient key.
        output_dir: Where the registered scans, their transforms and
            `AREG_report.json` are written. Nothing is written outside it.
        automation: Semi-Automated takes your own masks; Fully-Automated
            segments them; Oriented + Fully-Automated orients both timepoints
            first, which needs a reference.
        regions: The anatomy to register on -- what has NOT changed between the
            timepoints. The one argument a clinician must actually think about.
        t1_masks: Your own T1 segmentation masks, instead of having them
            segmented for you.
        segmentations: Anatomy to segment and return beside the registration,
            for the modes that segment. Independent of `regions`: ticking the
            skin does not register on it, and registering on the mandible does
            not return a mandible you can open. None by default -- a
            registration run returns a registration.
        segmentation_model: The mask model bundle, for the modes that segment.
            Left empty -- which is what a panel sends -- the AMASSS bundle this
            deployment publishes for AREG is used, there being no second answer
            to the question.
        segmentation_label: Which label value in the masks to register on.
        reference: The frame the scans are oriented onto before registering.
        landmark_model: The landmark bundle that orientation step predicts
            with.
        dicom_input: The inputs are DICOM series rather than volumes.
        output_suffix: Added to each output name, e.g. `scan_Reg.nii.gz`.

    Returns:
        The output directory.
    """
    # itk-elastix and SimpleITK are imported inside the engine: CI imports this
    # module on every PR to publish the schema, and that must not cost them.
    return main(
        t1=t1,
        t2=t2,
        output_dir=output_dir,
        automation=automation,
        cbct_regions=regions,
        t1_masks=t1_masks,
        segmentations=segmentations,
        segmentation_model=segmentation_model,
        segmentation_label=segmentation_label,
        cbct_reference=reference,
        landmark_model=landmark_model,
        dicom_input=dicom_input,
        output_suffix=output_suffix,
        sup=sup,
        data_root=data_root,
    )
