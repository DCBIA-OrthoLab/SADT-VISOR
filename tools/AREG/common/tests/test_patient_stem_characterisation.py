"""What `patient_stem` answers for a wide set of real cohort file names.

A characterisation table rather than a set of propositions: every row was first
recorded from the implementation as it stood, then read one by one and marked
right or wrong. The rows marked `# WAS:` are the ones the token-boundary fix
changed, and each carries the answer it used to give -- so this file is both
the regression pin for the 400+ names that must NOT move and the record of the
handful that had to.

Every name here is real: harvested from the tools' own fixtures and from the
cohort data staged under `DATA/{AREG,ASO,AMASSS,ALI}/testfiles/`, except the
`P_Seg1` family, which is the naming the defect was reported against.

Nothing in this file is a judgement about what a GOOD file name is. A key only
has to be the same for a subject's two timepoints and different between two
subjects; these rows say which names that held for.
"""

import pytest

from sadt_areg_common import catalogs, pairing

JAW = set(catalogs.JAW_TOKENS)

# Every anatomy and mask word, the set `discover_masks` drops so that
# `P1_MAND_seg.nii.gz` keys to the `P1` its scan keys to.
ANATOMY = {
    token for group in catalogs.REGION_TOKENS.values() for token in group
} | set(catalogs.MASK_TOKENS)


# ---------------------------------------------------------------------------
# The plain patient key: no `also_drop`, timepoint stripped.
# ---------------------------------------------------------------------------

PLAIN = [
    # -- The AREG cohort staged in DATA/AREG/testfiles, verbatim. Both
    #    timepoints of C_0001 reduce to one key, which IS the pairing.
    ("C_0001_T1.nii.gz", "C_0001"),
    ("C_0001_T2.nii.gz", "C_0001"),
    ("C_0001_T1_Or.nii.gz", "C_0001"),
    ("C_0001_T2_Or.nii.gz", "C_0001"),
    ("C_0001_T1_Or_CBMASK-Seg_Pred.nii.gz", "C_0001"),
    ("C_0001_T2_Or_MANDMASK-Seg_Pred.nii.gz", "C_0001"),
    ("C_0001_T1_scan.nii.gz", "C_0001"),
    ("IC_0005.nii.gz", "IC_0005"),
    ("IC_0005_lm_Pred.mrk.json", "IC_0005"),
    ("P_0001_T2.nii.gz", "P_0001"),
    ("Pat_0002.nii.gz", "Pat_0002"),

    # -- Upstream's own IOS test set. The jaw and the timepoint are glued.
    ("A2_UpperT1.vtk", "A2_Upper"),
    ("A2_UpperT2.vtk", "A2_Upper"),
    ("A2_UpperX.vtk", "A2_UpperX"),
    ("P1_Upper_T1.vtk", "P1_Upper"),
    ("P001_T2_U.vtk", "P001_U"),
    ("T1_test_file.vtk", "test_file"),
    ("T2_test_file.vtk", "test_file"),

    # -- An identifier that merely ENDS in something timepoint-shaped.
    ("PAT1.vtk", "PAT1"),
    ("PAT1_T1.nii.gz", "PAT1"),
    ("PAT1_T2.nii.gz", "PAT1"),
    ("T1000_T1.nii.gz", "T1000"),
    ("P1_T10.nii.gz", "P1_T10"),
    ("subject_T0.nii.gz", "subject"),

    # -- Extensions, compound ones included.
    ("P1.2_scan.nii.gz", "P1_2"),
    ("Patient.01_Scan.nii.gz", "Patient_01"),
    ("study.v2.nrrd", "study_v2"),
    ("P1_T1.gipl.gz", "P1"),
    ("P1_T1.nrrd.gz", "P1"),
    ("P1_lm.mrk.json", "P1"),

    # -- A suffix a previous run left, as a WHOLE token. All still dropped.
    ("P1_scan.nii.gz", "P1"),
    ("P1_Scan.nii.gz", "P1"),
    ("P1_seg.nii.gz", "P1"),
    ("P1_Seg.nii.gz", "P1"),
    ("P1_Or.nii.gz", "P1"),
    ("P1_MERGED.nii.gz", "P1"),
    ("P1_OutReg.nii.gz", "P1"),
    ("P1_lm_Pred.mrk.json", "P1"),
    ("SMITH_Or.nii.gz", "SMITH"),
    ("Segovia_scan.nii.gz", "Segovia"),
    ("MG_test_scan.nii.gz", "MG_test"),
    # Truncation, not deletion: what the suffix introduces goes with it.
    ("A1_seg_CBMASK.nii.gz", "A1"),
    ("patient01_Pred_MERGED.nii.gz", "patient01_Pred"),
    # A suffix at index 0 is the whole name and is left to `_drop_tokens`.
    ("_MERGED.nii.gz", "MERGED"),

    # -- Anatomy words stay in the key unless the caller asks for them.
    ("P1_T1_MAND_seg.nii.gz", "P1_MAND"),
    ("C_0001_T1_MAND_Seg.nii.gz", "C_0001_MAND"),
    ("P1_CBCT_seg.nii.gz", "P1_CBCT"),

    # -- Patients whose own name is an anatomy or vocabulary word. These are
    #    the rows the region-token defect cost, and they still hold.
    ("MAX_01_scan.nii.gz", "MAX_01"),
    ("MAXIMUS_01.nii.gz", "MAXIMUS_01"),
    ("Mandy_3.nii.gz", "Mandy_3"),
    ("MDX_1.nii.gz", "MDX_1"),
    ("Registry_7.nii.gz", "Registry_7"),
    ("Transformer_01.nii.gz", "Transformer_01"),

    # -- The defect. A suffix matched at ANY index truncated the identifier.
    ("P_Seg1_T1.nii.gz", "P_Seg1"),          # WAS: "P"
    ("P_Seg2_T1.nii.gz", "P_Seg2"),          # WAS: "P" -- same key as P_Seg1
    ("P_Seg10_T1.nii.gz", "P_Seg10"),        # WAS: "P"
    ("P_Scan1_T1.nii.gz", "P_Scan1"),        # WAS: "P"
    ("P_scanned_T1.nii.gz", "P_scanned"),    # WAS: "P"
    ("P_Orion_T1.nii.gz", "P_Orion"),        # WAS: "P"
    ("P_lmk_T1.nii.gz", "P_lmk"),            # WAS: "P"
    ("P_MERGEDATA_T1.nii.gz", "P_MERGEDATA"),  # WAS: "P"
    ("P_OutRegion_T1.nii.gz", "P_OutRegion"),  # WAS: "P"
    ("P_SegOrbit_T1.nii.gz", "P_SegOrbit"),  # WAS: "P"
    ("A_Segmentation.nii.gz", "A_Segmentation"),  # WAS: "A"
    ("SMITH_ORTHO_Scan.nii.gz", "SMITH_ORTHO"),   # WAS: "SMITH"
    ("SMITH_ORTHO_ROI.mrk.json", "SMITH_ORTHO_ROI"),  # WAS: "SMITH"
    ("T1_01_U_segmented.vtk", "01_U_segmented"),  # WAS: "01_U"
    ("Extraction_scanned.pdf.json", "Extraction_scanned_pdf"),  # WAS: "Extraction"

    # -- Transform names, which carry no suffix the shared table knows. Their
    #    vocabulary is AutoMatrix's and GreedyReg's `also_drop`, below.
    ("C_0001_CB_Reg_transform.tfm", "C_0001_CB_Reg_transform"),
    ("P1_CBReg_matrix.tfm", "P1_CBReg_matrix"),
    ("A1_transform.mat", "A1_transform"),
]


@pytest.mark.parametrize("filename,key", PLAIN)
def test_the_plain_patient_key(filename, key):
    assert pairing.patient_stem(filename) == key


# ---------------------------------------------------------------------------
# The same names under the `also_drop` sets the real callers pass.
# ---------------------------------------------------------------------------

WITH_ALSO_DROP = [
    # AREG_IOS: the jaw vocabulary, so both arches of a subject share a key.
    (JAW, "A2_UpperT1.vtk", "A2"),
    (JAW, "A2_UpperT2.vtk", "A2"),
    (JAW, "P1_Upper_T1.vtk", "P1"),
    (JAW, "P1_T1_Lower.vtk", "P1"),
    (JAW, "Lower_gold.vtk", "gold"),
    (JAW, "P_Seg1_Upper_T1.vtk", "P_Seg1"),   # WAS: "P"
    # `MAX_01` is a patient, and the jaw table holds "max". The first token is
    # not protected here the way AutoCrop3D protects it, so this is the shape
    # of key the IOS engine has always produced.
    (JAW, "MAX_01_scan.nii.gz", "01"),

    # `discover_masks`: anatomy plus the mask words, so a mask keys to its scan.
    (ANATOMY, "P1_T1_MAND_seg.nii.gz", "P1"),
    (ANATOMY, "A1_seg_CBMASK.nii.gz", "A1"),
    (ANATOMY, "P3_MAND_seg.nii.gz", "P3"),
    (ANATOMY, "P_Seg1_T1_MAND_seg.nii.gz", "P_Seg1"),   # WAS: "P"
    # `Segmentation` is in MASK_TOKENS, so this one keyed to "A" either way --
    # by truncation before, by token drop now.
    (ANATOMY, "A_Segmentation.nii.gz", "A"),

    # AutoMatrix: the transform vocabulary the shared table does not know.
    (("transform", "matrix", "warp", "reg", "cb", "mand", "max", "md", "mx"),
     "C_0001_CB_Reg_transform.tfm", "C_0001"),
    # What AREG CBCT writes now, with the T2 named in it.
    (("transform", "matrix", "warp", "reg", "cb", "mand", "max", "md", "mx"),
     "C_0001_T2_CB_Reg_transform.tfm", "C_0001"),
    (("transform", "matrix", "warp", "reg", "cb", "mand", "max", "md", "mx"),
     "P1_MAND_MAX_CB_reg_matrix_warp_transform.tfm", "P1"),
    (("transform",), "A1_T1_transform.mat", "A1"),
    (("transform",), "P_Seg1_transform.mat", "P_Seg1"),   # WAS: "P"
]


@pytest.mark.parametrize("also_drop,filename,key", WITH_ALSO_DROP)
def test_the_patient_key_with_a_caller_vocabulary(also_drop, filename, key):
    assert pairing.patient_stem(filename, also_drop=also_drop) == key


# ---------------------------------------------------------------------------
# `drop_timepoint=False`: the key of ONE SCAN, which is what a landmark file
# needs when both timepoints sit in the same folder.
# ---------------------------------------------------------------------------

SCAN_KEYS = [
    ("P1_T1_lm.mrk.json", "P1_T1"),
    ("P1_T2_scan.nii.gz", "P1_T2"),
    ("H10_T1_L_MG_edited.json", "H10_T1_L_MG_edited"),
    ("P_Seg1_T1.nii.gz", "P_Seg1_T1"),   # WAS: "P"
    ("P_Seg1_T2.nii.gz", "P_Seg1_T2"),   # WAS: "P"
]


@pytest.mark.parametrize("filename,key", SCAN_KEYS)
def test_the_scan_key_keeps_the_timepoint(filename, key):
    assert pairing.patient_stem(filename, drop_timepoint=False) == key
