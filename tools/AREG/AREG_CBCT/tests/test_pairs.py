"""`pairs()`: how a client may split a CBCT cohort without unpairing it."""

import sadt_areg_cbct


def test_a_cohort_folder_given_for_both_timepoints_is_split_by_patient():
    cohort = ["T1/C_0001_T1_Or.nii.gz", "T1/C_0001_T1_Or_SegOut/C_0001_T1_Or_CBMASK-Seg_Pred.nii.gz",
              "T2/C_0001_T2_Or.nii.gz", "T1/C_0002_T1_Or.nii.gz", "T2/C_0002_T2_Or.nii.gz"]
    result = sadt_areg_cbct.pairs(t1=cohort, t2=cohort)
    groups = {g["key"]: g["entries"] for g in result["groups"]}
    assert groups["C_0001"] == {"t1": ["T1/C_0001_T1_Or.nii.gz", "T1/C_0001_T1_Or_SegOut"],
                                "t2": ["T2/C_0001_T2_Or.nii.gz"]}
    assert set(groups) == {"C_0001", "C_0002"}


def test_it_is_declared_for_both_timepoints():
    assert sadt_areg_cbct.PAIRED == {"axes": ["t1", "t2"]}
