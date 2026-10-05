"""`pairs()`: GreedyReg's own pairing, by name, for splitting a cohort."""

import sadt_greedyreg


def test_patients_are_grouped_and_strays_reported():
    result = sadt_greedyreg.pairs(t1=["C_0001_T1_Or.nii.gz", "C_0003_T1.nii.gz"],
                                  t2=["C_0001_T2_Or.nii.gz"])
    assert [g["key"] for g in result["groups"]] == ["C_0001"]
    assert result["unpaired"] == {"t1": ["C_0003"]}
