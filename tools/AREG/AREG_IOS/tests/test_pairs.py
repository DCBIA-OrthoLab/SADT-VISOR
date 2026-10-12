"""`pairs()`: a patient travels with every arch it has at each timepoint."""

import sadt_areg_ios


def test_both_arches_of_a_patient_go_together():
    t1 = ["A2_UpperT1.vtk", "A2_LowerT1.vtk", "B7_UpperT1.vtk"]
    t2 = ["A2_UpperT2.vtk", "A2_LowerT2.vtk", "C9_UpperT2.vtk"]
    result = sadt_areg_ios.pairs(t1=t1, t2=t2)
    groups = {g["key"]: g["entries"] for g in result["groups"]}
    assert groups == {"A2": {"t1": ["A2_LowerT1.vtk", "A2_UpperT1.vtk"],
                             "t2": ["A2_LowerT2.vtk", "A2_UpperT2.vtk"]}}
    assert result["unpaired"] == {"t1": ["B7"], "t2": ["C9"]}
