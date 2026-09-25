"""The patient key, which is the one thing AREG's engines must agree on."""

import pytest

from sadt_areg_common import catalogs, pairing
from sadt_areg_common.errors import ToolInputError

JAW = set(catalogs.JAW_TOKENS)


def test_the_upstream_test_set_pairs_its_two_timepoints():
    """`A2_UpperT1.vtk` and `A2_UpperT2.vtk` are one patient, not two.

    These are upstream's own AREG_test_scans filenames, verbatim -- not a
    renamed version, because the point is that the tool reads the data its own
    project publishes. The jaw and the timepoint run together with no
    separator, so `uppert1` used to match neither the jaw table nor the
    timepoint one, nothing was dropped, and AREG_IOS refused with "no subject
    has a upper arch at both timepoints".
    """
    t1 = pairing.patient_stem("A2_UpperT1.vtk", also_drop=JAW)
    t2 = pairing.patient_stem("A2_UpperT2.vtk", also_drop=JAW)
    assert t1 == t2 == "A2"


def test_an_identifier_that_merely_ends_in_a_timepoint_is_left_alone():
    """`PAT1` is a patient, not a jaw plus a timepoint.

    The split is attempted only when the PREFIX is a known jaw token: `pa` is
    not one, so nothing happens. Widening it to any camelCase-ish boundary
    would start eating patient identifiers, which is the failure this function
    exists to prevent.
    """
    assert pairing.patient_stem("PAT1.vtk", also_drop=JAW) == "PAT1"


def test_the_separated_spelling_still_works():
    """The common case, unchanged: separators do the job on their own."""
    assert pairing.patient_stem("P1_Upper_T1.vtk", also_drop=JAW) == "P1"
    assert pairing.patient_stem("C_0001_T1_Or.nii.gz") == "C_0001"


def test_a_lone_jaw_or_timepoint_token_is_not_split():
    """Nothing to split when the token is already one thing."""
    assert pairing.patient_stem("Lower_gold.vtk", also_drop=JAW) == "gold"
    assert pairing.patient_stem("subject_T0.nii.gz") == "subject"


def test_the_split_only_fires_when_both_halves_are_droppable():
    """A jaw token glued to something that is not a timepoint stays whole."""
    # `upperx` is not jaw+timepoint, so it survives as one token.
    assert pairing.patient_stem("A2_UpperX.vtk", also_drop=JAW) == "A2_UpperX"


# ---------------------------------------------------------------------------
# One timepoint per side, and a mask is not a subject
# ---------------------------------------------------------------------------
def _scan(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


def test_a_mask_beside_a_scan_is_not_a_second_subject(tmp_path):
    """`<scan>_SegOut/` is what AMASSS and AREG both write, so a folder that has
    been through a run holds masks beside its scans. Counted as subjects, each
    was paired, segmented again and registered: four runs where there is one."""
    _scan(tmp_path / "C_0001_T1_Or.nii.gz")
    _scan(tmp_path / "C_0001_T1_Or_SegOut" / "C_0001_T1_Or_CBMASK-Seg_Pred.nii.gz")
    _scan(tmp_path / "C_0001_T1_Or_SegOut" / "C_0001_T1_Or_MANDMASK-Seg_Pred.nii.gz")

    assert sorted(pairing.discover(str(tmp_path), "Reg")) == ["C_0001"]


def test_a_whole_cohort_given_as_one_timepoint_is_refused(tmp_path):
    """Every shipped cohort is `<name>/{T1,T2}/`. Handed one as T1 and another
    as T2, pairing matched `T1/C_0001` with `T1/C_0001` and `T2/C_0001` with
    `T2/C_0001` -- a baseline registered onto a baseline, reported as a
    success. The names line up perfectly, so nothing downstream can notice."""
    for cohort in ("first", "second"):
        _scan(tmp_path / cohort / "T1" / "C_0001_T1_Or.nii.gz")
        _scan(tmp_path / cohort / "T2" / "C_0001_T2_Or.nii.gz")

    with pytest.raises(ToolInputError, match="more than one timepoint"):
        pairing.pair(str(tmp_path / "first"), str(tmp_path / "second"), "Reg")


def test_pointing_at_the_timepoint_subfolders_works(tmp_path):
    """The same data, used the way the refusal above asks for."""
    _scan(tmp_path / "T1" / "C_0001_T1_Or.nii.gz")
    _scan(tmp_path / "T2" / "C_0001_T2_Or.nii.gz")

    matched = pairing.pair(str(tmp_path / "T1"), str(tmp_path / "T2"), "Reg")

    assert sorted(matched.matched) == ["C_0001"]
