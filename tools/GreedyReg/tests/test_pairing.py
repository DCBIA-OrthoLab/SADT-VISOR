"""Who is a patient, and what is said about the files that pair with nothing.

Identity comes from `sadt_areg_common.pairing`, not from a rule of this tool's
own -- CONTRIBUTING.md lists identity derivation as one of the three things
that belong in a shared package, because two tools exchanging files by name
have to agree on where the identifier ends. These tests pin what that shared
rule gives GreedyReg, and what the report says about the leftovers.
"""

import pytest
from sadt_areg_common import pairing

import sadt_greedyreg
from conftest import cohort, make_scan


def paired(root, suffix="registered"):
    return pairing.pair(str(root / "t1"), str(root / "t2"), suffix)


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def test_two_timepoints_pair_through_different_decorations(tmp_path):
    """The two folders are written by different steps and rarely agree on
    anything but the patient: `_Or` is what ASO leaves, `_scan` what a
    clinician types."""
    make_scan(tmp_path / "t1" / "A1_T1_Or.nii.gz")
    make_scan(tmp_path / "t2" / "A1_T2_scan.nii.gz")

    matched = paired(tmp_path)

    assert list(matched.matched) == ["A1"]
    assert matched.matched["A1"]["t1"].endswith("A1_T1_Or.nii.gz")
    assert matched.matched["A1"]["t2"].endswith("A1_T2_scan.nii.gz")


def test_a_matched_pair_carries_both_paths_under_t1_and_t2(tmp_path):
    """`Pairing.matched` is {key: {"t1": path, "t2": path}}, and the batch
    reads exactly those two keys."""
    cohort(tmp_path, ["A1"])

    entry = paired(tmp_path).matched["A1"]

    assert sorted(entry) == ["t1", "t2"]
    assert entry["t1"].endswith("A1_T1.nii.gz") and entry["t2"].endswith("A1_T2.nii.gz")


def test_the_timepoint_token_may_be_written_in_either_case(tmp_path):
    make_scan(tmp_path / "t1" / "A1_t1.nii.gz")
    make_scan(tmp_path / "t2" / "A1_T2.nii.gz")

    assert list(paired(tmp_path).matched) == ["A1"]


def test_the_patient_identifier_keeps_its_own_case_and_that_case_counts(tmp_path):
    """`a1` and `A1` are two patients. Two names differing only in case are
    two files a filesystem is willing to hold side by side, so collapsing them
    would be a guess -- and the key ends up in output paths."""
    make_scan(tmp_path / "t1" / "a1_T1.nii.gz")
    make_scan(tmp_path / "t2" / "A1_T2.nii.gz")

    matched = paired(tmp_path)

    assert not matched
    assert matched.unmatched_report() == {"t1_without_t2": ["a1"], "t2_without_t1": ["A1"]}


def test_a_repeated_timepoint_token_is_dropped_every_time(tmp_path):
    """`P1_T1_T1.nii.gz` is one patient, not a patient called `P1_T1`."""
    make_scan(tmp_path / "t1" / "P1_T1_T1.nii.gz")
    make_scan(tmp_path / "t2" / "P1_T2.nii.gz")

    assert list(paired(tmp_path).matched) == ["P1"]


def test_an_identifier_that_merely_looks_like_a_token_survives(tmp_path):
    """`T1000` starts with the timepoint token and is not one: matching is on
    whole tokens, never on substrings."""
    make_scan(tmp_path / "t1" / "T1000_T1.nii.gz")
    make_scan(tmp_path / "t2" / "T1000_T2.nii.gz")

    assert list(paired(tmp_path).matched) == ["T1000"]


def test_an_identifier_that_ends_in_a_token_survives(tmp_path):
    """`PAT1` ends in `t1`, and the concatenated split fires only when what
    precedes the timepoint is a known jaw token -- `pa` is not one."""
    make_scan(tmp_path / "t1" / "PAT1_T1.nii.gz")
    make_scan(tmp_path / "t2" / "PAT1_T2.nii.gz")

    assert list(paired(tmp_path).matched) == ["PAT1"]


@pytest.mark.parametrize("filename,patient", [
    # The table in README.md, row for row. Upstream's `^([A-Za-z]+\d+)` wanted
    # letters then digits with nothing between, so it skipped every one of
    # these -- and said nothing, because a file that matched nothing never
    # entered the dict.
    ("C_0001_T1_Or.nii.gz", "C_0001"),
    ("MAMP_0002_T1.nii.gz", "MAMP_0002"),
    ("IC_0005.nii.gz", "IC_0005"),
])
def test_the_readme_table_of_names_upstream_dropped(filename, patient):
    assert pairing.patient_stem(filename) == patient


def test_two_scans_of_one_patient_are_kept_apart(tmp_path):
    """The last row of that table: upstream keyed on the ID prefix, so
    `A1_scan` and `A1_other` became one patient and one of them was lost."""
    assert pairing.patient_stem("A1_scan.nii.gz") != pairing.patient_stem("A1_other.nii.gz")


# ---------------------------------------------------------------------------
# What did not match, and what is said about it
# ---------------------------------------------------------------------------

def test_a_t1_with_no_t2_is_named_in_the_report(tmp_path):
    cohort(tmp_path, ["A1"])
    make_scan(tmp_path / "t1" / "Z9_T1.nii.gz")

    matched = paired(tmp_path)

    assert list(matched.matched) == ["A1"]
    assert matched.unmatched_report() == {"t1_without_t2": ["Z9"], "t2_without_t1": []}


def test_a_t2_with_no_t1_is_named_in_the_report(tmp_path):
    cohort(tmp_path, ["A1"])
    make_scan(tmp_path / "t2" / "C3_T2.nii.gz")

    matched = paired(tmp_path)

    assert matched.unmatched_report() == {"t1_without_t2": [], "t2_without_t1": ["C3"]}


def test_both_sides_unmatched_are_reported_separately(tmp_path):
    make_scan(tmp_path / "t1" / "Z9_T1.nii.gz")
    make_scan(tmp_path / "t2" / "C3_T2.nii.gz")

    matched = paired(tmp_path)

    assert len(matched) == 0
    assert matched.unmatched_report() == {"t1_without_t2": ["Z9"], "t2_without_t1": ["C3"]}


def test_an_empty_t1_folder_leaves_every_t2_unmatched(tmp_path):
    (tmp_path / "t1").mkdir()
    make_scan(tmp_path / "t2" / "A1_T2.nii.gz")
    make_scan(tmp_path / "t2" / "B2_T2.nii.gz")

    matched = paired(tmp_path)

    assert matched.unmatched_report() == {"t1_without_t2": [], "t2_without_t1": ["A1", "B2"]}


def test_an_empty_t2_folder_leaves_every_t1_unmatched(tmp_path):
    make_scan(tmp_path / "t1" / "A1_T1.nii.gz")
    make_scan(tmp_path / "t1" / "B2_T1.nii.gz")
    (tmp_path / "t2").mkdir()

    matched = paired(tmp_path)

    assert matched.unmatched_report() == {"t1_without_t2": ["A1", "B2"], "t2_without_t1": []}


def test_a_folder_that_does_not_exist_reads_as_an_empty_one(tmp_path):
    """A caller sending a path that is not there gets the unmatched report,
    not an OSError from halfway through a walk."""
    make_scan(tmp_path / "t2" / "A1_T2.nii.gz")

    matched = pairing.pair(str(tmp_path / "absent"), str(tmp_path / "t2"), "registered")

    assert matched.unmatched_report() == {"t1_without_t2": [], "t2_without_t1": ["A1"]}


def test_the_unmatched_report_is_empty_on_both_sides_when_everything_pairs(tmp_path):
    cohort(tmp_path, ["A1", "B2"])

    assert paired(tmp_path).unmatched_report() == {"t1_without_t2": [], "t2_without_t1": []}


# ---------------------------------------------------------------------------
# Folders
# ---------------------------------------------------------------------------

def test_nested_folders_pair_on_the_path_not_the_base_name(tmp_path):
    """The key is the path relative to the input root, which is what stops
    `scan_T1.nii.gz` in two subfolders from being one patient."""
    make_scan(tmp_path / "t1" / "site_a" / "A1_T1.nii.gz")
    make_scan(tmp_path / "t2" / "site_a" / "A1_T2.nii.gz")

    assert list(paired(tmp_path).matched) == ["site_a/A1"]


def test_the_same_name_in_two_subfolders_stays_two_patients(tmp_path):
    make_scan(tmp_path / "t1" / "site_a" / "scan_T1.nii.gz")
    make_scan(tmp_path / "t1" / "site_b" / "scan_T1.nii.gz")
    make_scan(tmp_path / "t2" / "site_a" / "scan_T2.nii.gz")
    make_scan(tmp_path / "t2" / "site_b" / "scan_T2.nii.gz")

    assert sorted(paired(tmp_path).matched) == ["site_a/scan", "site_b/scan"]


def test_a_patient_that_moved_between_subfolders_does_not_pair(tmp_path):
    make_scan(tmp_path / "t1" / "site_a" / "A1_T1.nii.gz")
    make_scan(tmp_path / "t2" / "site_b" / "A1_T2.nii.gz")

    matched = paired(tmp_path)

    assert matched.unmatched_report() == {
        "t1_without_t2": ["site_a/A1"], "t2_without_t1": ["site_b/A1"]}


def test_files_that_are_not_scans_are_ignored(tmp_path):
    cohort(tmp_path, ["A1"])
    (tmp_path / "t1" / "notes.txt").write_text("acquisition notes\n")
    (tmp_path / "t2" / "protocol.pdf").write_bytes(b"%PDF-")

    assert list(paired(tmp_path).matched) == ["A1"]
    assert paired(tmp_path).unmatched_report() == {"t1_without_t2": [], "t2_without_t1": []}


def test_hidden_files_are_ignored(tmp_path):
    """macOS leaves `._A1_T1.nii.gz` beside a real scan on a shared drive."""
    cohort(tmp_path, ["A1"])
    make_scan(tmp_path / "t1" / "._A1_T1.nii.gz")

    assert paired(tmp_path).unmatched_report() == {"t1_without_t2": [], "t2_without_t1": []}


# ---------------------------------------------------------------------------
# What the batch does with all of it
# ---------------------------------------------------------------------------

def test_nothing_matching_names_what_each_folder_held(tmp_path, greedy):
    """Counts alone say the run failed; the names say why -- almost always one
    folder using a decoration the other does not."""
    make_scan(tmp_path / "t1" / "Z9_T1.nii.gz")
    make_scan(tmp_path / "t2" / "C3_T2.nii.gz")

    with pytest.raises(ValueError) as failure:
        sadt_greedyreg.run(t1=tmp_path / "t1", t2=tmp_path / "t2",
                           output_dir=tmp_path / "out")

    message = str(failure.value)
    assert "No patient appears in both" in message
    assert "Z9" in message and "C3" in message
    assert not greedy.commands, "no registration should have been attempted"


def test_nothing_matching_is_not_a_successful_run_of_zero_patients(tmp_path, greedy):
    (tmp_path / "t1").mkdir()
    (tmp_path / "t2").mkdir()

    with pytest.raises(ValueError, match="No patient appears in both"):
        sadt_greedyreg.run(t1=tmp_path / "t1", t2=tmp_path / "t2",
                           output_dir=tmp_path / "out")

    assert not (tmp_path / "out" / "GreedyReg_report.json").exists()


def test_a_long_list_of_unmatched_patients_is_truncated(tmp_path, greedy):
    """The message is read in a Slicer dialog, so it names the first few and
    says there are more rather than printing forty."""
    for index in range(15):
        make_scan(tmp_path / "t1" / f"P{index:02d}_T1.nii.gz")
    (tmp_path / "t2").mkdir()

    with pytest.raises(ValueError) as failure:
        sadt_greedyreg.run(t1=tmp_path / "t1", t2=tmp_path / "t2",
                           output_dir=tmp_path / "out")

    assert "15 T1-only" in str(failure.value)
    assert "..." in str(failure.value)


def test_the_report_carries_what_could_not_be_paired(tmp_path, greedy):
    """A run that registered 1 of 3 has to say where the other two went."""
    import json

    cohort(tmp_path, ["A1"])
    make_scan(tmp_path / "t1" / "Z9_T1.nii.gz")
    make_scan(tmp_path / "t2" / "C3_T2.nii.gz")

    out = sadt_greedyreg.run(t1=tmp_path / "t1", t2=tmp_path / "t2",
                             output_dir=tmp_path / "out")

    report = json.loads((out / "GreedyReg_report.json").read_text())
    assert report["unmatched"] == {"t1_without_t2": ["Z9"], "t2_without_t1": ["C3"]}
    assert report["summary"]["cases"] == 1


def test_a_previous_run_left_in_the_t1_folder_is_not_taken_for_the_scan(tmp_path, greedy):
    """`A1_registered.nii.gz` sorts before `A1_T1.nii.gz`, so without the
    previous-output rule the second run would register its own output."""
    cohort(tmp_path, ["A1"])
    make_scan(tmp_path / "t1" / "A1_registered.nii.gz")

    sadt_greedyreg.run(t1=tmp_path / "t1", t2=tmp_path / "t2",
                       output_dir=tmp_path / "out")

    fixed = greedy.registration_for("A1")[greedy.registration_for("A1").index("-i") + 1]
    assert fixed.endswith("A1_T1.nii.gz")
