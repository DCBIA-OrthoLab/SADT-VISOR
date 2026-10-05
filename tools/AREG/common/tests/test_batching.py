"""Which files go together when a paired cohort is split into batches.

Names only, through the real pairing on a tree of empty files. What must hold:
every batch carries the same patients on both sides; what a run wrote beside a
scan goes with it; a DICOM series moves as one folder; a patient present on
one side only is reported and never sent; the placeholder tree never leaks.
"""

from sadt_areg_common import batching, pairing


def _match(roots):
    t1_root = pairing.timepoint_root(roots["t1"], "T1")
    t2_root = pairing.timepoint_root(roots["t2"], "T2")
    found = pairing.pair(t1_root, t2_root, "")
    every = {"t1": pairing.discover(t1_root, ""), "t2": pairing.discover(t2_root, "")}
    matched = {k: {"t1": [v["t1"]], "t2": [v["t2"]]} for k, v in found.matched.items()}
    unmatched = {"t1": {k: [every["t1"][k]] for k in found.t1_only},
                 "t2": {k: [every["t2"][k]] for k in found.t2_only}}
    return {"t1": t1_root, "t2": t2_root}, matched, unmatched


def _cohort():
    names = []
    for p in ("P01", "P02", "P03", "P04"):
        names += [f"T1/{p}_T1_Or.nii.gz", f"T1/{p}_T1_Or_SegOut/{p}_T1_Or_MANDMASK-Seg_Pred.nii.gz"]
    for p in ("P01", "P02", "P03", "P05"):
        names.append(f"T2/{p}_T2_Or.nii.gz")
    names += ["T1/P06/IMG0001", "T1/P06/IMG0002", "T2/P06/IMG0001", "README.txt"]
    return names


def test_each_patient_is_a_group_with_both_sides():
    result = batching.pairs_by_name({"t1": _cohort(), "t2": _cohort()}, _match)
    groups = {g["key"]: g["entries"] for g in result["groups"]}
    assert set(groups) == {"P01", "P02", "P03", "P06"}
    assert groups["P01"] == {"t1": ["T1/P01_T1_Or.nii.gz", "T1/P01_T1_Or_SegOut"],
                             "t2": ["T2/P01_T2_Or.nii.gz"]}


def test_a_dicom_series_moves_as_its_folder_and_its_placeholder_never_leaks():
    result = batching.pairs_by_name({"t1": _cohort(), "t2": _cohort()}, _match)
    p06 = [g for g in result["groups"] if g["key"] == "P06"][0]["entries"]
    assert p06 == {"t1": ["T1/P06"], "t2": ["T2/P06"]}
    every = [e for g in result["groups"] for es in g["entries"].values() for e in es]
    every += [e for es in result["shared"].values() for e in es]
    assert not any(e.endswith("P06.nii.gz") for e in every)


def test_a_patient_on_one_side_is_reported_and_nothing_of_it_is_sent():
    result = batching.pairs_by_name({"t1": _cohort(), "t2": _cohort()}, _match)
    assert result["unpaired"] == {"t1": ["P04"], "t2": ["P05"]}
    assert result["shared"] == {}


def test_entries_two_patients_share_keep_them_together():
    names = ["visit/P1_T1.nii.gz", "visit/P2_T1.nii.gz"]
    other = ["visit/P1_T2.nii.gz", "visit/P2_T2.nii.gz"]
    result = batching.pairs_by_name({"t1": names, "t2": other},
                                    lambda roots: _match(roots))
    assert len(result["groups"]) == 1
    assert sorted(result["groups"][0]["keys"]) == ["visit/P1", "visit/P2"]


def test_an_unrelated_entry_travels_with_every_batch():
    names = ["P1_T1.nii.gz", "notes.txt"]
    result = batching.pairs_by_name({"t1": names, "t2": ["P1_T2.nii.gz"]}, _match)
    assert result["shared"] == {"t1": ["notes.txt"]}


def test_a_name_that_escapes_the_tree_is_ignored():
    result = batching.pairs_by_name({"t1": ["../../etc/passwd", "P1_T1.nii.gz"],
                                     "t2": ["P1_T2.nii.gz"]}, _match)
    assert [g["key"] for g in result["groups"]] == ["P1"]
