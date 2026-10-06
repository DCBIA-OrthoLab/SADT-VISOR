"""What a caller gets back, and what it must never contain.

The two tables are the whole product: `predictions_outputs.xlsx` and
`predictions_outputs.csv`, one row per patient, `IDPatient` first and one
column per predicted measurement. Everything the server hands to a client is
built from this directory, so a server-side path appearing in it would travel.
"""

import pandas as pd
import pytest

from sadt_surgmovpred import run
from sadt_surgmovpred.pipeline import find_id_column, save_results

from conftest import write_package


# ---------------------------------------------------------------------------
# The output directory
# ---------------------------------------------------------------------------

def test_the_output_directory_is_created_when_it_does_not_exist(
    tmp_path, model_folder, patients_table
):
    destination = tmp_path / "does" / "not" / "exist"
    assert not destination.exists()

    run(measurements=patients_table, model=model_folder, output_dir=destination)

    assert destination.is_dir()


def test_an_existing_output_directory_keeps_what_was_already_in_it(
    tmp_path, model_folder, patients_table
):
    """The server hands over a job directory that may already hold inputs; the
    tool adds to it rather than clearing it."""
    destination = tmp_path / "out"
    destination.mkdir()
    (destination / "keep.txt").write_text("previous")

    run(measurements=patients_table, model=model_folder, output_dir=destination)

    assert (destination / "keep.txt").read_text() == "previous"


def test_exactly_two_files_are_written_and_they_are_the_named_ones(
    tmp_path, model_folder, patients_table
):
    """No scratch directory, no leftover pickle, no log file: the port removed
    the temporary-folder machinery the server now owns."""
    destination = tmp_path / "out"

    run(measurements=patients_table, model=model_folder, output_dir=destination)

    assert sorted(path.name for path in destination.rglob("*")) == [
        "predictions_outputs.csv",
        "predictions_outputs.xlsx",
    ]


def test_run_returns_paths_that_exist_under_the_directory_it_was_given(
    tmp_path, model_folder, patients_table
):
    outputs = run(measurements=patients_table, model=model_folder, output_dir=tmp_path / "out")

    assert set(outputs) == {"excel", "csv"}
    for path in outputs.values():
        assert path.is_file()
        assert path.parent == tmp_path / "out"


def test_run_accepts_plain_strings_for_its_paths(tmp_path, model_folder, patients_table):
    """`run()` normalises with `Path(...)`, so a runner passing strings from
    `job.json` works without every caller converting first."""
    outputs = run(
        measurements=str(patients_table),
        model=str(model_folder),
        output_dir=str(tmp_path / "out"),
    )

    assert outputs["csv"].is_file()


def test_save_results_creates_the_directories_above_it(tmp_path):
    outputs = save_results(
        pd.DataFrame({"IDPatient": [1], "t": [0.5]}), tmp_path / "a" / "b" / "c"
    )

    assert outputs["excel"].is_file() and outputs["csv"].is_file()


def test_a_failed_run_writes_no_partial_result(tmp_path, model_folder):
    """A table missing every feature must not leave a file that reads like a
    successful run of zero predictions."""
    table = tmp_path / "wrong.csv"
    pd.DataFrame({"PatientID": [1], "unrelated": [0]}).to_csv(table, index=False)

    with pytest.raises(ValueError):
        run(measurements=table, model=model_folder, output_dir=tmp_path / "out")

    assert not (tmp_path / "out").exists()


# ---------------------------------------------------------------------------
# What is in the tables
# ---------------------------------------------------------------------------

def test_both_tables_carry_the_same_values(tmp_path, model_folder, patients_table):
    """Upstream wrote both and returned only the Excel one, so the CSV never
    reached the client. They are returned together, so they have to agree."""
    outputs = run(measurements=patients_table, model=model_folder, output_dir=tmp_path / "out")

    excel = pd.read_excel(outputs["excel"], index_col=0)
    csv = pd.read_csv(outputs["csv"], index_col=0)

    pd.testing.assert_frame_equal(excel, csv, check_dtype=False)


def test_the_identifier_is_the_first_column(tmp_path, model_folder, patients_table):
    """`insert(0, 'IDPatient', ...)`: a table of unlabelled prediction rows is
    unusable, and a reader takes the first column as the key."""
    outputs = run(measurements=patients_table, model=model_folder, output_dir=tmp_path / "out")

    assert list(pd.read_csv(outputs["csv"], index_col=0).columns)[0] == "IDPatient"


def test_the_identifiers_keep_the_order_of_the_input_rows(tmp_path, model_folder):
    """Predictions are joined to patients by position. A reordering here
    attaches every prediction to the wrong person, silently."""
    table = tmp_path / "patients.csv"
    pd.DataFrame({"PatientID": [30, 10, 20], "f1": [20, 0, 10]}).to_csv(table, index=False)

    outputs = run(measurements=table, model=model_folder, output_dir=tmp_path / "out")

    results = pd.read_csv(outputs["csv"], index_col=0)
    assert results["IDPatient"].tolist() == [30, 10, 20]
    assert results["f1_Pred"].round(6).tolist() == [25.0, 5.0, 15.0]


def test_non_numeric_identifiers_survive_unchanged(tmp_path, model_folder):
    """Study identifiers are routinely strings -- `P-001`, `T1_07`."""
    table = tmp_path / "patients.csv"
    pd.DataFrame({"PatientID": ["P-001", "P-002"], "f1": [0, 10]}).to_csv(table, index=False)

    outputs = run(measurements=table, model=model_folder, output_dir=tmp_path / "out")

    assert pd.read_csv(outputs["csv"], index_col=0)["IDPatient"].tolist() == ["P-001", "P-002"]


def test_a_batch_carries_every_files_identifiers_in_the_sorted_order(tmp_path, model_folder):
    folder = tmp_path / "batch"
    folder.mkdir()
    pd.DataFrame({"PatientID": [10, 11], "f1": [0, 10]}).to_csv(folder / "b.csv", index=False)
    pd.DataFrame({"PatientID": [1], "f1": [20]}).to_csv(folder / "a.csv", index=False)

    outputs = run(measurements=folder, model=model_folder, output_dir=tmp_path / "out")

    assert pd.read_csv(outputs["csv"], index_col=0)["IDPatient"].tolist() == [1, 10, 11]


def test_one_column_per_loaded_model_that_could_predict(tmp_path, patients_table):
    models = tmp_path / "models"
    write_package(models, "a_Pred", features=("f1",))
    write_package(models, "b_Pred", features=("f1",))
    write_package(models, "c_Pred", features=("absent",))

    outputs = run(measurements=patients_table, model=models, output_dir=tmp_path / "out")

    results = pd.read_csv(outputs["csv"], index_col=0)
    assert list(results.columns) == ["IDPatient", "a_Pred", "b_Pred"]


def test_nothing_in_the_result_names_a_path_on_this_machine(
    tmp_path, model_folder, patients_table
):
    """The tables travel to a client. A server-side path in a header or a cell
    would leak the deployment's own layout, and the job id with it."""
    outputs = run(measurements=patients_table, model=model_folder, output_dir=tmp_path / "out")

    text = outputs["csv"].read_text(encoding="utf-8")
    assert str(tmp_path) not in text
    assert "/" not in text
    assert str(tmp_path).encode() not in outputs["excel"].read_bytes()


# ---------------------------------------------------------------------------
# The patient identifier column
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "column",
    ["#", "ID", "id", "IDPatient", "ID_Patient", "PatientID", "Patient ID", "patient-id",
     "Patient Number", "Patient_No", "Patient num", "Subject_ID", "Subject", "Patient"],
)
def test_every_documented_identifier_spelling_is_detected(column):
    """The README promises this list. A spelling that stopped matching would
    blank the identifier column and nothing would fail."""
    assert find_id_column([column, "Age"]) == column


def test_the_first_matching_pattern_wins_over_a_later_one():
    """Two candidate columns in one table: '#' is checked before 'PatientID',
    so the choice is the pattern order rather than the column order."""
    assert find_id_column(["PatientID", "#"]) == "#"


def test_a_column_that_only_looks_like_an_identifier_is_not_one():
    """`Identifier` and `Rapid` contain 'id' as a substring; only a whole-name
    match, or 'patient' and 'id' together, counts."""
    assert find_id_column(["Identifier", "Rapid", "Age"]) is None


def test_the_fallback_needs_both_patient_and_id():
    assert find_id_column(["patient_code"]) is None
    assert find_id_column(["the_patient_id_column"]) == "the_patient_id_column"


def test_a_non_string_column_label_does_not_break_detection():
    """A spreadsheet with a numeric first header row reaches pandas as ints."""
    assert find_id_column([0, 1, "PatientID"]) == "PatientID"


def test_without_an_identifier_the_column_is_present_and_empty(tmp_path, model_folder):
    """Blank, not absent: a client reading the first column must find it."""
    table = tmp_path / "anonymous.csv"
    pd.DataFrame({"f1": [0, 10]}).to_csv(table, index=False)

    outputs = run(measurements=table, model=model_folder, output_dir=tmp_path / "out")

    results = pd.read_csv(outputs["csv"], index_col=0)
    assert "IDPatient" in results.columns
    assert results["IDPatient"].isna().all()
