"""What the tool accepts as input, and what it says when it cannot.

`measurements` is one table or a folder of them; `model` is a folder of model
packages. Everything here runs for real -- the tool is tabular and CPU-only, so
there is nothing to stub.
"""

from pathlib import Path

import pandas as pd
import pytest

from sadt_surgmovpred.pipeline import (
    TABULAR_SUFFIXES,
    load_measurements,
    load_model_packages,
    load_tabular_file,
)

from conftest import write_package


# ---------------------------------------------------------------------------
# One table
# ---------------------------------------------------------------------------

def test_a_single_file_is_read_without_a_folder_around_it(tmp_path):
    """The single-patient case: one table handed straight to `measurements`."""
    table = tmp_path / "one.csv"
    pd.DataFrame({"PatientID": [7], "f1": [3.5]}).to_csv(table, index=False)

    loaded = load_measurements(table)

    assert len(loaded) == 1
    assert loaded["f1"].tolist() == [3.5]


def test_a_one_row_table_is_not_a_special_case(tmp_path, model_folder):
    """A single patient must predict exactly like a hundred of them."""
    from sadt_surgmovpred import run

    table = tmp_path / "one.csv"
    pd.DataFrame({"PatientID": [7], "f1": [10]}).to_csv(table, index=False)

    outputs = run(measurements=table, model=model_folder, output_dir=tmp_path / "out")

    results = pd.read_csv(outputs["csv"], index_col=0)
    assert len(results) == 1
    assert results["f1_Pred"].round(6).tolist() == [15.0]


def test_a_table_with_no_rows_is_reported_rather_than_silently_written(
    tmp_path, model_folder
):
    """Zero patients cannot be scaled, and must not produce an empty result
    file that reads like a successful run of nothing. It is the caller's
    table, so the refusal is an input error naming the argument."""
    from sadt_surgmovpred import run

    table = tmp_path / "empty.csv"
    pd.DataFrame({"PatientID": [], "f1": []}).to_csv(table, index=False)

    with pytest.raises(ValueError, match="'measurements' table has no rows"):
        run(measurements=table, model=model_folder, output_dir=tmp_path / "out")

    assert not (tmp_path / "out").exists(), "nothing is written for a run that failed"


def test_an_unsupported_extension_names_the_extension_and_the_file(tmp_path):
    """The message reaches the caller as a 422, so it has to name what is
    wrong -- not just that something is."""
    bad = tmp_path / "measurements.txt"
    bad.write_text("not a table")

    with pytest.raises(ValueError) as failure:
        load_tabular_file(bad)

    assert "'.txt'" in str(failure.value)
    assert "measurements.txt" in str(failure.value)


def test_a_zip_is_not_extracted_by_the_tool(tmp_path):
    """The server unpacks archives before `run()` is called. A tool that also
    unpacked them would be doing it twice, and the port deleted its half."""
    archive = tmp_path / "batch.zip"
    archive.write_bytes(b"PK\x03\x04not really a zip")

    with pytest.raises(ValueError, match=r"'\.zip'"):
        load_tabular_file(archive)


def test_the_extension_is_matched_case_insensitively(tmp_path):
    """A table exported from Windows arrives as PATIENTS.XLSX often enough."""
    table = tmp_path / "PATIENTS.CSV"
    pd.DataFrame({"PatientID": [1], "f1": [0]}).to_csv(table, index=False)

    assert len(load_tabular_file(table)) == 1


def test_a_compound_extension_is_read_by_its_last_suffix(tmp_path):
    """`patients.2026.01.csv` is a csv; `Path.suffix` is the last one only."""
    table = tmp_path / "patients.2026.01.csv"
    pd.DataFrame({"PatientID": [1], "f1": [0]}).to_csv(table, index=False)

    assert len(load_tabular_file(table)) == 1


def test_a_file_the_reader_cannot_parse_fails_loudly(tmp_path):
    """A .xlsx that is not a zip container at all: pandas raises, and nothing
    here swallows it into an empty table."""
    broken = tmp_path / "patients.xlsx"
    broken.write_bytes(b"this is not an excel file")

    with pytest.raises(Exception):
        load_tabular_file(broken)


def test_a_missing_input_names_the_path_it_looked_for(tmp_path):
    with pytest.raises(FileNotFoundError) as failure:
        load_measurements(tmp_path / "absent.csv")

    assert str(tmp_path / "absent.csv") in str(failure.value)


# ---------------------------------------------------------------------------
# A folder of tables
# ---------------------------------------------------------------------------

def test_a_folder_is_concatenated_in_sorted_order(tmp_path):
    """readdir order varies between filesystems, and an unsorted concatenation
    renumbers every row between two runs on the same folder."""
    folder = tmp_path / "batch"
    folder.mkdir()
    for name, patient in (("z.csv", 3), ("a.csv", 1), ("m.csv", 2)):
        pd.DataFrame({"PatientID": [patient], "f1": [patient]}).to_csv(
            folder / name, index=False
        )

    loaded = load_measurements(folder)

    assert loaded["PatientID"].tolist() == [1, 2, 3]


def test_a_folder_mixing_the_three_formats_is_one_batch(tmp_path):
    folder = tmp_path / "batch"
    folder.mkdir()
    pd.DataFrame({"PatientID": [1], "f1": [1]}).to_csv(folder / "a.csv", index=False)
    pd.DataFrame({"PatientID": [2], "f1": [2]}).to_excel(folder / "b.xlsx", index=False)
    pd.DataFrame({"PatientID": [3], "f1": [3]}).to_excel(
        folder / "c.ods", engine="odf", index=False
    )

    assert load_measurements(folder)["PatientID"].tolist() == [1, 2, 3]


def test_a_folder_ignores_files_that_are_not_tables(tmp_path):
    """A README or a leftover .pkl beside the spreadsheets is not an input."""
    folder = tmp_path / "batch"
    folder.mkdir()
    pd.DataFrame({"PatientID": [1], "f1": [1]}).to_csv(folder / "a.csv", index=False)
    (folder / "README.md").write_text("how these were collected")
    (folder / "notes.pkl").write_bytes(b"\x80\x04")

    assert len(load_measurements(folder)) == 1


def test_a_folder_is_searched_flat_not_recursively(tmp_path):
    """Pinned because the two discoveries in this tool differ: model packages
    are found with `**/`, tables are not. A nested tree therefore yields only
    the tables at its top level -- documented in README, and the reason a batch
    arrives as a flat folder."""
    folder = tmp_path / "batch"
    (folder / "clinic_a").mkdir(parents=True)
    pd.DataFrame({"PatientID": [1], "f1": [1]}).to_csv(folder / "top.csv", index=False)
    pd.DataFrame({"PatientID": [2], "f1": [2]}).to_csv(
        folder / "clinic_a" / "nested.csv", index=False
    )

    assert load_measurements(folder)["PatientID"].tolist() == [1]


def test_a_folder_holding_only_subfolders_is_reported_not_silently_empty(tmp_path):
    """The failure mode the flat search has to make loud: a zip of per-clinic
    folders yields no table at all, and says so."""
    folder = tmp_path / "batch"
    (folder / "clinic_a").mkdir(parents=True)
    pd.DataFrame({"PatientID": [1], "f1": [1]}).to_csv(
        folder / "clinic_a" / "nested.csv", index=False
    )

    with pytest.raises(FileNotFoundError) as failure:
        load_measurements(folder)

    assert str(folder) in str(failure.value)


def test_an_empty_folder_names_itself(tmp_path):
    folder = tmp_path / "batch"
    folder.mkdir()

    with pytest.raises(FileNotFoundError) as failure:
        load_measurements(folder)

    assert "CSV, XLSX or ODS" in str(failure.value)
    assert str(folder) in str(failure.value)


def test_the_supported_suffixes_are_the_ones_the_reader_handles(tmp_path):
    """`TABULAR_SUFFIXES` gates folder discovery and `load_tabular_file` gates
    a single file; the two drifting apart means a file discovered in a batch
    and refused in isolation."""
    for suffix in TABULAR_SUFFIXES:
        table = tmp_path / f"a{suffix}"
        frame = pd.DataFrame({"PatientID": [1], "f1": [0]})
        if suffix == ".csv":
            frame.to_csv(table, index=False)
        elif suffix == ".ods":
            frame.to_excel(table, engine="odf", index=False)
        else:
            frame.to_excel(table, index=False)
        assert len(load_tabular_file(table)) == 1, suffix


# ---------------------------------------------------------------------------
# The model folder
# ---------------------------------------------------------------------------

def test_model_packages_are_loaded_in_sorted_path_order(tmp_path):
    """The change this port made against upstream: `sorted(model.glob(...))`
    instead of glob's filesystem order, so the output columns come out in the
    same order on every run and on every machine."""
    models = tmp_path / "models"
    for subdir, target in (("z_dir", "z_Pred"), ("a_dir", "a_Pred"), ("m_dir", "m_Pred")):
        write_package(models, target, subdir=subdir)

    packages = load_model_packages(models)

    assert list(packages) == ["a_Pred", "m_Pred", "z_Pred"]


def test_the_package_key_is_its_declared_target_not_its_folder(tmp_path):
    """`target_name` inside the pickle names the output column; the folder is
    only where it happens to sit."""
    models = tmp_path / "models"
    write_package(models, "SNA_Pred", subdir="00_first")

    assert list(load_model_packages(models)) == ["SNA_Pred"]


def test_a_model_folder_that_is_a_file_names_itself(tmp_path):
    table = tmp_path / "patients.csv"
    table.write_text("PatientID,f1\n1,0\n")

    with pytest.raises(FileNotFoundError) as failure:
        load_model_packages(table)

    assert str(table) in str(failure.value)


def test_a_model_folder_with_no_packages_names_the_file_it_wanted(tmp_path):
    empty = tmp_path / "models"
    empty.mkdir()

    with pytest.raises(FileNotFoundError) as failure:
        load_model_packages(empty)

    assert "stacking_package.pkl" in str(failure.value)
    assert str(empty) in str(failure.value)


def test_one_unreadable_package_does_not_lose_the_others(tmp_path, caplog):
    """112 packages ship together; one corrupt pickle must cost one column,
    not the run."""
    models = tmp_path / "models"
    write_package(models, "good_Pred")
    (models / "broken").mkdir()
    (models / "broken" / "stacking_package.pkl").write_bytes(b"not a pickle")

    packages = load_model_packages(models)

    assert list(packages) == ["good_Pred"]
    assert "model package 1 of 2 (target 'broken') failed (" in caplog.text
    assert "loaded 1 of 2 model package(s); 1 failed" in caplog.text
    # By its target folder, never by its path.
    assert str(models) not in caplog.text


def test_a_table_of_a_batch_that_cannot_be_read_is_named_by_position(tmp_path):
    folder = tmp_path / "batch"
    folder.mkdir()
    pd.DataFrame({"PatientID": [1], "f1": [1]}).to_csv(folder / "a.csv", index=False)
    (folder / "b.xlsx").write_bytes(b"this is not an excel file")

    with pytest.raises(ValueError) as failure:
        load_measurements(folder)

    message = str(failure.value)
    assert message.startswith("'measurements' table 2 of 2 could not be read (")
    assert "b.xlsx" not in message


def test_a_missing_installed_models_folder_is_the_deployments_fault(tmp_path):
    """The default folder missing is no argument the caller can fix: a 503,
    and a message that says it was the installed folder, not 'model'."""
    from sadt_surgmovpred.errors import ToolUnavailableError

    with pytest.raises(ToolUnavailableError, match="installed models folder"):
        load_model_packages(tmp_path / "absent", installed=True)

    with pytest.raises(FileNotFoundError, match="the 'model' argument"):
        load_model_packages(tmp_path / "absent")


def test_every_package_being_unreadable_is_a_failure_not_an_empty_run(tmp_path):
    models = tmp_path / "models"
    for name in ("a", "b"):
        (models / name).mkdir(parents=True)
        (models / name / "stacking_package.pkl").write_bytes(b"not a pickle")

    with pytest.raises(RuntimeError, match="could be loaded"):
        load_model_packages(models)


def test_a_package_file_by_another_name_is_not_a_model(tmp_path):
    """Only `stacking_package.pkl` is a model package. A scaler someone left
    beside it is not loaded and cannot become a column."""
    models = tmp_path / "models"
    write_package(models, "good_Pred")
    (models / "good_Pred" / "scaler.pkl").write_bytes(b"not a pickle")

    assert list(load_model_packages(models)) == ["good_Pred"]


def test_two_packages_declaring_the_same_target_collapse_to_the_last_one(tmp_path):
    """Pinned because it is silent: the dict is keyed by `target_name`, so a
    duplicated target yields one column, and sorted loading order decides
    which package wrote it."""
    models = tmp_path / "models"
    write_package(models, "same_Pred", subdir="a_first")
    write_package(models, "same_Pred", subdir="z_last")

    packages = load_model_packages(models)

    assert list(packages) == ["same_Pred"]


def test_a_model_path_that_does_not_exist_is_reported_as_not_a_folder(tmp_path):
    with pytest.raises(FileNotFoundError, match="not a folder"):
        load_model_packages(Path(tmp_path / "absent"))
