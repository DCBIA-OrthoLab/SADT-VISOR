"""End-to-end tests for SurgMovPred.

The models are built here from scikit-learn rather than committed: the shipped
packages are 1.4 GB and the reference input is patient data, so neither can go
into git. `test_real_models_*` runs against the real ones when
`SADT_SURGMOVPRED_MODELS` and `SADT_SURGMOVPRED_INPUT` point at them, and is
skipped otherwise -- see tests/data/README.md.
"""

import logging
import os
from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from sadt_surgmovpred import run
from sadt_surgmovpred.pipeline import (
    clean_name,
    find_id_column,
    load_measurements,
    load_model_packages,
    predict_all_targets,
    save_results,
)


def make_model(folder: Path, target_name: str, feature: str = "f1"):
    """A one-feature stacking package with the shape the real ones have."""
    # Fitted on a named frame, like the real packages: a scaler fitted on a bare
    # array warns on every transform and buries anything else in the log.
    training = pd.DataFrame({feature: [0, 10, 20]})
    scaler = StandardScaler().fit(training)
    scaled = pd.DataFrame(scaler.transform(training), columns=training.columns)
    model = LinearRegression().fit(scaled, [5, 15, 25])
    package = {
        "target_name": target_name,
        "features_names": [feature],
        "scaler": scaler,
        "model": model,
    }
    target_dir = folder / target_name
    target_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(package, target_dir / "stacking_package.pkl")
    return folder


@pytest.fixture
def model(tmp_path):
    return make_model(tmp_path / "models", "f1_Pred")


@pytest.fixture
def measurements(tmp_path):
    table = tmp_path / "patients.xlsx"
    pd.DataFrame({"PatientID": [1, 2, 3], "f1": [0, 10, 20]}).to_excel(table, index=False)
    return table


# --- run() -----------------------------------------------------------------


def test_run_writes_both_tables(model, measurements, tmp_path):
    outputs = run(measurements=measurements, model=model, output_dir=tmp_path / "out")

    assert set(outputs) == {"excel", "csv"}
    assert all(path.exists() and path.stat().st_size > 0 for path in outputs.values())

    results = pd.read_excel(outputs["excel"], index_col=0)
    assert list(results.columns) == ["IDPatient", "f1_Pred"]
    assert list(results["IDPatient"]) == [1, 2, 3]
    # The fitted line maps 0/10/20 onto 5/15/25.
    assert results["f1_Pred"].round(6).tolist() == [5.0, 15.0, 25.0]

    csv_results = pd.read_csv(outputs["csv"], index_col=0)
    assert csv_results.shape == results.shape


def test_run_accepts_a_folder_of_tables(model, tmp_path):
    """The batch form: one call for a folder, not one call per file."""
    folder = tmp_path / "batch"
    folder.mkdir()
    pd.DataFrame({"PatientID": [1, 2], "f1": [0, 10]}).to_csv(folder / "a.csv", index=False)
    pd.DataFrame({"PatientID": [3], "f1": [20]}).to_csv(folder / "b.csv", index=False)

    outputs = run(measurements=folder, model=model, output_dir=tmp_path / "out")

    results = pd.read_excel(outputs["excel"], index_col=0)
    assert list(results["IDPatient"]) == [1, 2, 3], "sorted, so a batch is reproducible"


def test_run_writes_nothing_outside_the_output_directory(model, measurements, tmp_path):
    before = {path: path.stat().st_mtime_ns for path in tmp_path.rglob("*") if path.is_file()}

    run(measurements=measurements, model=model, output_dir=tmp_path / "out")

    untouched = {
        path: mtime for path, mtime in before.items() if not path.is_relative_to(tmp_path / "out")
    }
    assert all(path.stat().st_mtime_ns == mtime for path, mtime in untouched.items())
    assert {path.name for path in (tmp_path / "out").iterdir()} == {
        "predictions_outputs.xlsx",
        "predictions_outputs.csv",
    }


def test_run_without_an_id_column_leaves_the_identifier_blank(model, tmp_path):
    table = tmp_path / "anonymous.csv"
    pd.DataFrame({"f1": [0, 10]}).to_csv(table, index=False)

    outputs = run(measurements=table, model=model, output_dir=tmp_path / "out")

    results = pd.read_csv(outputs["csv"], index_col=0)
    assert results["IDPatient"].isna().all()
    assert results["f1_Pred"].notna().all(), "predictions still happen without an ID"


def test_a_model_whose_features_are_absent_is_skipped_not_fatal(tmp_path, measurements):
    """A thinner input table yields fewer targets, never an error."""
    models = make_model(tmp_path / "models", "f1_Pred")
    make_model(models, "other_Pred", feature="not_in_the_input")

    outputs = run(measurements=measurements, model=models, output_dir=tmp_path / "out")

    results = pd.read_csv(outputs["csv"], index_col=0)
    assert "f1_Pred" in results.columns
    assert "other_Pred" not in results.columns


def test_missing_inputs_are_reported_clearly(model, tmp_path, measurements):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        run(measurements=tmp_path / "nope.csv", model=model, output_dir=tmp_path / "out")

    empty = tmp_path / "empty_models"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="stacking_package.pkl"):
        run(measurements=measurements, model=empty, output_dir=tmp_path / "out")

    with pytest.raises(FileNotFoundError, match="not a folder"):
        run(measurements=measurements, model=measurements, output_dir=tmp_path / "out")


# --- the ported pieces, unchanged from upstream ----------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Age (years)", "Age_years"),
        ("SNA total", "SNA_Total"),
        ("  spaced  name  ", "spaced_name"),
        ('weird"name[1]', "weirdname1"),
        ("1stMeasure", "f_1stMeasure"),
        ("", "f_unnamed"),
        ("Jarabak's Ratio", "Jarabak's_Ratio"),
    ],
)
def test_clean_name(raw, expected):
    assert clean_name(raw) == expected


@pytest.mark.parametrize(
    "columns, expected",
    [
        (["PatientID", "Age"], "PatientID"),
        (["Patient ID", "Age"], "Patient ID"),
        (["#", "Age"], "#"),
        (["Subject_ID", "Age"], "Subject_ID"),
        (["Patient Number", "Age"], "Patient Number"),
        (["Age", "Weight"], None),
        (["patient_id_extra_stuff"], "patient_id_extra_stuff"),  # fallback match
    ],
)
def test_find_id_column(columns, expected):
    assert find_id_column(columns) == expected


def test_features_are_matched_without_their_T0_suffix():
    """Some files carry T0 measurements without the '_T0' training-time suffix."""
    training = pd.DataFrame({"f1_T0": [0, 10, 20]})
    scaler = StandardScaler().fit(training)
    scaled = pd.DataFrame(scaler.transform(training), columns=training.columns)
    model = LinearRegression().fit(scaled, [0, 100, 200])
    packages = {"target_a": {"features_names": ["f1_T0"], "scaler": scaler, "model": model}}

    results = predict_all_targets(pd.DataFrame({"f1": [0, 10, 20]}), packages)

    assert results["target_a"].round(6).tolist() == [0.0, 100.0, 200.0]


def test_load_model_packages_walks_nested_folders(tmp_path):
    make_model(tmp_path / "models" / "nested" / "deeper", "a_Pred")
    make_model(tmp_path / "models", "b_Pred")

    packages = load_model_packages(tmp_path / "models")

    assert sorted(packages) == ["a_Pred", "b_Pred"]


def test_load_measurements_reads_every_supported_format(tmp_path):
    frame = pd.DataFrame({"PatientID": [1, 2], "f1": [10, 20]})
    frame.to_csv(tmp_path / "a.csv", index=False)
    frame.to_excel(tmp_path / "a.xlsx", index=False)
    frame.to_excel(tmp_path / "a.ods", engine="odf", index=False)

    for suffix in ("csv", "xlsx", "ods"):
        loaded = load_measurements(tmp_path / f"a.{suffix}")
        assert list(loaded.columns) == ["PatientID", "f1"], suffix
        assert len(loaded) == 2, suffix


def test_a_batch_folder_is_logged_by_count_and_never_by_its_path(tmp_path, caplog):
    """The input folder arrives named by the caller and unpacked under that
    name, so its path can carry a patient's.

    A tool's stderr is captured to a file in the job directory, and on a FAILED
    run the server copies its tail into its own persistent log -- so a path
    written here outlives the run and its job directory. The count is what this
    line was for.
    """
    batch = tmp_path / "Smith_John_2019_measurements"
    batch.mkdir()
    frame = pd.DataFrame({"PatientID": [1, 2], "f1": [10, 20]})
    frame.to_csv(batch / "a.csv", index=False)
    frame.to_csv(batch / "b.csv", index=False)

    with caplog.at_level(logging.INFO, logger="sadt_surgmovpred.pipeline"):
        loaded = load_measurements(batch)

    assert len(loaded) == 4
    messages = [record.getMessage() for record in caplog.records]
    assert messages == ["Loading 2 table(s) from the input folder"], messages


def test_save_results_writes_excel_and_csv(tmp_path):
    outputs = save_results(pd.DataFrame({"IDPatient": [1], "target_a": [10.0]}), tmp_path / "out")

    assert outputs["excel"].suffix == ".xlsx" and outputs["excel"].exists()
    assert outputs["csv"].suffix == ".csv" and outputs["csv"].exists()


# --- the real models -------------------------------------------------------

REAL_MODELS = os.environ.get("SADT_SURGMOVPRED_MODELS")
REAL_INPUT = os.environ.get("SADT_SURGMOVPRED_INPUT")


@pytest.mark.models
@pytest.mark.skipif(
    not (REAL_MODELS and REAL_INPUT),
    reason="set SADT_SURGMOVPRED_MODELS and SADT_SURGMOVPRED_INPUT (see tests/data/README.md)",
)
def test_real_models_predict_every_target(tmp_path):
    """The shipped packages, on the reference input, in one call.

    Numbers are checked against the pre-port implementation separately (see
    README, "Validated against"); what this asserts is that nothing about the
    real packages -- LightGBM sub-estimators, 145-feature models, the '#' ID
    column -- trips over the repackaging.
    """
    import numpy as np

    outputs = run(
        measurements=Path(REAL_INPUT),
        model=Path(REAL_MODELS),
        output_dir=tmp_path / "out",
    )

    results = pd.read_csv(outputs["csv"], index_col=0)
    predictions = results.drop(columns="IDPatient")

    assert len(results) == 101
    assert len(predictions.columns) == 112, "every shipped model produced a column"
    assert results["IDPatient"].notna().all()
    assert np.isfinite(predictions.to_numpy(dtype=float)).all()


# --- the models, when nobody names a folder ---------------------------------


def test_the_installed_models_are_used_when_none_are_named(measurements, tmp_path):
    """A panel with no model field sends none, and the tool finds its own.

    The packages are filed one level down in the published bundle
    (`models/all_models/<target>/`), which costs nothing: `load_model_packages`
    globs for the package file underneath whatever it is handed.
    """
    make_model(tmp_path / "data" / "Surg_Mov_Pred" / "models" / "all_models", "f1_Pred")

    outputs = run(measurements=measurements, output_dir=tmp_path / "out",
                  data_root=tmp_path / "data")

    assert Path(outputs["csv"]).is_file()
    assert "f1_Pred" in pd.read_csv(outputs["csv"]).columns


def test_a_named_model_folder_still_wins(model, measurements, tmp_path):
    """Naming one is pinning which models ran, and is obeyed."""
    make_model(tmp_path / "data" / "Surg_Mov_Pred" / "models", "other_Pred")

    outputs = run(measurements=measurements, model=model,
                  output_dir=tmp_path / "out", data_root=tmp_path / "data")

    columns = pd.read_csv(outputs["csv"]).columns
    assert "f1_Pred" in columns and "other_Pred" not in columns


def test_no_model_and_no_data_root_says_which_of_the_two_is_missing(measurements, tmp_path):
    with pytest.raises(ValueError, match="no data root"):
        run(measurements=measurements, output_dir=tmp_path / "out")


# --- progress ---------------------------------------------------------------


def test_progress_records_are_written_when_the_server_names_a_file(
    tmp_path, monkeypatch
):
    """Every stage reports itself, in order, and the bar never goes back.

    Each stage owns a slice of the run, so a later stage's "1 of n" starts
    where the previous one ended instead of at zero.
    """
    import json

    models = tmp_path / "models"
    make_model(models, "a_Pred")
    make_model(models, "b_Pred")
    batch = tmp_path / "batch"
    batch.mkdir()
    pd.DataFrame({"PatientID": [1], "f1": [10]}).to_csv(batch / "a.csv", index=False)
    pd.DataFrame({"PatientID": [2], "f1": [20]}).to_csv(batch / "b.csv", index=False)

    progress_file = tmp_path / "progress.jsonl"
    monkeypatch.setenv("SADT_PROGRESS_FILE", str(progress_file))

    run(measurements=batch, model=models, output_dir=tmp_path / "out")

    records = [json.loads(line) for line in progress_file.read_text().splitlines()]
    messages = [record["message"] for record in records]
    assert messages == [
        "reading table 1 of 2",
        "reading table 2 of 2",
        "loading model packages",
        "package 1 of 2",
        "package 2 of 2",
        "predicting target 1 of 2",
        "predicting target 2 of 2",
        "writing results",
    ]
    fractions = [record["fraction"] for record in records]
    assert fractions == sorted(fractions)
    assert all(0.0 <= fraction < 1.0 for fraction in fractions)
    # Position, never a name: nothing the caller called its files reaches it.
    assert "a.csv" not in progress_file.read_text()
