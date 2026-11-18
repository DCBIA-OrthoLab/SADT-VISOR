"""The path a number actually takes, from a column to a predicted value.

The port's headline claim is that its predictions are bit-identical to the
pre-port implementation's -- 0.000e+00 across 11 312 values. Nothing in a CI
run can compare against that reference (1.4 GB of models, and patient data),
so what is pinned here is what the claim RESTS on: the frame handed to each
model is the scaler's output, over that model's own features, in that model's
own order, on that model's own index. If any of those four moved, the numbers
would move with them.
"""

import numpy as np
import pandas as pd
import pytest

from sadt_surgmovpred import run
from sadt_surgmovpred.pipeline import clean_name, predict_all_targets, silence_sklearn_version_warning

from conftest import RecordingModel, RecordingScaler, build_package, write_package


def recording_package(features):
    scaler = RecordingScaler()
    model = RecordingModel()
    return scaler, model, {"features_names": list(features), "scaler": scaler, "model": model}


# ---------------------------------------------------------------------------
# What the model is handed
# ---------------------------------------------------------------------------

def test_the_model_receives_the_scalers_output_not_the_raw_column():
    """Skipping the scaler would still produce numbers -- wrong ones."""
    package = build_package(["f1"])
    seen = []
    real_transform = package["scaler"].transform
    package["scaler"] = type(
        "Spy", (), {"transform": lambda self, frame: seen.append(frame) or real_transform(frame)}
    )()
    package["model"] = RecordingModel()

    predict_all_targets(pd.DataFrame({"f1": [0, 10, 20]}), {"t": package})

    handed = package["model"].seen[0]
    assert seen, "the scaler was called"
    assert handed["f1"].tolist() != [0, 10, 20], "the raw column never reaches the model"
    assert handed["f1"].round(6).tolist() == [-1.224745, 0.0, 1.224745]


def test_the_frame_is_ordered_by_the_models_features_not_the_tables_columns():
    """A model trained on [b, a] must be handed [b, a]. Column order is the
    silent way to get plausible numbers that are wrong."""
    scaler, model, package = recording_package(["b", "a"])

    predict_all_targets(pd.DataFrame({"a": [1.0, 2.0], "b": [10.0, 20.0]}), {"t": package})

    assert list(scaler.seen[0].columns) == ["b", "a"]
    assert scaler.seen[0]["b"].tolist() == [10.0, 20.0]
    assert list(model.seen[0].columns) == ["b", "a"]


def test_the_frame_carries_the_feature_names_the_model_was_trained_with():
    """`X_target.columns = expected_features` is what renames a `_T0`-less
    column back to what the model expects -- an unnamed array would make
    sklearn's own feature-name check fire on every target."""
    scaler, model, package = recording_package(["f1_T0"])

    predict_all_targets(pd.DataFrame({"f1": [0.0, 10.0]}), {"t": package})

    assert list(model.seen[0].columns) == ["f1_T0"]


def test_the_input_index_survives_into_the_frame_and_the_result():
    """The result is joined back to the patient identifiers by position; a
    reindex here would shift every row against its patient."""
    scaler, model, package = recording_package(["f1"])
    frame = pd.DataFrame({"f1": [0.0, 10.0, 20.0]}, index=[10, 11, 12])

    results = predict_all_targets(frame, {"t": package})

    assert list(model.seen[0].index) == [10, 11, 12]
    assert list(results.index) == [10, 11, 12]


def test_extra_columns_in_the_table_never_reach_the_model():
    """A clinic's spreadsheet carries far more than any one model wants."""
    scaler, model, package = recording_package(["f1"])

    predict_all_targets(
        pd.DataFrame({"f1": [0.0, 10.0], "unused": [99.0, 98.0], "PatientID": [1, 2]}),
        {"t": package},
    )

    assert list(model.seen[0].columns) == ["f1"]


def test_the_column_names_are_cleaned_before_they_are_matched():
    """`clean_name` is the join between a clinician's header and a training-time
    feature name; without it every model would be skipped."""
    scaler, model, package = recording_package(["SNA_Total"])

    results = predict_all_targets(pd.DataFrame({"SNA total": [1.0, 2.0]}), {"t": package})

    assert "t" in results.columns
    assert model.seen[0]["SNA_Total"].tolist() == [1.0, 2.0]


def test_the_t0_fallback_only_applies_to_the_suffix_it_names():
    """`f1_T0` may be satisfied by `f1`; `f1_T1` may not be satisfied by `f1`,
    and a model wanting it is skipped rather than fed the wrong timepoint."""
    scaler, model, package = recording_package(["f1_T1"])

    with pytest.raises(ValueError, match="produced a prediction"):
        predict_all_targets(pd.DataFrame({"f1": [0.0, 10.0]}), {"t": package})

    assert model.seen == []


def test_an_exact_match_wins_over_the_t0_fallback():
    """A table carrying both `f1` and `f1_T0` must use `f1_T0`."""
    scaler, model, package = recording_package(["f1_T0"])

    predict_all_targets(
        pd.DataFrame({"f1": [1.0, 2.0], "f1_T0": [100.0, 200.0]}), {"t": package}
    )

    assert model.seen[0]["f1_T0"].tolist() == [100.0, 200.0]


# ---------------------------------------------------------------------------
# Reproducibility -- what "bit-identical" is checked against locally
# ---------------------------------------------------------------------------

def test_two_runs_over_the_same_inputs_write_byte_identical_csv(
    tmp_path, model_folder, patients_table
):
    """The reference comparison was exact, so any difference is a regression
    rather than noise. Here that is asserted where it can be: the same inputs
    twice must give the same bytes."""
    first = run(measurements=patients_table, model=model_folder, output_dir=tmp_path / "a")
    second = run(measurements=patients_table, model=model_folder, output_dir=tmp_path / "b")

    assert first["csv"].read_bytes() == second["csv"].read_bytes()


def test_the_column_order_is_the_sorted_model_order_on_every_run(tmp_path, patients_table):
    """Upstream used glob's filesystem order. Sorted loading is what makes the
    output columns reproducible between two machines."""
    models = tmp_path / "models"
    for subdir, target in (("z_dir", "z_Pred"), ("a_dir", "a_Pred"), ("m_dir", "m_Pred")):
        write_package(models, target, subdir=subdir)

    outputs = run(measurements=patients_table, model=models, output_dir=tmp_path / "out")

    results = pd.read_csv(outputs["csv"], index_col=0)
    assert list(results.columns) == ["IDPatient", "a_Pred", "m_Pred", "z_Pred"]


def test_the_predicted_values_do_not_depend_on_where_the_columns_sit(
    tmp_path, model_folder
):
    """Two tables with the same measurements in a different column order are
    the same patients, and must predict the same numbers."""
    straight = tmp_path / "straight.csv"
    shuffled = tmp_path / "shuffled.csv"
    pd.DataFrame({"PatientID": [1, 2], "f1": [0, 10], "other": [7, 8]}).to_csv(
        straight, index=False
    )
    pd.DataFrame({"other": [7, 8], "f1": [0, 10], "PatientID": [1, 2]}).to_csv(
        shuffled, index=False
    )

    a = run(measurements=straight, model=model_folder, output_dir=tmp_path / "a")
    b = run(measurements=shuffled, model=model_folder, output_dir=tmp_path / "b")

    assert (
        pd.read_csv(a["csv"], index_col=0)["f1_Pred"].tolist()
        == pd.read_csv(b["csv"], index_col=0)["f1_Pred"].tolist()
    )


# ---------------------------------------------------------------------------
# A target that cannot be predicted
# ---------------------------------------------------------------------------

def test_a_target_that_raises_costs_its_column_and_nothing_else():
    """112 models ship together. One that throws must not take the other 111."""
    class Exploding:
        def predict(self, frame):
            raise ZeroDivisionError("boom")

    good = build_package(["f1"])
    bad = build_package(["f1"])
    bad["model"] = Exploding()

    results = predict_all_targets(
        pd.DataFrame({"f1": [0.0, 10.0]}), {"bad": bad, "good": good}
    )

    assert list(results.columns) == ["good"]


def test_a_nan_in_a_feature_skips_the_target_rather_than_predicting_from_it():
    """sklearn refuses a NaN, and a silently imputed one would be a number a
    clinician cannot tell from a measured one."""
    package = build_package(["f1"])

    with pytest.raises(RuntimeError, match="produced a prediction"):
        predict_all_targets(pd.DataFrame({"f1": [0.0, np.nan]}), {"t": package})


def test_a_text_value_in_a_numeric_column_skips_the_target():
    """A merged cell or an 'n/a' typed into a spreadsheet."""
    package = build_package(["f1"])

    with pytest.raises(RuntimeError, match="produced a prediction"):
        predict_all_targets(pd.DataFrame({"f1": ["n/a", "n/a"]}), {"t": package})


def test_no_rows_at_all_skips_every_target():
    package = build_package(["f1"])

    with pytest.raises(RuntimeError, match="produced a prediction"):
        predict_all_targets(pd.DataFrame({"f1": pd.Series([], dtype=float)}), {"t": package})


def test_predicting_nothing_at_all_names_how_many_models_were_loaded():
    """Loading every model and predicting nothing used to write both result
    files and report success. The guard counts what was produced, and its
    message has to say how far the run got."""
    packages = {name: build_package(["absent"]) for name in ("a", "b", "c")}

    with pytest.raises(ValueError) as failure:
        predict_all_targets(pd.DataFrame({"f1": [0.0]}), packages)

    assert "3 loaded model(s)" in str(failure.value)


def test_every_target_skipped_for_missing_features_is_the_callers_input_error():
    """Nothing errored: the table simply lacks what the models were trained on.
    That is the caller's to fix, so it is a ValueError (a 422) that names the
    argument and the columns most often missing."""
    packages = {name: build_package(["absent", "f1"]) for name in ("a", "b")}

    with pytest.raises(ValueError) as failure:
        predict_all_targets(pd.DataFrame({"f1": [0.0]}), packages)

    message = str(failure.value)
    assert "2 skipped for missing input features, 0 errored" in message
    assert "'measurements'" in message
    assert "absent (2 of 2)" in message


def test_predicting_nothing_counts_the_errored_apart_and_quotes_the_first_error():
    """The old message blamed missing features even when every target had
    raised. An errored target is counted on its own, and the first error is
    what an operator reads first."""
    class Exploding:
        def predict(self, frame):
            raise ZeroDivisionError("boom")

    bad = build_package(["f1"])
    bad["model"] = Exploding()
    packages = {"bad": bad, "thin": build_package(["absent"])}

    with pytest.raises(RuntimeError) as failure:
        predict_all_targets(pd.DataFrame({"f1": [0.0, 10.0]}), packages)

    message = str(failure.value)
    assert "1 skipped for missing input features, 1 errored" in message
    assert "first error: ZeroDivisionError: boom" in message


def test_a_failed_target_is_logged_by_position_class_and_message(caplog):
    class Exploding:
        def predict(self, frame):
            raise ZeroDivisionError("boom")

    good = build_package(["f1"])
    bad = build_package(["f1"])
    bad["model"] = Exploding()

    predict_all_targets(pd.DataFrame({"f1": [0.0, 10.0]}), {"bad": bad, "good": good})

    assert "target 1 of 2 ('bad'): prediction failed (ZeroDivisionError: boom)" in caplog.text
    assert "1 of 2 target(s) predicted" in caplog.text


def test_skipped_targets_are_one_aggregated_warning_not_one_each(caplog):
    """112 targets over a thin table used to be up to 112 warnings."""
    import logging

    packages = {f"t{index}_Pred": build_package(["absent"]) for index in range(10)}
    packages["ok_Pred"] = build_package(["f1"])

    with caplog.at_level(logging.INFO, logger="sadt_surgmovpred.pipeline"):
        predict_all_targets(pd.DataFrame({"f1": [0.0, 10.0]}), packages)

    skips = [r for r in caplog.records if "skipped for missing input features" in r.getMessage()
             and r.levelno == logging.WARNING]
    assert len(skips) == 1
    assert "10 of 11 target(s) skipped" in skips[0].getMessage()
    assert "absent (10 of 11)" in skips[0].getMessage()


def test_a_skipped_target_says_which_features_were_missing(caplog):
    """The warning is the only place a caller learns why a column is absent."""
    package = build_package(["a_T0", "b_T0", "c_T0"])

    with pytest.raises(ValueError):
        predict_all_targets(pd.DataFrame({"unrelated": [0.0]}), {"SNA_Pred": package})

    assert "SNA_Pred" in caplog.text
    assert "a_T0" in caplog.text


def test_two_headers_that_clean_to_one_name_skip_the_target_rather_than_guess():
    """'SNA total' and 'SNA_Total' both clean to `SNA_Total`. Feeding either
    one to a model that asked for it would be a coin toss between two
    measurements, so the target is skipped instead."""
    package = build_package(["SNA_Total"])
    frame = pd.DataFrame({"SNA total": [1.0, 2.0], "SNA_Total": [9.0, 8.0]})

    with pytest.raises(RuntimeError, match="produced a prediction"):
        predict_all_targets(frame, {"t": package})


# ---------------------------------------------------------------------------
# clean_name
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("SM_A'_CP", "SM_A'_CP"),
        ("a\tb", "a_b"),
        ("a\r\nb", "a_b"),
        ("a---b", "a_b"),
        ("__leading__", "leading"),
        ("total", "Total"),
        ("subtotal", "subTotal"),
        ("2026", "f_2026"),
        ("!!!", "f_unnamed"),
        ("   ", "f_unnamed"),
        (42, "f_42"),
        (3.5, "f_3_5"),
    ],
)
def test_clean_name_beyond_the_upstream_cases(raw, expected):
    """Every branch of the ported function, including the ones no upstream
    header exercised -- the apostrophe kept for Jarabak's, the digit prefix,
    and the empty result."""
    assert clean_name(raw) == expected


def test_clean_name_refuses_to_invent_a_name_it_could_not_derive():
    """Two columns that both failed became the same `f_unnamed`, so one
    silently shadowed the other and feature resolution matched the wrong
    measurement -- a wrong prediction rather than a missing one."""
    class Unnameable:
        def __str__(self):
            raise TypeError("no name")

    with pytest.raises(ValueError, match="Could not derive a feature name"):
        clean_name(Unnameable())


def test_clean_name_is_idempotent():
    """It runs once per run today; a second pass must not rename a feature."""
    for raw in ("Age (years)", "SNA total", "1stMeasure", "Jarabak's Ratio"):
        once = clean_name(raw)
        assert clean_name(once) == once, raw


# ---------------------------------------------------------------------------
# The sklearn version warning
# ---------------------------------------------------------------------------

def test_the_version_warning_is_silenced_because_compatibility_is_checked_first():
    """The shipped packages were pickled under scikit-learn 1.6.1 and load
    under 1.7.2. That mismatch is upstream's design, not a failure -- but the
    warning would fire once per package and bury everything else."""
    import warnings

    from sklearn.exceptions import InconsistentVersionWarning

    stale = InconsistentVersionWarning(
        estimator_name="StackingRegressor",
        current_sklearn_version="1.7.2",
        original_sklearn_version="1.6.1",
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        silence_sklearn_version_warning()
        warnings.warn(stale)

    assert caught == []


def test_silencing_the_warning_is_not_a_blanket_filter():
    """It names `InconsistentVersionWarning` and nothing else -- a bare
    `simplefilter("ignore")` here would hide a deprecation the pins depend on."""
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        silence_sklearn_version_warning()
        warnings.warn("something else entirely", UserWarning)

    assert [str(entry.message) for entry in caught] == ["something else entirely"]


def test_the_pipeline_imports_nothing_heavy_at_module_level():
    """`scripts/describe.py` imports this package on every CI run to publish
    the schema. sklearn, pandas and joblib cost seconds, so every one of them
    is imported inside the function that needs it."""
    import ast

    from sadt_surgmovpred import pipeline

    tree = ast.parse(open(pipeline.__file__, encoding="utf-8").read())
    top_level = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level.update(alias.name.split(".")[0] for alias in node.names)
        # A relative import is this package's own stdlib-only modules
        # (progress, errors), which cost nothing to load.
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            top_level.add(node.module.split(".")[0])

    assert top_level == {"logging", "re", "collections", "pathlib"}
