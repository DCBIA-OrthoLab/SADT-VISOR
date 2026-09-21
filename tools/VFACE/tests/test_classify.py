"""Reading a feature table as a classification.

The models are real LightGBM boosters, trained here on a few rows of synthetic
data. That is not a mock: `classify` selects the columns each model names in
its own `feature_name_`, and the only way to check that seam is to hand it a
model that has one.
"""

import os

import numpy as np
import pytest

from sadt_vface import classify, features
from sadt_vface.errors import ToolInputError


FEATURES = ["CB_ROr_ROr_RL", "MAND_RCo_RCo_IS", "MAX_ANS_ANS_AP"]


def train(path, columns, rule, seed=0):
    """A booster that has actually learned something, saved as the bundle holds it.

    `rule` decides each row's label from its features, so the model can be
    asked a question with a known answer.
    """
    import joblib
    import lightgbm as lgb
    import pandas as pd

    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(rng.normal(size=(200, len(columns))), columns=columns)
    labels = np.array([int(rule(row)) for _index, row in frame.iterrows()])

    booster = lgb.LGBMClassifier(n_estimators=12, num_leaves=4, min_child_samples=5,
                                 verbose=-1, random_state=seed)
    booster.fit(frame, labels)
    os.makedirs(os.path.dirname(str(path)) or ".", exist_ok=True)
    joblib.dump(booster, str(path))
    return booster


@pytest.fixture
def bundle(tmp_path):
    """The three models the classifier loads, by what each answers."""
    folder = tmp_path / "vface_models"
    # Symmetric (1) when the transverse feature is small.
    train(folder / "sym_asymm.txt", FEATURES, lambda row: abs(row[FEATURES[0]]) < 0.5)
    train(folder / "mand_asym.txt", FEATURES, lambda row: row[FEATURES[1]] > 0, seed=1)
    train(folder / "max_asym.txt", FEATURES, lambda row: row[FEATURES[2]] > 0, seed=2)
    return str(folder)


def records(count=6, seed=7):
    rng = np.random.default_rng(seed)
    return [
        dict({"ID": str(index)},
             **{name: float(value) for name, value in zip(FEATURES, rng.normal(size=3))})
        for index in range(count)
    ]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def test_the_three_models_are_loaded_by_what_they_answer(bundle):
    loaded = classify.load_models(bundle)
    assert set(loaded) == {"symmetry", "mandible", "maxilla"}
    assert all(hasattr(model, "feature_name_") for model in loaded.values())


def test_a_bundle_missing_a_model_is_refused_naming_it(bundle, tmp_path):
    """A classification that ran two of three models would answer the
    sub-questions on a symmetry verdict nothing made."""
    os.remove(os.path.join(bundle, "mand_asym.txt"))
    with pytest.raises(ToolInputError, match="mand_asym.txt"):
        classify.load_models(bundle)


def test_a_bundle_that_is_not_a_folder_is_refused(tmp_path):
    with pytest.raises(ToolInputError, match="is not a folder"):
        classify.load_models(str(tmp_path / "nowhere"))


# ---------------------------------------------------------------------------
# Classifying
# ---------------------------------------------------------------------------

def test_every_patient_gets_a_verdict(bundle):
    classified = classify.classify(records(), bundle)
    assert len(classified) == 6
    assert all(row["Asymmetry"] in ("Symmetric", "Asymmetric") for row in classified)
    assert all(row["Mand"] in ("True", "False") for row in classified)


def test_a_symmetric_patient_is_not_asked_the_finer_question(bundle):
    """The sub-models were trained on asymmetric patients alone; asking them
    about a symmetric one is a question outside their domain."""
    symmetric = [dict({"ID": "1"}, **{FEATURES[0]: 0.0, FEATURES[1]: 5.0, FEATURES[2]: 5.0})]
    classified = classify.classify(symmetric, bundle)
    assert classified[0]["Asymmetry"] == "Symmetric"
    assert classified[0]["Mand"] == "False"
    assert classified[0]["Max"] == "False"


def test_the_model_chooses_its_own_columns(bundle):
    """Never the table's order. A template that gained a column would otherwise
    feed the model a different feature in each position."""
    shuffled = [
        {"ID": row["ID"], FEATURES[2]: row[FEATURES[2]],
         "Something_Else": 99.0, FEATURES[0]: row[FEATURES[0]],
         FEATURES[1]: row[FEATURES[1]]}
        for row in records()
    ]
    assert [row["Asymmetry"] for row in classify.classify(shuffled, bundle)] == \
        [row["Asymmetry"] for row in classify.classify(records(), bundle)]


def test_a_table_missing_a_feature_the_model_was_trained_on_is_refused(bundle):
    incomplete = [{"ID": row["ID"], FEATURES[0]: row[FEATURES[0]]} for row in records()]
    with pytest.raises(ToolInputError) as raised:
        classify.classify(incomplete, bundle)
    assert FEATURES[1] in str(raised.value)
    assert "same training run" in str(raised.value)


def test_an_empty_table_is_refused_rather_than_answered(bundle):
    with pytest.raises(ToolInputError, match="no patient to classify"):
        classify.classify([], bundle)


def test_a_column_naming_two_lines_is_normalised_the_way_training_was(bundle, tmp_path):
    """LightGBM refuses a `/` in a feature name, so a column naming two lines --
    `MAX_ANS_PNS/ROr_LOr_Yaw` -- has to be normalised the same way the training
    set was, and only those columns are touched."""
    columns = ["CB_ROr_ROr_RL", "MAX_ANS_PNS_ROr_LOr_Yaw"]
    train(tmp_path / "b" / "sym_asymm.txt", columns, lambda row: row[columns[0]] > 0)
    train(tmp_path / "b" / "mand_asym.txt", columns, lambda row: row[columns[1]] > 0)
    train(tmp_path / "b" / "max_asym.txt", columns, lambda row: row[columns[1]] > 0)

    rows = [{"ID": "1", "CB_ROr_ROr_RL": 1.0, "MAX_ANS_PNS/ROr_LOr_Yaw": 2.0}]
    classified = classify.classify(rows, str(tmp_path / "b"))
    assert classified[0]["Asymmetry"] in ("Symmetric", "Asymmetric")


@pytest.mark.parametrize(("raw", "cleaned"), [
    ("MAX_ANS_PNS/ROr_LOr_Yaw", "MAX_ANS_PNS_ROr_LOr_Yaw"),
    ("3D Distance", "f_3D_Distance"),
    ('a"b', "ab"),
    ("", "f_unnamed"),
])
def test_a_name_is_normalised_the_way_upstream_normalises_it(raw, cleaned):
    assert classify.clean_name(raw) == cleaned


# ---------------------------------------------------------------------------
# The feature table it reads
# ---------------------------------------------------------------------------

def test_a_feature_column_says_which_region_and_which_axis_it_is():
    described = features.describe_feature("MAND_RCo_LCo_RL")
    assert described["region"] == "MAND"
    assert described["axis"] == "Transverse"
    assert described["landmarks"] == ["RCo", "LCo"]
    assert described["average"] is False


def test_a_column_naming_more_than_two_landmarks_is_an_average():
    described = features.describe_feature("CB_RPo_LPo_ROr_LOr_IS")
    assert described["average"] is True
    assert described["landmarks"] == ["RPo", "LPo", "ROr", "LOr"]


def test_a_feature_is_looked_up_by_the_landmarks_cell_it_rebuilds():
    """The separator is part of the contract between the two steps: a distance
    is written `A - B` and an angle `A-B / C-D`."""
    distance = features.describe_feature("CB_ROr_LOr_RL")
    angle = features.describe_feature("CB_ANS_PNS/ROr_LOr_Yaw")
    assert features._label_for(distance, 0) == "ROr - LOr"
    assert features._label_for(angle, 0) == "ANS-PNS / ROr-LOr"


def test_a_feature_naming_a_landmark_the_run_never_made_is_left_empty(tmp_path):
    """Upstream indexed it blind, so one missing landmark raised a KeyError,
    the workbook was never written, and every later step failed looking for a
    file that did not exist."""
    stats = {"CB": {"ID": ["1"], "Landmarks": ["ROr - LOr"], "Transverse": ["2.5"]}}
    report = {}
    built = features.build_feature_table(
        stats, ["CB_ROr_LOr_RL", "CB_Me_Gn_RL"], report=report
    )
    assert built == [{"ID": "1", "CB_ROr_LOr_RL": 2.5}]
    assert report["features_empty"]["1"] == ["CB_Me_Gn_RL"]


def test_an_average_is_not_taken_over_what_is_left(tmp_path):
    """It would quietly report a different measurement than the column claims."""
    stats = {"CB": {"ID": ["1"], "Landmarks": ["ROr - LOr"], "Transverse": ["2.5"]}}
    built = features.build_feature_table(stats, ["CB_ROr_LOr_Me_Gn_RL"])
    assert built == [{"ID": "1"}]


def test_a_template_naming_no_feature_is_refused(tmp_path):
    import pandas as pd

    pd.DataFrame(columns=["ID", "Asymmetry", "Mand", "Max"]).to_excel(
        str(tmp_path / "empty.xlsx"), index=False
    )
    with pytest.raises(ToolInputError, match="names no feature column"):
        features.read_template_columns(str(tmp_path / "empty.xlsx"))


def test_the_answers_are_not_read_back_as_features(tmp_path):
    """A model asked to read its own answer would be reading the training set's
    labels."""
    import pandas as pd

    pd.DataFrame(columns=["ID", "CB_ROr_LOr_RL", "Asymmetry", "Mand", "Max"]).to_excel(
        str(tmp_path / "template.xlsx"), index=False
    )
    assert features.read_template_columns(str(tmp_path / "template.xlsx")) == \
        ["CB_ROr_LOr_RL"]
