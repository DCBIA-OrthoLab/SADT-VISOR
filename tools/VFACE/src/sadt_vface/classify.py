"""Reading the feature table as a classification of the patient's asymmetry.

Three gradient-boosted models, loaded from the bundle the server hosts:

    sym_asymm.txt   symmetric, or not
    mand_asym.txt   of the asymmetric ones: is the mandible involved
    max_asym.txt    of the asymmetric ones: is the maxilla involved

The two sub-models are asked only about the patients the first called
asymmetric, which is how they were trained: a "is the mandible asymmetric"
answer for a patient who is symmetric is not a lower-confidence answer, it is a
question outside the model's domain.

**The model chooses its own columns.** Each carries `feature_name_`, the list
it was trained on, and that is what is selected out of the feature table --
never the table's own column order, which would silently feed the model a
different feature in each position the moment the template gained a column.
"""

import logging
import os
import re

from .errors import ToolInputError, describe_failure

logger = logging.getLogger(__name__)

MODEL_FILES = {
    "symmetry": "sym_asymm.txt",
    "mandible": "mand_asym.txt",
    "maxilla": "max_asym.txt",
}

# What the two model outputs mean. 0 is the positive finding in the first --
# the model is trained to predict "symmetric" -- and the sub-models answer a
# plain yes or no.
SYMMETRY_LABELS = {0: "Asymmetric", 1: "Symmetric"}
INVOLVED_LABELS = {0: "False", 1: "True"}


def clean_name(name: str) -> str:
    """A column name LightGBM can hold.

    It refuses `"` `'` `\\` and a few others in a feature name, so a column
    naming two lines -- `MAX_ANS_PNS/ROr_LOr_Yaw` -- has to be normalised the
    same way the training set was. Applied only to the columns that carry a
    `/`, which is what upstream does: renaming every column would rename ones
    the model knows under their original spelling.
    """
    try:
        name = str(name).strip()
        name = re.sub(r"[\r\n\t]", " ", name)
        name = re.sub(r"[\"'\\\\]", "", name)
        name = re.sub(r"[^0-9a-zA-Z_]+", "_", name)
        name = re.sub(r"_+", "_", name).strip("_")
        if re.match(r"^\d", name):
            name = f"f_{name}"
        return name or "f_unnamed"
    except Exception:  # noqa: BLE001 - a name that cannot be cleaned is still a column
        logger.warning("VFACE: a feature column's name could not be normalised")
        return "f_unnamed"


def load_models(model_dir: str) -> dict:
    """The three models, by what they answer.

    Refused by name when one is missing: a classification that ran two of three
    models would answer the sub-questions on a symmetry verdict nothing made.
    """
    import joblib

    if not os.path.isdir(model_dir):
        raise ToolInputError(
            "'classifier_model' is not a folder: it should be the classifier "
            "bundle, the folder holding the three models."
        )

    missing = [name for name in MODEL_FILES.values()
               if not os.path.isfile(os.path.join(model_dir, name))]
    if missing:
        raise ToolInputError(
            f"The classifier bundle is missing {', '.join(sorted(missing))}. It should "
            f"hold all three of {', '.join(sorted(MODEL_FILES.values()))}."
        )

    models = {}
    for question, filename in MODEL_FILES.items():
        try:
            models[question] = joblib.load(os.path.join(model_dir, filename))
        except Exception as exc:  # noqa: BLE001 - re-raised, named, below
            # A model pickled under another joblib, LightGBM or scikit-learn
            # than the one installed fails here with a message about module
            # internals. The bundle is the deployment's, not the caller's, so
            # this is a RuntimeError rather than a 422 -- and it names WHICH
            # model, which the unpickler's own message never does.
            raise RuntimeError(
                f"the classifier's '{question}' model could not be loaded "
                f"({describe_failure(exc)})"
            ) from exc
    return models


def _select(frame, model, question: str):
    """The columns this model was trained on, or a refusal naming what is absent."""
    wanted = list(getattr(model, "feature_name_", []) or [])
    if not wanted:
        raise ToolInputError(
            f"The '{question}' model does not say which features it was trained on, "
            "so nothing can be selected for it."
        )
    absent = [name for name in wanted if name not in frame.columns]
    if absent:
        raise ToolInputError(
            f"The feature table is missing {len(absent)} of the {len(wanted)} features "
            f"the '{question}' model was trained on, the first being "
            f"{', '.join(absent[:5])}. The feature template and the model bundle have "
            "to come from the same training run."
        )
    return frame[wanted]


def classify(records, model_dir: str, report: dict = None) -> list:
    """`records` with an `Asymmetry`, `Mand` and `Max` column added to each.

    The sub-models are allowed to fail without costing the symmetry verdict:
    theirs is the finer question, and "asymmetric, and we could not say where"
    is a usable answer where nothing at all is not.
    """
    import pandas as pd

    models = load_models(model_dir)
    frame = pd.DataFrame(records)
    if frame.empty:
        raise ToolInputError("There is no patient to classify.")

    renamed = {column: clean_name(column) for column in frame.columns if "/" in str(column)}
    if renamed:
        frame = frame.rename(columns=renamed)

    verdict = models["symmetry"].predict(_select(frame, models["symmetry"], "symmetry"))
    frame["Asymmetry"] = pd.Series(verdict, index=frame.index).map(SYMMETRY_LABELS)
    frame["Mand"] = "False"
    frame["Max"] = "False"

    asymmetric = pd.Series(verdict, index=frame.index) == 0
    if asymmetric.any():
        logger.info("VFACE: %d of %d patient(s) read as asymmetric",
                    int(asymmetric.sum()), len(frame))
        for question, column in (("mandible", "Mand"), ("maxilla", "Max")):
            try:
                sub = frame.loc[asymmetric]
                answer = models[question].predict(_select(sub, models[question], question))
                frame.loc[asymmetric, column] = (
                    pd.Series(answer, index=sub.index).map(INVOLVED_LABELS)
                )
            except Exception as exc:  # noqa: BLE001 - the finer question may fail alone
                logger.warning("VFACE: the %s sub-classification failed (%s)",
                               question, describe_failure(exc))
                if report is not None:
                    report.setdefault("classification_partial", {})[question] = (
                        f"{type(exc).__name__}: {exc}"
                    )
    else:
        logger.info("VFACE: no patient read as asymmetric; the sub-models are not asked")

    return frame.to_dict("records")


def write_table(records, path: str) -> str:
    import pandas as pd

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pd.DataFrame(records).to_excel(path, index=False)
    return path
