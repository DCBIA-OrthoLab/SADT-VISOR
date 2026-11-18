"""SurgMovPred -- predicted surgical movements from pre-operative cephalometrics.

One stacking regressor per predicted measurement, each shipped with the scaler
it was trained against. A run loads every model found under `model`, resolves
each model's expected features against the input table's columns, and skips any
model whose features the table does not carry -- so an input with fewer
measurements yields fewer predicted targets rather than an error.

The pipeline is in pipeline.py. Only `run` is public.
"""

from pathlib import Path

from .pipeline import predict

# The DATA folder this tool's models live in: its own name. Written rather than
# derived, because which folder holds which bundle is a deployment fact and a
# wrong guess is a directory that is simply not there.
_DATA_NAME = "Surg_Mov_Pred"


def _own_models(data_root):
    """`<root>/Surg_Mov_Pred/models`, or a refusal a caller can act on.

    Handed straight to `load_model_packages`, which globs `**/*.pkl` under what
    it is given -- so the packages being filed one level down, under a bundle
    folder of their own, costs nothing here.
    """
    if data_root is None:
        raise ValueError(
            "No 'model' given and no data root to look in. Name the model "
            "folder, or run this through a server that publishes one."
        )
    return Path(data_root) / _DATA_NAME / "models"


def run(
    measurements: Path,
    output_dir: Path,
    # After `output_dir` and optional: there is one published set of models and
    # the tool loads every package it finds under it, so naming the folder was
    # a step that could only be got wrong.
    model: Path = "",
    *,
    data_root=None,
) -> dict[str, Path]:
    """Predict post-surgical skeletal movements from cephalometric measurements.

    Args:
        measurements: One CSV/XLSX/ODS table of pre-operative measurements, one
            row per patient, or a folder of them concatenated into a batch. A
            patient identifier column ('#', 'ID', 'PatientID'...) is detected
            and carried into the output; without one the identifiers are blank.
        model: Folder of model packages, one subfolder per predicted
            measurement, each holding a `stacking_package.pkl`. Left empty,
            this tool uses the models installed for it -- every package found
            underneath, however they are filed.
        output_dir: Where the two result tables are written.

    Returns:
        The Excel and CSV predictions tables, one row per patient and one column
        per predicted measurement.
    """
    # sklearn, pandas, joblib and lightgbm are imported inside the pipeline:
    # loading them costs seconds, and generating this tool's schema must not
    # pay for it.
    return predict(
        measurements=Path(measurements),
        # Its OWN models when nobody named a folder. A caller that names one
        # is pinning which models ran and is obeyed.
        model=Path(model) if model else _own_models(data_root),
        output_dir=Path(output_dir),
        # Whether a missing folder is the caller's fault or the deployment's,
        # which the path alone does not say.
        installed_model=not model,
    )
