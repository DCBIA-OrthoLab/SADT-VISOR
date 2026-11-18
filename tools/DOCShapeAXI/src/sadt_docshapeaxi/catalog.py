"""What each checkpoint is, and everything that follows from it.

Upstream's CLI takes five arguments to say one thing: `data_type`, `task`,
`model`, `nn` and `num_classes`. Only five of their combinations exist, and the
other four values are functions of the checkpoint:

    airways_2_class    binary      2 classes   classification
    airways_4_class    severity    4 classes   classification
    airways_4_regress  regression  1 output    REGRESSION
    condyles_4_class   severity    4 classes   classification
    cleft_4_class      severity    4 classes   classification

Pairing them by hand is how a caller gets a plausible wrong answer: ask for
`airways_2_class` with `num_classes=4` and the run completes, reporting a
severity grade the network never learned. `num_classes` also drives how many
GradCAM maps are produced, so a wrong value silently changes the explanation
too. One argument names the checkpoint; the rest is read from here.

The same reasoning is why Batch_Dental_Seg has one `model` argument rather than
a bundle plus a label table.
"""

from .errors import ToolInputError

# The `nn` values upstream resolves out of `shapeaxi.saxi_nets_lightning`.
CLASSIFICATION = "SaxiMHAFBClassification"
REGRESSION = "SaxiMHAFBRegression"


class Analysis:
    """One checkpoint and what it implies."""

    def __init__(self, checkpoint: str, anatomy: str, task: str,
                 classes: int, network: str):
        self.checkpoint = checkpoint
        self.anatomy = anatomy
        self.task = task
        self.classes = classes
        self.network = network

    @property
    def is_regression(self) -> bool:
        return self.network == REGRESSION


# Keyed by the file name as it is STAGED, which is the name a client sends.
#
# Upstream's `find_model_name` returns `clefts_4_class`, with an s, while the
# published release asset is `cleft_4_class.ckpt`. The file name is what exists
# on disk, so it is what this table is keyed by; the module's spelling appears
# nowhere else and would resolve to nothing.
ANALYSES = {
    "airways_2_class.ckpt": Analysis(
        "airways_2_class.ckpt", "Nasopharynx airway obstruction",
        "binary", 2, CLASSIFICATION,
    ),
    "airways_4_class.ckpt": Analysis(
        "airways_4_class.ckpt", "Nasopharynx airway obstruction",
        "severity", 4, CLASSIFICATION,
    ),
    "airways_4_regress.ckpt": Analysis(
        "airways_4_regress.ckpt", "Nasopharynx airway obstruction",
        "regression", 1, REGRESSION,
    ),
    "condyles_4_class.ckpt": Analysis(
        "condyles_4_class.ckpt", "Mandibular condyle",
        "severity", 4, CLASSIFICATION,
    ),
    "cleft_4_class.ckpt": Analysis(
        "cleft_4_class.ckpt", "Alveolar bone defect in cleft",
        "severity", 4, CLASSIFICATION,
    ),
}


def analysis_for(checkpoint_name: str) -> Analysis:
    """The row for a checkpoint, by file name.

    Raises ToolInputError (a ValueError) naming what IS known rather than falling through with an
    unbound `model_name`, which is what upstream did when no branch matched:
    it logged "no model found for undefined task" and carried on to a
    NameError.
    """
    known = ANALYSES.get(checkpoint_name)
    if known is not None:
        return known
    # The analyses are described in words first: the server redacts file
    # names, so the list of checkpoints at the end reaches an operator's log
    # as a row of placeholders, while the anatomy and task pairs survive.
    raise ToolInputError(
        f"'model' is not a checkpoint this tool knows how to read. Known "
        f"analyses: {describe_known()}. Expected one of: "
        f"{', '.join(sorted(ANALYSES))}."
    )


def describe_known() -> str:
    """Every anatomy with the tasks it has a checkpoint for, in words."""
    tasks = {}
    for analysis in ANALYSES.values():
        tasks.setdefault(analysis.anatomy, []).append(analysis.task)
    return "; ".join(
        f"{anatomy} ({', '.join(names)})" for anatomy, names in tasks.items()
    )
