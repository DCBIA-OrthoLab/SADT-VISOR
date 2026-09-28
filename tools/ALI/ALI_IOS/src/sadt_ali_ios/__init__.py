"""ALI_IOS -- Automatic Landmark Identification on intraoral surface scans.

Per tooth, the mesh is rendered from a dozen viewpoints and a 2D UNet predicts
masks that are projected back onto the surface. Writes one Slicer markups file
per scan.

Split out of the former single `ALI` tool. The reason is concrete: this engine
needs pytorch3d, which ships as a wheel built against one exact torch version
(`+pt2110cu128`), so it is pinned to torch 2.11. The CBCT engine has no reason
to move, and while the two shared a virtualenv, neither could be pinned without
the other. They share their output format and their input vocabulary, both in
`sadt_ali_common`, and nothing else.
"""

from pathlib import Path
from typing import Literal

from .dispatch import identify
from .errors import ToolInputError

# What this engine appends to the name it was handed, and it is a CONSTANT.
# It used to be an argument, `prediction_ID`, defaulting to "Pred". That made
# the marker a property of the REQUEST rather than of the tool, so nothing
# downstream could know it: pairing a scan with its landmarks, and working out
# which results belong to one patient, both have to strip a marker they can
# predict. A caller wanting to label a run labels the output FOLDER, which
# costs nobody a guess.
#
# Published through `OUTPUT_SUFFIXES` below, which is how the server learns it
# without holding a table of dental names.
PREDICTION_ID = "Pred"

# Only the part that identifies the tool, not the whole written name: a file
# is `<patient>_lm_Pred.mrk.json`, and cutting at `_lm` is what recovers the
# patient whatever follows it.
OUTPUT_SUFFIXES = ("_lm_Pred", "_lm")

# What a reader may do with what this engine produced, and therefore whether a
# stop after a call to it is somewhere to come BACK to. Landmarks are dragged
# and saved back to their own file, which is the whole reason a reader is shown
# them: an orientation computed from a bad point cannot be fixed where it is
# looked at.
REVIEW_KIND = "landmarks"


# The DATA folder this tool's weights live in. Its own name, except that
# ALI_IOS and ALI_CBCT are two engines behind the ALI facade and share one.
# Written rather than derived, for the reason the CBCT engine gives beside its
# own copy: which folder serves which engine is a deployment fact, and a wrong
# guess is a folder that is simply not there.
_DATA_NAME = "ALI"


def _own_models(data_root):
    """`<root>/ALI/models`, or a refusal a caller can act on."""
    if data_root is None:
        raise ToolInputError(
            "No 'model' given and no data root to look in. Name the model "
            "bundle, or run this through a server that publishes one."
        )
    return Path(data_root) / _DATA_NAME / "models"


def run(
    input: Path,
    output_dir: Path,
    # After `output_dir` and optional, the shape the CBCT engine already has
    # and for the same reason: it lets a neighbour ask for landmarks WITHOUT
    # naming weights. Required, a supervised call omitting it died on
    # `TypeError: run() missing 1 required positional argument: 'model'`, and
    # the caller had to hold a name for this tool's storage. Empty means "my
    # own", resolved below from this tool's data folder.
    model: Path = "",
    # Mucogingival is OFF by default: it is one point per lower tooth on the
    # gingival margin, wanted by a mandible registration and by nobody asking
    # for crown landmarks. On by default would add a third pass over every mesh
    # of every existing request.
    networks: list[
        Literal["Occlusal", "Cervical", "Mucogingival"]
    ] = ["Occlusal", "Cervical"],
    # Spelled out because `Literal` takes literals only -- it cannot be built
    # from catalog.LANDMARKS. That makes this a second declaration of the same
    # set, which is the thing this contract otherwise avoids, so a test asserts
    # the two agree.
    landmarks: list[
        Literal[
            # Occlusal Upper
            "UL7O", "UL7MB", "UL7DB", "UL6O", "UL6MB", "UL6DB", "UL5O", "UL5MB",
            "UL5DB", "UL4O", "UL4MB", "UL4DB", "UL3O", "UL3MB", "UL3DB", "UL2O",
            "UL2MB", "UL2DB", "UL1O", "UL1MB", "UL1DB", "UR1O", "UR1MB", "UR1DB",
            "UR2O", "UR2MB", "UR2DB", "UR3O", "UR3MB", "UR3DB", "UR4O", "UR4MB",
            "UR4DB", "UR5O", "UR5MB", "UR5DB", "UR6O", "UR6MB", "UR6DB", "UR7O",
            "UR7MB", "UR7DB",
            # Occlusal Lower
            "LL7O", "LL7MB", "LL7DB", "LL6O", "LL6MB", "LL6DB", "LL5O", "LL5MB",
            "LL5DB", "LL4O", "LL4MB", "LL4DB", "LL3O", "LL3MB", "LL3DB", "LL2O",
            "LL2MB", "LL2DB", "LL1O", "LL1MB", "LL1DB", "LR1O", "LR1MB", "LR1DB",
            "LR2O", "LR2MB", "LR2DB", "LR3O", "LR3MB", "LR3DB", "LR4O", "LR4MB",
            "LR4DB", "LR5O", "LR5MB", "LR5DB", "LR6O", "LR6MB", "LR6DB", "LR7O",
            "LR7MB", "LR7DB",
            # Cervical Upper
            "UL7CL", "UL7CB", "UL6CL", "UL6CB", "UL5CL", "UL5CB", "UL4CL", "UL4CB",
            "UL3CL", "UL3CB", "UL2CL", "UL2CB", "UL1CL", "UL1CB", "UR1CL", "UR1CB",
            "UR2CL", "UR2CB", "UR3CL", "UR3CB", "UR4CL", "UR4CB", "UR5CL", "UR5CB",
            "UR6CL", "UR6CB", "UR7CL", "UR7CB",
            # Cervical Lower
            "LL7CL", "LL7CB", "LL6CL", "LL6CB", "LL5CL", "LL5CB", "LL4CL", "LL4CB",
            "LL3CL", "LL3CB", "LL2CL", "LL2CB", "LL1CL", "LL1CB", "LR1CL", "LR1CB",
            "LR2CL", "LR2CB", "LR3CL", "LR3CB", "LR4CL", "LR4CB", "LR5CL", "LR5CB",
            "LR6CL", "LR6CB", "LR7CL", "LR7CB",
            # Mucogingival Lower
            "LL6MG", "LL5MG", "LL4MG", "LL3MG", "LL2MG", "LL1MG", "L0MG", "LR1MG",
            "LR2MG", "LR3MG", "LR4MG", "LR5MG", "LR6MG",
        ]
    ] = [],
    device: Literal["cuda", "cpu"] = "cuda",
    *,
    sup=None,
    data_root=None,
) -> Path:
    """Place anatomical landmarks on an intraoral surface scan.

    Args:
        input: One intraoral surface (.vtk/.stl), or a folder of them for a
            batch. Folders are searched recursively. An input holding CBCT
            volumes is refused by name rather than half processed -- run
            ALI_CBCT on those.
        model: The model bundle, holding flat checkpoints named with an 'O' or
            'C' token and an 'Upper' or 'Lower' one, e.g. `Upper_O_model.pth`.
        output_dir: Where results are written -- one `<scan>_lm_<ID>.mrk.json`
            per scan, mirroring the input's own folder tree, plus
            `run_report.json`. Nothing is written outside it.
        networks: Occlusal predicts the occlusal point and the mesio- and
            disto-buccal cusps; Cervical predicts the cervical lingual and
            buccal points; Mucogingival predicts one point per lower tooth on
            the gingival margin rather than on the crown, and runs on the
            mandible only. A point it had to place from a fit of the arch,
            rather than from the render, carries a caveat in its own
            `description` field and in `landmarks_degraded` in the report.
        landmarks: Predict exactly these landmarks -- naming any of them
            REPLACES the family selection rather than narrowing it, which is
            what lets a caller ask for the points it needs instead of taking a
            whole family to use three of them. Left empty, `networks` decides,
            which is what a client showing no family control relies on.
        device: "cuda" or "cpu". CUDA falls back to CPU when no card is
            visible, with a warning.

    Returns:
        The output directory, holding the markups files and the run report.

    The landmark networks are pointed at teeth, so a mesh has to carry a
    per-point tooth-label array. One that does not is sent to `Crown_Seg`
    through the supervisor and processed after the meshes that already had
    theirs -- the ready ones first, so their landmarks are on disk before a
    second of segmentation is spent, and `segmented_on_the_fly` in the run
    report says which meshes needed it. A mesh `Crown_Seg` cannot label is
    reported as that one mesh failing, never as the batch failing.

    With no supervisor -- a direct call, or a server too old to inject one --
    an unlabelled mesh refuses the batch as it always did, naming `Crown_Seg`:
    run it over the meshes yourself and pass its output here.
    """
    # torch and pytorch3d are imported inside the engine: CI imports this
    # module on every PR to publish the schema, and that must not cost a CUDA
    # stack.
    output_dir = Path(output_dir)
    identify(
        input_path=str(input),
        model_path=str(model or _own_models(data_root)),
        output_dir=str(output_dir),
        ios_networks=networks,
        landmarks=landmarks,
        prediction_ID=PREDICTION_ID,
        device=device,
        sup=sup,
    )
    return output_dir
