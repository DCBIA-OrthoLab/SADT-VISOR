"""How a client should lay this tool's panel out. Presentation only.

Nothing here changes what `run()` accepts: `scripts/describe.py` merges these
hints into the published schema, and delete this file and the tool still works.

`scale_by` and `scale` are hidden because a clinician should not choose them. A
wrong value changes the size of the crop box with nothing on screen to show it,
and every crop of a study has to share one box to be comparable. They stay in
`run()` for research use through the API (reusing one patient's first time point
for the next, or normalising by mandible volume).

`keep_intermediate` is hidden for the reason AREG gives: what a reader wanted
from the raw step folders (the oriented scan, the mandible mask) comes back as
an Extra output, so the name means one thing.
"""

from . import catalogs

LAYOUT = {
    "sides": {"label": "Joints to crop"},
    "outputs": {
        "label": catalogs.OUTPUT_LABEL,
        "note": catalogs.OUTPUT_NOTE,
        "option_help": dict(catalogs.OUTPUT_HELP),
    },
    "scale_by": {"hidden": True},
    "scale": {"hidden": True},
    "keep_intermediate": {"hidden": True, "label": "Raw intermediate files"},
}
