"""The names of the Extra outputs, written once.

`__init__.py` has to spell them out in `run()`'s annotation (the schema is read
from it), so `tests/test_run.py` checks that the two lists agree.
"""

OUTPUT_ORIENTED_SCAN = "Oriented scan"
OUTPUT_MANDIBLE = "Mandible segmentation"

OUTPUT_LABEL = "Extra outputs"
OUTPUT_NOTE = ("Saved beside the crops, in the oriented frame. The crops, the "
               "report and the quick view are always written.")
OUTPUT_HELP = {
    OUTPUT_ORIENTED_SCAN: "The whole scan in the Frankfurt horizontal frame the "
                          "crops were cut from, as <scan>_oriented.nii.gz.",
    OUTPUT_MANDIBLE: "The mandible mask (AMASSS) in that same frame, as "
                     "<scan>_mandible.nii.gz.",
}
