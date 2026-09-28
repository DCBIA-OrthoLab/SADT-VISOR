"""How a client lays this tool's panel out. Presentation only.

The section name and the wording on the chips are the local module's, on
purpose: this is the same choice a clinician made there, and calling it
something else here would make one feature look like two.
"""

from . import mesh_export

# The box the local module called "Export format", holding the same choice.
EXPORT_SECTION = "Export format"

_FORMAT_HELP = {
    mesh_export.NIFTI: "The label volume, one file per scan. What this tool "
                       "has always written",
    "STL": "One mesh per label. What most printers and mesh viewers open",
    "OBJ": "One mesh per label, as text. The largest of the three -- the "
           "format has no binary form",
    "VTK": "One mesh per label, binary, carrying its label number",
    mesh_export.MERGED_VTK: "Every label in ONE mesh, separable again by its "
                            "`Label` array",
}

LAYOUT = {
    "export_formats": {
        "section": EXPORT_SECTION,
        # Chips rather than a column of check boxes: five short options fit on
        # one line, and the format's name reads on the button itself.
        "ui": "chips",
        "option_help": _FORMAT_HELP,
        # A run has to produce something. The panel says so before Apply
        # rather than after a request that could only come back empty.
        "min_selected": 1,
    },
    "surface_decimation": {
        "section": EXPORT_SECTION,
        "label": "Surface reduction (%)",
        # Deliberately a plain number and not a slider, and deliberately NOT
        # hidden behind `visible_when`. A `visible_when` naming a multichoice
        # compares the panel's value -- a {option: ticked} dict -- against a
        # list of option names, so it can never match and the field would be
        # hidden for good. Nothing else in this repo renders `ui: "slider"`
        # either, and a first use of it belongs in its own change.
    },
}
