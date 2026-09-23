"""How a client should lay this tool's panel out. Presentation only.

Short, because this tool's schema is short. Before the split it shared a panel
with the CBCT arguments and needed a `modality` condition on every field; now
the only arguments published are the ones an intraoral run reads.
"""

from sadt_areg_common import catalogs

_INPUTS = "Inputs"
_REGISTRATION = "Registration"
_OUTPUTS = "Outputs"

# The patch decides the rest of the panel: the mucogingival band is built from
# landmarks and involves no network at all, while the palate is predicted and
# involves nothing else. Asking for a checkpoint on the MGL side is how a user
# comes to believe that mode needs one. Listed rather than negated:
# `visible_when` compares, so "every patch but MGL" is written by naming them.
_MGL = {"patch": catalogs.PATCH_MGL}
_PREDICTED = {"patch": [p for p in catalogs.PATCH_CHOICES if p != catalogs.PATCH_MGL]}
# Only the fully-automated mode labels and orients the meshes itself; the
# semi-automated one takes meshes that already carry both.
_FULLY = {"automation": catalogs.AUTOMATION_FULLY}

LAYOUT = {
    # Injected by the server for every tool that calls another (see
    # describe.INJECTED_ARGUMENTS), and unnamed it arrives with a generic label
    # in a section of its own -- an "Intermediate results" box in a panel that
    # never asked for one. Hidden rather than renamed: what this chain leaves
    # behind on the way is not something a clinician is being offered yet, and
    # a check box that promises files nobody has decided to return is worse
    # than no check box.
    "keep_intermediate": {"hidden": True},

    "t1": {"section": _INPUTS, "label": "T1 (baseline)"},
    "t2": {"section": _INPUTS, "label": "T2 (follow-up)"},
    "automation": {"section": _INPUTS, "label": "Mode"},

    "patch": {"section": _REGISTRATION, "label": "Registration patch"},
    "reference": {
        "section": _REGISTRATION, "label": "Orientation reference", "visible_when": _FULLY,
    },
    "registration_model": {
        "section": _REGISTRATION, "label": "Patch model", "visible_when": _PREDICTED,
    },
    "crown_model": {
        "section": _REGISTRATION, "label": "Crown segmentation model", "visible_when": _FULLY,
    },
    "mgl_model": {
        "section": _REGISTRATION, "label": "Mucogingival landmark bundle", "visible_when": _MGL,
    },
    "mgl_landmarks": {
        "section": _REGISTRATION, "label": "Mucogingival landmarks", "visible_when": _MGL,
    },
    "mgl_patch_height": {
        "section": _REGISTRATION, "label": "Patch height (mm)", "visible_when": _MGL,
    },

    "output_suffix": {"section": _OUTPUTS, "label": "Output suffix"},
}
