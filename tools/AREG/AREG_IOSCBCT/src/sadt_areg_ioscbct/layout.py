"""How a client should lay this tool's panel out. Presentation only."""

from sadt_areg_common import catalogs

_INPUTS = "Inputs"
_LANDMARKS = "Landmarks"
_MODELS = "Models"
_OUTPUTS = "Outputs"

# Registration takes the landmarks; the other two predict them. Showing the
# landmark folders in a mode that overwrites them is how a user comes to believe
# their files were used.
_SUPPLIED = {"automation": catalogs.AUTOMATION_REGISTRATION}
_PREDICTED = {"automation": [catalogs.AUTOMATION_SEMI, catalogs.AUTOMATION_FULLY]}
_ORIENTED = {"automation": catalogs.AUTOMATION_FULLY}

LAYOUT = {
    # Injected by the server for every tool that calls another (see
    # describe.INJECTED_ARGUMENTS), and unnamed it arrives with a generic label
    # in a section of its own -- an "Intermediate results" box in a panel that
    # never asked for one. Hidden rather than renamed: what this chain leaves
    # behind on the way is not something a clinician is being offered yet, and
    # a check box that promises files nobody has decided to return is worse
    # than no check box.
    "keep_intermediate": {"hidden": True},

    # Injected by the server too, from the checkpoints this chain offers, and
    # it lands in a "Quality control" section of its own. Not offered yet:
    # stopping a run for review is a workflow this deployment has not decided
    # on, and a box that stops a cohort halfway is not one to leave lying
    # around until it has.
    "stop_after": {"hidden": True},

    "ios": {"section": _INPUTS, "label": "Intraoral scans"},
    "cbct": {"section": _INPUTS, "label": "CBCT volumes"},
    # NOT "Mode". The facade publishes its own `mode` -- the modality -- and
    # labels it "Mode" too, so a panel reached through AREG shows two dropdowns
    # side by side under the same word: one saying CBCT, the next saying
    # Semi-Automated. A user who read the first as "the mode" then hunted for
    # check boxes that only exist in another value of the SECOND one, and had
    # no way to tell which was which. The engines keep this label when opened
    # directly, where there is only one dropdown and no ambiguity, so the word
    # has to carry its own meaning either way.
    "automation": {"section": _INPUTS, "label": "Automation"},

    "ios_landmarks": {
        "section": _LANDMARKS, "label": "Intraoral landmarks", "visible_when": _SUPPLIED,
    },
    "cbct_landmarks": {
        "section": _LANDMARKS, "label": "CBCT landmarks", "visible_when": _SUPPLIED,
    },

    "crown_model": {
        "section": _MODELS, "label": "Crown segmentation model", "visible_when": _PREDICTED,
    },
    "ios_landmark_model": {
        "section": _MODELS, "label": "Intraoral landmark bundle", "visible_when": _PREDICTED,
    },
    "landmark_model": {
        "section": _MODELS, "label": "CBCT landmark bundle", "visible_when": _PREDICTED,
    },
    "cbct_reference": {
        "section": _MODELS, "label": "Orientation reference", "visible_when": _ORIENTED,
    },

    "max_dist": {"section": _OUTPUTS, "label": "ICP match distance (mm)"},
    "output_suffix": {"section": _OUTPUTS, "label": "Output suffix"},
}
