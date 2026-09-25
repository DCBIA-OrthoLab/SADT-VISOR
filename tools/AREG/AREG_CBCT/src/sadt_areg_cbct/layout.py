"""How a client should lay this tool's panel out. Presentation only.

Nothing here changes what `run()` accepts -- `describe.py` merges these hints
into the published schema and refuses any that name an argument the signature
does not take. Delete this file and the tool still works; the panel gets worse.

No `modality` condition anywhere, and that is the split showing through. The
merged AREG carried one on almost every field, because two modalities and three
automation modes shared a single schema and most arguments applied to exactly
one combination. There is no other modality here now, so what remains are the
conditions that are really about the MODE.
"""

from sadt_areg_common import catalogs

_INPUTS = "Inputs"
_REGISTRATION = "Registration"
_OUTPUTS = "Outputs"

# Each mirrors a check in dispatch.py. An argument the chosen mode never reads
# is not merely noise: shown as optional beside the ones that matter, it reads
# as something the user chose not to fill, and the refusal then arrives at the
# end of a run instead of before it.
_SEGMENTED = {  # the modes that produce their own masks
    "automation": [catalogs.AUTOMATION_FULLY, catalogs.AUTOMATION_ORIENTED]
}
_ORIENTED = {"automation": catalogs.AUTOMATION_ORIENTED}
_SEMI = {"automation": catalogs.AUTOMATION_SEMI}

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

    "t1": {"section": _INPUTS, "label": "T1 (baseline)"},
    "t2": {"section": _INPUTS, "label": "T2 (follow-up)"},
    # NOT "Mode". The facade publishes its own `mode` -- the modality -- and
    # labels it "Mode" too, so a panel reached through AREG shows two dropdowns
    # side by side under the same word: one saying CBCT, the next saying
    # Semi-Automated. A user who read the first as "the mode" then hunted for
    # check boxes that only exist in another value of the SECOND one, and had
    # no way to tell which was which. The engines keep this label when opened
    # directly, where there is only one dropdown and no ambiguity, so the word
    # has to carry its own meaning either way.
    "automation": {"section": _INPUTS, "label": "Automation"},
    "dicom_input": {"section": _INPUTS, "label": "Input is DICOM"},

    # The one argument a clinician must actually think about: register on what
    # has NOT changed between the two timepoints.
    "regions": {
        "section": _REGISTRATION,
        "label": "Register on",
        "ui": "inline",
    },
    "t1_masks": {"section": _REGISTRATION, "label": "T1 masks", "visible_when": _SEMI},
    # The original's second group, under its own name. Stacked -- the default
    # layout -- and NOT inline like `regions` right above it: three short
    # labels fit across a panel that is 425 px wide, six of "Cervical
    # vertebra" length do not, and they pushed the section wider than the
    # module panel. Read down, one per line.
    "segmentations": {
        "section": _REGISTRATION, "label": "AMASSS segmentation",
        "visible_when": _SEGMENTED,
    },
    # Not offered: the modes that segment segment with AMASSS, and the bundle
    # is the one the deployment already publishes (see dispatch._own_segmentation).
    # The argument stays, so a caller with a reason can still name another one.
    "segmentation_model": {
        "section": _REGISTRATION, "label": "Segmentation model",
        "visible_when": _SEGMENTED, "hidden": True,
    },
    "segmentation_label": {
        "section": _REGISTRATION, "label": "Mask label value", "visible_when": _SEGMENTED,
    },
    "reference": {
        "section": _REGISTRATION, "label": "Orientation reference", "visible_when": _ORIENTED,
    },
    "landmark_model": {
        "section": _REGISTRATION, "label": "Landmark model bundle", "visible_when": _ORIENTED,
    },

    "output_suffix": {"section": _OUTPUTS, "label": "Output suffix"},
}
