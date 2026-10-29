"""How a client should lay this tool's panel out. Presentation only.

Short, because this tool's schema is short. Before the split it shared a panel
with the CBCT arguments and needed a `modality` condition on every field; now
the only arguments published are the ones an intraoral run reads.
"""

from sadt_areg_common import catalogs

_INPUTS = "Inputs"

# Which of the two input shapes the panel asks for. The per-type fields and the
# folder of patients are never shown together: a reader who filled both would
# not know which one the run read. Each is REQUIRED in its own layout, which
# the signature cannot say -- both are optional there, since either may be
# the one not sent.
_BY_TYPE = {"input_layout": [catalogs.LAYOUT_BY_TYPE]}
_BY_PATIENT = {"input_layout": [catalogs.LAYOUT_BY_PATIENT]}

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
#
# `AUTOMATION_AUTO` is in it because that is now the DEFAULT value of
# `automation`: a condition naming only Fully-Automated matches nothing on a
# request nobody has overridden. Both fields it guards happen to be `hidden`
# today, so nothing is visibly wrong either way -- which is exactly why it is
# written down, rather than left as a trap for the next field added here.
_FULLY = {"automation": [catalogs.AUTOMATION_AUTO, catalogs.AUTOMATION_FULLY]}

LAYOUT = {
    # Injected by the server for any tool that calls another, so it is not in
    # run()'s signature -- see describe.INJECTED_ARGUMENTS. The options are the
    # steps themselves; only the wording is this tool's to give, and it is
    # worth giving: "intermediate results" says nothing to a clinician, while
    # the name of a step plus what it hands back is what makes a chain
    # inspectable. ASO, ALI_IOS and VFACE already offer it, and AREG was the
    # odd one out. The server only collects the output of the calls THIS tool
    # makes, so ticking `ASO` returns ASO's own result and not the landmarks
    # ASO predicted on the way.
    # Hidden in every engine of the facade, so the name means one thing: what
    # a reader wanted from it -- the oriented scan and ASO's landmarks -- comes
    # back as an Extra output now, in the result's space, and the raw step
    # folders were a second, harder way to get the same files.
    "keep_intermediate": {
        "hidden": True,
        "label": "Raw intermediate files",
        "option_help": {
            "ALI_IOS": "The landmarks placed on the intraoral scans, which the "
                       "mucogingival patch is built from.",
            "ASO": "The scans once both timepoints sit in a common orientation, "
                   "before the registration is run on them.",
            "Crown_Seg": "The per-tooth labelled meshes, which this tool "
                         "otherwise only uses to name the teeth.",
        },
    },

    # Injected by the server too, from the checkpoints this chain offers, and
    # it lands in a "Quality control" section of its own. Offered like ASO's:
    # nothing is ticked by default, so a run nobody asked to stop runs to the
    # end, and a ticked step pauses the run and opens what it produced in VISU.
    "stop_after": {"label": "Stop after"},

    # Segments, not a drop-down: two layouts of equal standing, both on screen,
    # and the client puts it above the inputs it switches between.
    "input_layout": {
        "section": _INPUTS, "label": "Data organisation", "ui": "segmented",
        "option_help": dict(catalogs.LAYOUT_HELP),
    },
    "patients": {
        "section": _INPUTS, "label": "Patient folders", "visible_when": _BY_PATIENT,
        "required_when": _BY_PATIENT,
    },
    "t1": {
        "section": _INPUTS, "label": "T1 (baseline)",
        "visible_when": _BY_TYPE, "required_when": _BY_TYPE,
    },
    "t2": {
        "section": _INPUTS, "label": "T2 (follow-up)",
        "visible_when": _BY_TYPE, "required_when": _BY_TYPE,
    },
    # NOT "Mode". The facade publishes its own `mode` -- the modality -- and
    # labels it "Mode" too, so a panel reached through AREG shows two dropdowns
    # side by side under the same word: one saying CBCT, the next saying
    # Semi-Automated. A user who read the first as "the mode" then hunted for
    # check boxes that only exist in another value of the SECOND one, and had
    # no way to tell which was which. The engines keep this label when opened
    # directly, where there is only one dropdown and no ambiguity, so the word
    # has to carry its own meaning either way.
    # Read off the meshes (see dispatch.derive_automation): ones that already
    # carry their tooth labels are Semi-Automated, ones that do not are
    # Fully-Automated. Asking the clinician meant the answer could disagree with
    # the folder, and when it did a run either relabelled meshes that were
    # already labelled or failed on meshes that were not.
    #
    # Still an argument, as the override it can express -- and the one case that
    # needs it: a deployment without Crown_Seg, handed labelled meshes, has to
    # name 'Semi-Automated' because the door-check tests FULLY's requirements
    # while the mode is still unknown (see dispatch.main).
    #
    # Hidden here as well as in the other two engines: the facade publishes ONE
    # `automation` and takes its presentation from whichever engine composes
    # first, so a hint left in one place would depend on the order of a dict in
    # deployment.toml.
    "automation": {"section": _INPUTS, "label": "Automation", "hidden": True},

    "patch": {"section": _REGISTRATION, "label": "Registration patch"},
    # `ios_reference`, NOT `reference`: AREG_CBCT publishes a `reference` that is
    # a real clinical choice -- Frankfurt or Occlusal -- and keeps its dropdown.
    # One name has to mean one thing across the engines the AREG facade
    # composes, so this engine opting its own out of the hosted-model convention
    # under the shared name would have made the facade refuse to publish at all.
    # AREG_IOSCBCT names its own `cbct_reference` for exactly this reason.
    #
    # Not offered: there is one intraoral reference and it is ASO's, which ASO
    # resolves itself, so there is nothing here for a clinician to decide and a
    # wrong pick is a chain oriented into the wrong reference.
    "ios_reference": {
        "hidden": True,
        "section": _REGISTRATION, "label": "Orientation reference", "visible_when": _FULLY,
    },
    # Not offered: `AREG_model` holds exactly one checkpoint, and this field
    # left empty arrived as the whole `models/` folder -- sixty checkpoints --
    # which killed the run on "has to name an entry holding exactly one" AFTER
    # it had segmented and oriented both timepoints.
    "registration_model": {
        "hidden": True,
        "section": _REGISTRATION, "label": "Patch model", "visible_when": _PREDICTED,
    },
    # Not offered: Crown_Seg resolves its own weights, and this name has to
    # mean the same thing in every engine the AREG facade composes.
    "crown_model": {"hidden": True, 
        "section": _REGISTRATION, "label": "Crown segmentation model", "visible_when": _FULLY,
    },
    # Not offered: ALI_IOS finds its own weights, as it does for every other
    # landmark bundle in this family.
    "mgl_model": {
        "hidden": True,
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
