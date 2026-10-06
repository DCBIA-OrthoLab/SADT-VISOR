"""How a client should lay this tool's panel out. Presentation only.

Nothing here changes what `run()` accepts -- `describe.py` merges these hints
into the published schema and refuses any that name an argument or an option
the signature does not offer. Delete this file and the tool still works; the
panel just gets worse.

ASO is the tool that needs this most, because its **four modes share one
schema**. Without `visible_when` a panel shows all 115 CBCT landmarks next to
all 32 teeth, whichever mode is selected, and a clinician has to know which half
to ignore. The conditions below are the old Slicer module's four-page
`QStackedWidget` expressed as data instead of as widget code.

**Everything is DERIVED, never restated** -- the tabs come from `catalogs`, so a
landmark or a tooth added there appears in its tab with no edit here. That is
the difference from the `ArgSpec` tables this replaces, which listed options by
hand and drifted from the code they described.
"""

from . import catalogs

_INPUTS = "Inputs"
_CBCT = "Landmark Reference"
_IOS = "Teeth & Landmarks"
_ADVANCED = "Advanced"
_OUTPUTS = "Outputs"

_CBCT_ONLY = {"modality": catalogs.MODALITY_CBCT}
_IOS_ONLY = {"modality": catalogs.MODALITY_IOS}
LAYOUT = {
    # Injected by the server for any tool that calls another, so it is not in
    # run()'s signature -- see describe.INJECTED_ARGUMENTS. Named here because
    # "intermediate results" says nothing to a clinician: what this chain
    # produces on the way is ALI's raw prediction, before ASO moves it.
    # Injected by the server for any tool that calls another, so it is not in
    # run()'s signature -- see describe.INJECTED_ARGUMENTS. The options are the
    # steps themselves; only the wording is this tool's to give.
    "keep_intermediate": {
        "label": "Steps to keep",
        "option_help": {
            "ALI_CBCT": "The landmarks the prediction placed, before this tool "
                        "oriented them -- what to look at when an orientation "
                        "comes out wrong.",
        },
    },
    "input": {"section": _INPUTS, "label": "Scan / Landmark Folder"},
    # Chosen by what the selection needs, not by the clinician (see
    # dispatch.choose_reference). The original extension asked for a FOLDER on
    # the user's own disk; here the bundles are on the server, and which one
    # applies is already settled by the landmarks being registered on -- the two
    # CBCT references carry disjoint sets. Still an argument: naming one is how
    # a caller uses a reference of their own.
    "reference": {"section": _INPUTS, "label": "Reference", "hidden": True},
    # For the tools that call this one: a neighbour asks for a frame BY NAME
    # and this tool finds the bundle that defines it. A clinician's selection
    # already settles the frame, as above, so the panel does not ask twice.
    "frame": {"section": _INPUTS, "label": "Reference frame", "hidden": True,
              "visible_when": _CBCT_ONLY},
    "modality": {"section": _INPUTS, "label": "Input Type"},
    # Derived from the data, per patient and per jaw: landmarks beside a scan
    # are landmarks to register on, a scan with none is one to predict or to
    # take centroids from. Asking the clinician to declare it meant the answer
    # could disagree with the folder -- and when it did, the folder won, in
    # silence. Still an argument, as the one override it can still express:
    # "ignore the landmarks that are there and predict anyway".
    "automation": {"section": _INPUTS, "label": "Mode", "hidden": True},
    # Landmarks a caller supplies rather than has predicted: what ALI wrote on
    # an earlier run, or points placed by hand. Useful in both modalities, so no
    # condition -- it is the escape hatch that makes fully-automated work with
    # no landmark tool at all.
    #
    # NOT labelled "(optional)": the client already says so for any optional
    # file argument, and saying it here too printed "Landmark folder (optional)
    # (optional)". A tool names the thing; whether it is required is the
    # schema's to say, and the panel's to show.
    #
    # In Advanced because of who it is FOR. Someone who already has landmarks
    # goes looking for it; someone who does not should not have to step over it
    # on the way to Apply.
    "landmarks": {"section": _ADVANCED, "label": "Landmark folder"},

    # -- CBCT ---------------------------------------------------------------
    "cbct_landmarks": {
        "section": _CBCT,
        "label": "Registration landmarks",
        "ui": "tabs",
        "groups": {
            group: list(names)
            for group, names in catalogs.CBCT_LANDMARK_GROUPS.items()
        },
        "visible_when": _CBCT_ONLY,
    },
    # Found in the same model folder the server already hands over -- one
    # directory per landmark, which is the shape ALI_CBCT lays its weights
    # out in. A clinician choosing a fully-automated run is not choosing
    # which weights predict; they are asking not to place points by hand.
    "landmark_model": {"section": _CBCT, "label": "Landmark model bundle",
                       "hidden": True},
    # Detected from the data (see dispatch._run_cbct), so the panel does not
    # put the question to a clinician who cannot see the answer either.
    # Still an argument: a caller who knows better can force it.
    "dicom_input": {"section": _INPUTS, "label": "Input is DICOM", "hidden": True},

    # -- IOS ----------------------------------------------------------------
    "ios_teeth": {
        "section": _IOS,
        "label": "Teeth",
        # A dental chart, one row per arch, in universal-numbering order --
        # not 32 check boxes stacked in a column.
        "ui": "tabs",
        "groups": {jaw: list(teeth) for jaw, teeth in catalogs.TOOTH_GROUPS.items()},
        "visible_when": _IOS_ONLY,
    },
    "ios_landmark_types": {
        "section": _IOS, "label": "Landmark types", "ui": "inline",
        "visible_when": _IOS_ONLY,
    },
    "ios_jaws": {
        "section": _IOS, "label": "Jaws", "ui": "inline", "visible_when": _IOS_ONLY,
    },
    "ios_occlusion": {
        "section": _IOS, "label": "Occlusion", "visible_when": _IOS_ONLY,
    },

    # -- Outputs ------------------------------------------------------------
    "output_suffix": {"section": _OUTPUTS, "label": "Output suffix"},
}
