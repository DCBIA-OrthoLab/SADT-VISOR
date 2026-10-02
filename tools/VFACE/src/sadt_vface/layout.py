"""How a client should lay this tool's panel out. Presentation only."""

from . import catalogs

_INPUTS = "Inputs"
_LISTS = "Measurements"
_MODELS = "Models"
_REFERENCES = "References"

# A follow-up scan is only a question in a longitudinal study; an asymmetry
# assessment makes its own second scan. Showing the field in a study that
# ignores it is how a user comes to believe their scans were used.
_LONGITUDINAL = {"study": catalogs.STUDY_LONGITUDINAL}
_ASYMMETRY = {"study": catalogs.STUDY_ASYMMETRY}

# The modes that still have to orient, and the ones that still have to register.
_ORIENTS = {"mode": catalogs.MODE_FULL}
_REGISTERS = {"mode": [catalogs.MODE_FULL, catalogs.MODE_ORIENTED]}

# Measurements and heat maps need different things, and a panel that asked for
# both would ask for a 4 GB bundle nobody's run is going to use.
_MEASURES = {"outputs": [catalogs.OUTPUT_BOTH, catalogs.OUTPUT_QUANTITATIVE]}
_DRAWS = {"outputs": [catalogs.OUTPUT_BOTH, catalogs.OUTPUT_VISUALISATION]}

LAYOUT = {
    # Injected by the server for any tool that calls another, so neither is in
    # run()'s signature -- see describe.INJECTED_ARGUMENTS. The options are the
    # steps themselves; only the wording is this tool's to give, and it is
    # worth giving: "intermediate results" says nothing to a clinician, while
    # the name of a step plus what it hands back is what makes a chain of six
    # inspectable at all.
    #
    # Both were HIDDEN here until 2026-10-01, on the grounds that a box which
    # stops a cohort halfway was not worth leaving around until there was a
    # workflow to resume it. ASO and ALI_IOS have always shown them, so this
    # was VFACE being the exception rather than VFACE being protected -- and
    # it is the worst tool in the catalogue to make that exception for, for a
    # reason its chain's depth understates: it is the only one whose answer is
    # a VERDICT. A registration that lands on the wrong anatomy is visible in
    # the scan it returns; a classification reading "Asymmetric" looks exactly
    # the same whether the landmarks behind it were right or wrong, and the
    # one cell that carries it carries none of the working. These are the
    # points where the working can still be looked at.
    "keep_intermediate": {
        "label": "Steps to keep",
        # Named in the order `dispatch._run` runs them, which is not the order
        # the server lists them in -- the schema's `calls` arrives sorted. A
        # reader comparing the two should not read the difference as a claim
        # about the pipeline. `AutoMatrix` is called twice: the mirrored second
        # scan, then the landmarks carried into the registered frame.
        "option_help": {
            "ASO": "The scans once each sits in the cranial base and "
                   "maxillary frames, before a mask or a registration is "
                   "built on them.",
            "AMASSS": "The bone each region's registration is confined to -- "
                      "what to look at when a registration lands on the wrong "
                      "anatomy.",
            "AutoMatrix": "The mirrored scan an asymmetry assessment compares "
                          "the patient against, and the landmarks carried "
                          "into the registered frame.",
            "AREG_CBCT": "The registration itself, per region: the transform "
                         "every measurement is taken through.",
            "ALI_CBCT": "The landmarks the measurements are made on. A "
                        "measurement is no better than the point it is taken "
                        "from, and nothing downstream reveals a point that "
                        "landed badly.",
            "Batch_Dental_Seg": "The segmented surfaces the heat maps are "
                                "painted onto.",
        },
    },

    # `stop_after` is deliberately NOT named here. The server derives its
    # options from the chain, nested points included, and the generic label
    # reads correctly -- which is what ASO and ALI_IOS rely on by declaring
    # nothing for it either.
    "t1": {"section": _INPUTS, "label": "CBCT volumes"},
    "t2": {
        "section": _INPUTS, "label": "Follow-up CBCT volumes",
        "visible_when": _LONGITUDINAL,
    },
    "mode": {"section": _INPUTS, "label": "Starting from"},
    "study": {"section": _INPUTS, "label": "Study"},
    "outputs": {"section": _INPUTS, "label": "What to produce"},
    # Chips, the same as AMASSS's `structures` and AREG's `regions`: three lists
    # of anatomy across this family's panels should read alike, and a clinician
    # who ticks regions in AREG and then here is looking at one control, not two
    # spellings of it. No `groups` -- three options are not two kinds of thing.
    #
    # `option_help` gives each region its code, which names the workbook the
    # answer comes back in (`Measurements_CB.xlsx`) and prefixes every feature
    # column, so the panel says where to look for the result.
    "regions": {
        "section": _INPUTS, "label": "Regions to measure",
        "ui": "chips",
        "option_help": dict(catalogs.REGION_CODES),
    },

    "measurements": {
        "hidden": True,
        "section": _LISTS, "label": "Measurement lists (one per region)",
        "visible_when": _MEASURES,
    },
    "feature_template": {
        "hidden": True,
        "section": _LISTS, "label": "Feature list the classifier was trained on",
        "visible_when": _MEASURES,
    },
    "registration_transforms": {
        "section": _INPUTS, "label": "Transforms from a registration you already made",
        "visible_when": {"mode": catalogs.MODE_REGISTERED},
    },

    "cranial_base_reference": {
        "hidden": True,
        "section": _REFERENCES, "label": "Cranial base orientation reference",
        "visible_when": _ORIENTS,
    },
    "maxilla_reference": {
        "hidden": True,
        "section": _REFERENCES, "label": "Maxilla orientation reference",
        "visible_when": _ORIENTS,
    },
    "mirror_reference": {
        "hidden": True,
        "section": _REFERENCES, "label": "Mirror transform",
        "visible_when": _ASYMMETRY,
    },

    "segmentation_model": {
        "hidden": True,
        "section": _MODELS, "label": "Bone segmentation bundle",
        "visible_when": _REGISTERS,
    },
    "landmark_model": {
        "hidden": True,
        "section": _MODELS, "label": "CBCT landmark bundle",
        "visible_when": _MEASURES,
    },
    "classifier_model": {
        "hidden": True,
        "section": _MODELS, "label": "Asymmetry classifier bundle",
        "visible_when": _MEASURES,
    },
    "surface_model": {
        "hidden": True,
        "section": _MODELS, "label": "Surface segmentation bundle",
        "visible_when": _DRAWS,
    },
}
