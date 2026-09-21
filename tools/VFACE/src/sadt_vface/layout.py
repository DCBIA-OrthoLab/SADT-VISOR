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
    "t1": {"section": _INPUTS, "label": "CBCT volumes"},
    "t2": {
        "section": _INPUTS, "label": "Follow-up CBCT volumes",
        "visible_when": _LONGITUDINAL,
    },
    "mode": {"section": _INPUTS, "label": "Starting from"},
    "study": {"section": _INPUTS, "label": "Study"},
    "outputs": {"section": _INPUTS, "label": "What to produce"},
    "regions": {"section": _INPUTS, "label": "Regions to measure"},

    "measurements": {
        "section": _LISTS, "label": "Measurement lists (one per region)",
        "visible_when": _MEASURES,
    },
    "feature_template": {
        "section": _LISTS, "label": "Feature list the classifier was trained on",
        "visible_when": _MEASURES,
    },
    "registration_transforms": {
        "section": _INPUTS, "label": "Transforms from a registration you already made",
        "visible_when": {"mode": catalogs.MODE_REGISTERED},
    },

    "cranial_base_reference": {
        "section": _REFERENCES, "label": "Cranial base orientation reference",
        "visible_when": _ORIENTS,
    },
    "maxilla_reference": {
        "section": _REFERENCES, "label": "Maxilla orientation reference",
        "visible_when": _ORIENTS,
    },
    "mirror_reference": {
        "section": _REFERENCES, "label": "Mirror transform",
        "visible_when": _ASYMMETRY,
    },

    "segmentation_model": {
        "section": _MODELS, "label": "Bone segmentation bundle",
        "visible_when": _REGISTERS,
    },
    "landmark_model": {
        "section": _MODELS, "label": "CBCT landmark bundle",
        "visible_when": _MEASURES,
    },
    "classifier_model": {
        "section": _MODELS, "label": "Asymmetry classifier bundle",
        "visible_when": _MEASURES,
    },
    "surface_model": {
        "section": _MODELS, "label": "Surface segmentation bundle",
        "visible_when": _DRAWS,
    },
}
