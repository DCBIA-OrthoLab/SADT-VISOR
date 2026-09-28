"""How a client should lay this tool's panel out. Presentation only.

Nothing here changes what `run()` accepts -- `describe.py` merges these hints
into the published schema and refuses any that name an argument or an option
the signature does not offer. Delete this file and the tool still works; the
panel just gets worse.

**Everything is DERIVED, never restated.** The tabs over the landmarks are
computed from `catalog.LANDMARK_GROUPS`, so a landmark added to the catalog
appears in its tab with no edit here and no client release -- the failure the
old hand-written tables produced was a landmark the schema offered and no tab
could reach.

No `visible_when` anywhere. Before the split this panel shared a schema with
119 CBCT landmark options and needed a rule on every field to stop an intraoral
user scrolling past them; every argument published here now applies to every
run of it.
"""

from . import catalog

_INPUTS = "Inputs"
_LANDMARKS = "Landmarks"
_OUTPUTS = "Outputs"

LAYOUT = {
    # Injected by the server for any tool that calls another, so it is not in
    # run()'s signature -- see describe.INJECTED_ARGUMENTS.
    # Injected by the server for any tool that calls another, so it is not in
    # run()'s signature -- see describe.INJECTED_ARGUMENTS.
    "keep_intermediate": {
        "label": "Steps to keep",
        "option_help": {
            "Crown_Seg": "The per-tooth labelled meshes, which this tool "
                         "otherwise only uses to place the landmarks.",
        },
    },
    "input": {"section": _INPUTS, "label": "Surface or Folder"},
    # Not rendered, for the same reason as ALI_CBCT's: this engine IS the
    # intraoral one and there is exactly one bundle it can use. Left out of the
    # request, the server hands the tool its models DIRECTORY and
    # `engine.discover_weights` finds its own checkpoints inside -- flat `.pth`
    # carrying a network and a jaw token, which matches nothing in the CBCT
    # bundle's `<landmark>/<scale>/` tree. Required in run() either way, so a
    # direct call with no server still works.
    "model": {"section": _INPUTS, "label": "Model Bundle", "hidden": True},
    # Not rendered, and this is the one hint here that is not about looks.
    # The landmark table below IS this selection, made one point at a time, and
    # two controls over the same thing is one too many: with both on screen the
    # family boxes go on deciding a run whose landmarks were already named,
    # which reads as the panel ignoring the clinician. It is also closer to
    # what `SlicerAutomatedDentalTools` offered: two lists, teeth and landmark
    # types, whose cross-product IS the table below -- asked for once instead
    # of twice, and without the combinations no network predicts.
    #
    # Hiding it costs nothing it could do: `run()` keeps its default, so a
    # caller driving this package directly still selects by family (AREG asks
    # for Mucogingival alone, which is the pass nobody else wants).
    "networks": {"section": _LANDMARKS, "label": "Landmark families",
                 "ui": "inline", "hidden": True},
    "landmarks": {
        "section": _LANDMARKS,
        "label": "Individual landmarks",
        # 153 check boxes in one column is a scroll, not a choice. The tabs are
        # the catalog's own family-by-jaw grouping, published rather than
        # restated -- and an intraoral scan being one arch, the tab a clinician
        # opens is usually the only one they need.
        "ui": "tabs",
        # One line per landmark, so hovering `UR1MB` says which tooth and where
        # on it. Derived in the catalog from the table the engine itself indexes
        # by, never from the label's spelling -- the mucogingival names are
        # positional and the midline one shifts the right side by a tooth.
        "option_help": catalog.DESCRIPTIONS,
        "groups": {
            name: list(labels) for name, labels in catalog.LANDMARK_GROUPS.items()
        },
    },
}
