"""How a client should lay this tool's panel out. Presentation only.

Nothing here changes what `run()` accepts -- `describe.py` merges these hints
into the published schema and refuses any that name an argument or an option
the signature does not offer. Delete this file and the tool still works; the
panel just gets worse.

**Everything is DERIVED, never restated.** That is the whole difference from the
`ArgSpec` tables this replaces. Those listed the anatomical tabs by hand, and a
landmark added to the catalog was then reachable through no tab at all -- offered
by the schema, invisible in the UI. Here the tabs are computed from
`catalog.GROUP_LABELS`, so a landmark added there appears in its tab with no
edit here and no client release.

Why it is needed at all: this tool publishes **119** landmark options. A schema
that says "119 strings, pick some" is honest and produces a panel nobody can
use.

No `visible_when` anywhere, and that is the split showing through. It used to
carry seven rules whose whole job was hiding the intraoral half of a shared
schema; there is no intraoral half here any more, so every argument in this
panel applies to every run of it.
"""

from . import catalog

_INPUTS = "Inputs"
_LANDMARKS = "Landmarks"
_OUTPUTS = "Outputs"

LAYOUT = {
    "input": {"section": _INPUTS, "label": "Scan or Folder"},
    # Not rendered. This engine IS the CBCT one: there is exactly one bundle it
    # can use, and asking a clinician which was asking a question with one
    # answer. Worse, the facade composes ONE dropdown for both engines with
    # nothing to tell them apart, so picking the intraoral bundle for a CBCT run
    # was one click away -- and it happened.
    #
    # Left out of the request, the server hands this tool its models DIRECTORY
    # and `engine.discover_weights` finds its own bundle inside: the CBCT layout
    # is `<landmark>/<scale>/*.pth` and matches nothing the intraoral bundle
    # holds. The argument stays required in run(), so a direct call with no
    # server at all still works exactly as before.
    "model": {"section": _INPUTS, "label": "Model Bundle", "hidden": True},
    # Not rendered, and this is the ONE hint here that is not about looks.
    # `SlicerAutomatedDentalTools` never offered a region selection: its
    # `ALI.ui` contains the word "region" zero times, and the anatomical groups
    # appear only as the NAMES OF THE TABS over the landmarks. A clinician
    # coming from that extension is looking for a control that was never there,
    # beside a landmark list that already carries the same grouping.
    #
    # Hiding it costs nothing it could do: `run()` keeps its default (every
    # region on), and asking for one region is now one click on that region's
    # tab -- which sends the landmarks explicitly, exactly as the original did
    # (`LandmarkTabWidget.GetSelected`, a list of landmark names).
    "regions": {"section": _LANDMARKS, "label": "Regions", "ui": "inline",
                "hidden": True},
    "landmarks": {
        # One line per landmark, so hovering `UR6MB` says which tooth and where
        # on it. Sourced from Gillot et al. 2023 (PMC10440369) Table 1; a
        # landmark the paper does not reach carries no line rather than a
        # guessed one.
        "option_help": catalog.DESCRIPTIONS,
        "section": _LANDMARKS,
        "label": "Individual landmarks",
        # 119 check boxes in one column is a scroll, not a choice. The tabs are
        # the SAME grouping the engine names its output files by, published
        # rather than restated.
        "ui": "tabs",
        "groups": {
            display: list(catalog.GROUP_LABELS[code])
            for display, code in catalog.REGION_NAMES.items()
        },
    },
}
