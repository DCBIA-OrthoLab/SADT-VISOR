"""How a client lays this tool's panel out. Presentation only.

Everything here is DERIVED from `catalog`, never restated: the structure codes,
their readable names and their grouping all already exist there because the
pipeline needs them, and a second copy would drift.
"""

from . import catalog

# Code -> the words behind it. `MAND` and `CBMASK` tell a clinician nothing, and
# the alternative was a 452-character paragraph beside the argument spelling out
# all nine at once -- which every reader had to scan to find the one they were
# hovering.
_STRUCTURE_HELP = {
    code: name
    for group in catalog.STRUCTURE_GROUPS.values()
    for name, code in group.items()
}

_MERGE_HELP = {
    "MERGED": "One multi-label file per scan, every structure in it",
    "SEPARATE": "One binary file per structure",
}

LAYOUT = {
    # One bundle ships, so choosing it is a dropdown with a single entry and a
    # way to get it wrong. The server hands a hidden hosted-model argument the
    # whole of `DATA/AMASSS/models/`, and `pipeline.resolve_models` already
    # descends into a bundle wrapped in one top-level folder -- which is exactly
    # that shape. Left empty, `run()` resolves the same bundle from the data
    # root itself, which is what a neighbour calling through the supervisor
    # relies on.
    "model": {"hidden": True},
    "structures": {
        # Chips rather than a column of check boxes, and no tabs: nine options
        # fit on a couple of lines, so hiding them behind a tab bar would cost a
        # click and show nothing more. The headings are the catalog's own
        # grouping -- bones, soft tissue and the three masks are not the same
        # kind of thing, and a mask ticked by mistake is a wasted run.
        "ui": "chips",
        "groups": {
            group: list(codes.values())
            for group, codes in catalog.STRUCTURE_GROUPS.items()
        },
        "option_help": _STRUCTURE_HELP,
        # `pipeline` refuses a run with none of these, so the panel says so
        # before Apply rather than after a request.
        "min_selected": 1,
    },
    "merge": {
        # Two options and no grouping: a plain pair of check boxes says it, and
        # both may be ticked at once, which is exactly what a check box means
        # and what a pair of chips would not.
        "option_help": _MERGE_HELP,
        # A run has to produce SOMETHING. Same rule, same reason.
        "min_selected": 1,
    },
}
