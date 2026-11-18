"""How a client lays the panel out. Presentation only.

Nothing here can change what `run()` accepts: `describe.py` refuses a layout
naming an argument the signature does not publish, which is what keeps this from
drifting the way the old hand-written argument tables did.

Two boxes. The first is what a clinician fills in; the second is the generation
settings, which have working defaults and exist so a run can be reproduced and
audited rather than so anyone has to choose them.
"""

INPUT = "Notes"
GENERATION = "Generation"

LAYOUT = {
    "notes": {"section": INPUT, "label": "Clinical notes"},
    "notes_type": {"section": INPUT, "label": "Note type"},
    "model": {"section": INPUT, "label": "Model"},
    "max_tokens": {"section": GENERATION, "label": "Longest answer (tokens)"},
    "temperature": {"section": GENERATION, "label": "Temperature"},
    "context_tokens": {"section": GENERATION, "label": "Context window (tokens, 0 = auto)"},
    "seed": {"section": GENERATION, "label": "Seed"},
    "device": {"section": GENERATION, "label": "Device"},
}
