"""Pick a tool, then fill its parameters. Two model calls, both post-checked.

The router's answer is verified against the candidate list rather than trusted:
"choose ONLY from the candidates" is an instruction, and an instruction is not a
type. A name that matches no candidate is recorded and the decision becomes "no
tool", which is a refusal a person can read -- upstream passed it straight into
`build_cli_args`, where it raised a `KeyError` about a manifest.
"""

import logging

from . import catalog, llm, prompts, validation

logger = logging.getLogger("Agent")

# Below this the proposal is still returned -- it is a proposal -- but `execute`
# refuses to act on it. The number is deliberately low: it is a floor against a
# model that has said in its own answer that it is guessing, not a quality bar.
EXECUTE_MIN_CONFIDENCE = 0.5


def choose_tool(client, prompt, candidates, folders, history):
    """`(tool, confidence, reasoning, rejected)` from the router call."""
    messages = (
        [{"role": "system", "content": prompts.ROUTER_SYSTEM}]
        + list(history)
        + [{
            "role": "user",
            "content": prompts.router_user(
                prompt, candidates, folders, catalog.fillable_arguments
            ),
        }]
    )
    answer = llm.parse_json_object(
        client(messages, json_format=True, call="router call"), "routing"
    )

    name = answer.get("tool")
    confidence = _confidence(answer.get("confidence"))
    reasoning = str(answer.get("reason") or answer.get("reasoning") or "").strip()

    # `"null"` and `"None"` as strings: qwen3 emits both, and upstream had the
    # same two special cases (`Agent_CLI.py:145`).
    if not name or str(name).strip().lower() in ("null", "none"):
        return None, confidence, reasoning, None

    chosen = catalog.find_tool(candidates, str(name))
    if chosen is None:
        # Not an exception: a router that names something outside its list has
        # answered "none of these" badly, and the useful report says exactly
        # that, with what it said.
        logger.warning(
            "the router named %r, which is not among the %d candidates",
            name, len(candidates),
        )
        return None, min(confidence, 0.2), reasoning, str(name)
    return chosen, confidence, reasoning, None


def extract_arguments(client, prompt, tool, folders, history):
    """`(values, confidence, errors, unknown)` from the extraction call."""
    arguments = catalog.fillable_arguments(tool)
    if not arguments:
        return {}, 1.0, [], []

    messages = (
        [{"role": "system", "content": prompts.EXTRACTOR_SYSTEM}]
        + list(history)
        + [{
            "role": "user",
            "content": prompts.extractor_user(prompt, tool, arguments, folders),
        }]
    )
    answer = llm.parse_json_object(
        client(
            messages, json_format=True,
            call="extraction call (tool {})".format(tool["name"]),
        ),
        "extraction",
    )

    values, errors, unknown = validation.validate(
        tool, arguments, answer.get("extracted", {})
    )
    confidence = _confidence(answer.get("confidence"))
    if errors:
        # Upstream multiplied by 0.6 for any validation failure. Kept, and kept
        # a single factor rather than a per-error one: this number is the
        # model's own self-report, and refining it past "something did not
        # check out" would read as a measurement.
        confidence *= 0.6
    return values, confidence, errors, unknown


def advise(client, prompt, candidates, history) -> str:
    """Ask mode: methodology advice, and nothing is ever run."""
    messages = (
        [{
            "role": "system",
            "content": prompts.advisor_system(candidates, catalog.fillable_arguments),
        }]
        + list(history)
        + [{"role": "user", "content": prompt}]
    )
    # Not `format="json"`: this answer is prose for a person to read. Upstream
    # then stripped every `*` and `#` from it (`Agent_CLI.py:215-216`) to make
    # it look like plain text in a Qt label; the answer is written to a `.md`
    # file here, so the Markdown is kept.
    return client(messages, json_format=False, call="advice call").strip()


def can_execute(decision):
    """`(allowed, reason)`: whether this proposal may be run as it stands.

    Every refusal below is a decision a person would otherwise have made in
    front of a dialog box. Upstream asked with a `QMessageBox.question`; a
    packaged tool has no dialog box, so the approval became the `execute`
    argument and these are the checks that argument does not override.
    """
    if not decision.get("tool"):
        return False, "no tool was chosen, so there is nothing to run"
    if decision.get("missing_required"):
        return False, "required argument(s) not supplied: {}".format(
            ", ".join(decision["missing_required"])
        )
    if decision.get("confidence", 0.0) < EXECUTE_MIN_CONFIDENCE:
        return False, (
            "the router's confidence is {:.2f}, below the {:.2f} needed to run "
            "a tool unattended".format(
                decision.get("confidence", 0.0), EXECUTE_MIN_CONFIDENCE
            )
        )
    return True, ""


def _confidence(value) -> float:
    """A number in [0, 1], whatever the model answered with."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number != number:  # NaN
        return 0.0
    return max(0.0, min(1.0, number))
