"""Agent -- route a free-text request to one of this server's tools.

Not an imaging tool. A local language model reads what a clinician typed, picks
one tool out of the server's own catalogue, and extracts that tool's parameters
from the same text. It **proposes**; it runs the chosen tool only when the
caller asks it to, with `execute`.

The catalogue is an input, and its default source is the live registry. See
`catalog.py` for why that replaced the hand-written manifest upstream carried.
"""

import json
import logging
import time
from pathlib import Path
from typing import Literal

from . import catalog, execution, llm, progress, ranking, routing
from .conversation import parse_history
from .errors import ToolInputError

logger = logging.getLogger("Agent")

# Where each routing stage starts on this tool's bar. The routing is a few model
# round trips before `execution.EXECUTION_SPAN` begins, so it shares the slice
# below that span's start; the numbers only have to rise, and the message is
# what tells the operator which round trip a stalled run is waiting on.
STAGE_CATALOGUE = 0.0
STAGE_FIRST_CALL = 0.02
STAGE_EXTRACTION = 0.06

# What the caller gets, and what a client renders.
DECISION_NAME = "routing.json"
ADVICE_NAME = "advice.md"
REPORT_NAME = "Agent_report.json"

MODE_AGENT = "Agent (Automated)"
MODE_ASK = "Ask (Interactive)"

__all__ = ["run"]


def run(
    prompt: str,
    output_dir: Path,
    catalog_file: Path = "",
    mode: Literal["Agent (Automated)", "Ask (Interactive)"] = MODE_AGENT,
    folders: list[Path] = [],
    history: str = "",
    candidates: int = 8,
    execute: bool = False,
    endpoint: str = "",
    model_tag: str = "qwen3:8b",
    temperature: float = 0.0,
    seed: int = 0,
    timeout_seconds: int = 300,
    *,
    sup=None,
) -> Path:
    """Route a free-text request to one of this server's tools.

    Args:
        prompt: What the user wants, in their own words. The whole routing
            decision is made from this and the folders below.
        output_dir: Where the decision is written -- `routing.json`, the run
            report, and in Ask mode `advice.md`. With `execute`, the chosen
            tool's own outputs land in `run/` inside it. Nothing is written
            outside it.
        catalog_file: A JSON file holding what the server's `GET /tools`
            returns: every tool, its arguments, their types, whether they are
            required, and their choices. Leave it empty and the live registry is
            read instead, from `SADT_API` -- which is the point of the whole
            design: the catalogue is generated from each tool's own signature by
            that tool's own interpreter, so it cannot drift from what the tools
            actually take.
        mode: "Agent (Automated)" picks a tool and fills its parameters.
            "Ask (Interactive)" answers with methodology advice and never
            proposes a run, whatever `execute` says.
        folders: The folders holding the data this request is about. They are
            the only paths the model is shown, and it is told to use them
            verbatim. Sent as a list, so a folder whose name contains a comma
            stays one folder.
        history: Earlier turns of this conversation, as a JSON array of
            `{"role": "user"|"assistant"|"system", "content": "..."}`, so a
            follow-up that only supplies a missing parameter still makes sense.
            Empty means a fresh conversation; anything that is not that shape is
            an error rather than a silently forgotten conversation.
        candidates: How many tools the model is shown, best first. 0 means all
            of them. The ranker only narrows when it has signal: a request that
            matches nothing gets the whole catalogue rather than an arbitrary
            slice of it.
        execute: Run the chosen tool, through the supervisor, once the proposal
            is complete. Off by default and deliberately so -- a model choosing
            what to run on patient data unattended is a decision a person should
            make. It is refused anyway when no tool was chosen, when a required
            argument is missing, or when the router's own confidence is below
            0.5.
        endpoint: The Ollama server to talk to, e.g. `http://127.0.0.1:11434`.
            Empty means `OLLAMA_HOST`, then loopback. The endpoint is an
            operational prerequisite: this tool does not install Ollama, does
            not start it, and does not pull models.
        model_tag: The Ollama model to route with, e.g. `qwen3:8b`. It must
            already be pulled on that server. Called a tag rather than a model
            because it is neither a file nor a bundle this server hosts: an
            argument named `model` is published as a name picked from
            `DATA/<tool>/models/`, and this is a string an Ollama server
            resolves.
        temperature: Sampling temperature. 0 makes the same request route to the
            same tool twice, which is what lets a decision be checked.
        seed: The sampler's seed, so a run above temperature 0 is reproducible.
        timeout_seconds: How long to wait for one model call, and for reading
            the live registry.

    Returns:
        The output directory, holding `routing.json` -- the decision, as
        `{tool, arguments, confidence, reasoning, alternatives}` -- the run
        report, and whatever `execute` produced.
    """
    started = time.monotonic()
    output_dir = Path(output_dir)

    if not str(prompt or "").strip():
        raise ToolInputError(
            "`prompt` is empty. There is nothing to route without a request."
        )
    if candidates < 0:
        raise ToolInputError(
            "`candidates` is {}. Use 0 to show the model every tool.".format(candidates)
        )

    turns = parse_history(history)
    folder_list = [str(folder) for folder in (folders or []) if str(folder).strip()]

    progress.emit(STAGE_CATALOGUE, "reading catalogue")
    began = time.monotonic()
    tools, source = catalog.load_catalog(catalog_file, timeout_seconds)
    logger.info(
        "catalogue read: %d tools from the %s in %.1f s", len(tools),
        "live registry" if source.startswith("registry:") else "catalog_file",
        time.monotonic() - began,
    )
    selected, scores, narrowed = ranking.select_candidates(tools, prompt, candidates)

    resolved_endpoint = llm.resolve_endpoint(endpoint)
    logger.info(
        "Agent: mode=%s, catalogue=%s (%d tools), candidates=%d%s, model=%s",
        mode, source, len(tools), len(selected),
        "" if narrowed else " (unnarrowed)", model_tag,
    )

    def client(messages, json_format, call="model call"):
        # Timed here rather than in `llm.chat` so a stubbed model is timed the
        # same way, and logged on failure too: "failed after 300.0 s" and
        # "failed after 0.0 s" are two different problems.
        began = time.monotonic()
        try:
            answer = llm.chat(
                resolved_endpoint, model_tag, messages,
                json_format=json_format, temperature=temperature, seed=seed,
                timeout_seconds=timeout_seconds, call=call,
            )
        except Exception as exc:
            logger.warning(
                "%s failed after %.1f s (%s)", call, time.monotonic() - began,
                type(exc).__name__,
            )
            raise
        logger.info("%s answered in %.1f s", call, time.monotonic() - began)
        return answer

    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "tool": "Agent",
        "mode": mode,
        "catalog_source": source,
        "catalog_size": len(tools),
        "candidates": [tool["name"] for tool in selected],
        "candidates_narrowed": narrowed,
        "ranker_scores": scores,
        "endpoint": resolved_endpoint,
        "model_tag": model_tag,
        "temperature": temperature,
        "seed": seed,
        "history_turns": len(turns),
        "warnings": [],
    }

    if mode == MODE_ASK:
        decision = _ask(client, prompt, selected, turns, output_dir, report)
    else:
        decision = _route(
            client, prompt, selected, folder_list, turns, scores,
            execute, sup, output_dir, report,
        )

    report["duration_seconds"] = round(time.monotonic() - started, 2)
    _write(output_dir / DECISION_NAME, decision)
    _write(output_dir / REPORT_NAME, report)
    return output_dir


def _ask(client, prompt, selected, turns, output_dir, report):
    """Advice only. Nothing is proposed and nothing is run."""
    progress.emit(STAGE_FIRST_CALL, "advice call")
    advice = routing.advise(client, prompt, selected, turns)
    (output_dir / ADVICE_NAME).write_text(advice, encoding="utf-8")
    report["advice_characters"] = len(advice)
    return {
        "tool": None,
        "arguments": {},
        "confidence": None,
        "reasoning": advice,
        "alternatives": _alternatives(selected, None, report["ranker_scores"]),
        "mode": MODE_ASK,
        "advice_file": ADVICE_NAME,
    }


def _route(client, prompt, selected, folder_list, turns, scores,
           execute, sup, output_dir, report):
    """Pick a tool, fill its parameters, and run it if that was asked for."""
    progress.emit(STAGE_FIRST_CALL, "router call")
    tool, confidence, reasoning, rejected = routing.choose_tool(
        client, prompt, selected, folder_list, turns
    )
    if rejected:
        report["warnings"].append(
            "The router named '{}', which is not one of the {} candidates it was "
            "shown, so no tool was chosen.".format(rejected, len(selected))
        )

    decision = {
        "tool": tool["name"] if tool else None,
        "arguments": {},
        "confidence": round(confidence, 3),
        "reasoning": reasoning,
        "alternatives": _alternatives(selected, tool, scores),
        "mode": MODE_AGENT,
        "missing_required": [],
        "errors": [],
    }
    if rejected:
        decision["rejected_tool_name"] = rejected
    if tool is None:
        decision["errors"].append(
            "The router named '{}', which is not a tool it was offered.".format(rejected)
            if rejected else
            "No candidate tool matches this request."
        )
        decision["executed"] = False
        _log_outcome(decision)
        return decision

    progress.emit(STAGE_EXTRACTION, "extraction call (tool {})".format(tool["name"]))
    values, parameter_confidence, errors, unknown = routing.extract_arguments(
        client, prompt, tool, folder_list, turns
    )
    decision["arguments"] = values
    decision["parameter_confidence"] = round(parameter_confidence, 3)
    decision["missing_required"] = catalog.missing_required(tool, values)
    decision["errors"] = errors
    if unknown:
        decision["dropped_arguments"] = unknown
    _log_outcome(decision)

    allowed, reason = routing.can_execute(decision)
    if not execute:
        decision["executed"] = False
        return decision
    if not allowed:
        decision["executed"] = False
        decision["execution_refused"] = reason
        report["warnings"].append("Not executed: " + reason)
        logger.warning("not executing '%s': %s", tool["name"], reason)
        return decision

    produced_dir, produced = execution.invoke(sup, tool["name"], values, output_dir)
    decision["executed"] = True
    decision["execution"] = {
        "tool": tool["name"],
        "output": produced_dir.name,
        "returned": produced,
    }
    return decision


def _log_outcome(decision):
    """One line an operator can read the whole routing decision from.

    Counts rather than contents: the errors and the missing arguments are in
    `routing.json`, and their text can quote the request.
    """
    logger.info(
        "routing outcome: tool=%s, confidence=%.2f, %d error(s), "
        "%d required argument(s) missing",
        decision["tool"] or "none", decision["confidence"],
        len(decision["errors"]), len(decision["missing_required"]),
    )


def _alternatives(selected, chosen, scores):
    """The other candidates, with the score that put them there.

    The ranker's alternatives, not the model's: the model answers with one name
    and a sentence. Naming them lets a person see what was on the table without
    running the request again.
    """
    chosen_name = chosen["name"] if chosen else None
    return [
        {"tool": tool["name"], "score": scores.get(tool["name"], 0.0)}
        for tool in selected
        if tool["name"] != chosen_name
    ]


def _write(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
