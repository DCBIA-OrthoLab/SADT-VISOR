"""The failures this tool raises on purpose, and who each one is addressed to.

These replace the server's `base.ToolArgumentError`, which a tool package cannot
import: nothing here knows the server exists. The server maps by the EXACT
exception class name, not by inheritance -- `ToolInputError`/`ValueError`/
`FileNotFoundError` to 422 with the message passed through,
`ToolUnavailableError` to 503, every other name to 500. A subclass therefore
does not inherit its parent's status: whatever must reach the caller as a 422
is raised as `ToolInputError` itself, and whatever is a deployment fault as
`ToolUnavailableError` itself. The subclasses below are for the faults that
are neither, where the class name is what tells the operator what broke.
"""


class ToolInputError(ValueError):
    """An argument the tool cannot work with, phrased for whoever sent it."""


class ToolUnavailableError(RuntimeError):
    """The tool is installed but the service it needs is not reachable.

    A separate class because the fix is a deployment one, not a request one: no
    argument the caller changes will make an unreachable Ollama endpoint or an
    unpulled model appear. The server answers 503 rather than 422.
    """


class SupervisorRequired(ToolInputError):
    """`execute` was asked for with no way to reach the chosen tool.

    Its own class because nothing about the request is wrong: either the caller
    drops `execute` and runs the proposal themselves, or whatever is running
    this tool has to inject a supervisor -- and a plain `uv run` never will.
    """


class CatalogError(RuntimeError):
    """The server's live registry answered, but not with a usable catalogue.

    A server fault, not the caller's: the caller sent no catalogue and the
    server's own `GET /tools` is what was malformed. A catalogue the CALLER
    passed as `catalog_file` is refused as a plain `ToolInputError` instead,
    and an unreachable registry is a `ToolUnavailableError`.
    """


class ModelAnswerError(RuntimeError):
    """The model answered, but not with the JSON object the call asked for.

    Not the caller's fault -- no argument they change makes the model produce
    JSON -- and not an unreachable service either: the endpoint is up and the
    model is loaded. It says the model in use is not following the prompt,
    which is for whoever chose `model_tag`'s default and deployed it.
    """


class RankingError(RuntimeError):
    """The candidate ranker could not produce a candidate set.

    Deliberately fatal. Upstream's ranker caught every exception and fell back
    to `scripts[:k]` -- the first three tools in file order -- while the router
    prompt went on saying "choose ONLY from the candidate list". With no network
    or a stale model cache, a registration request was offered landmarking and
    segmentation and nothing else, and the only sign was a line on stdout that
    also broke the JSON the caller parsed. A ranker that cannot rank must not
    narrow.

    A `RuntimeError` rather than an input error: the only way to reach it is an
    empty tool list, which `catalog.normalise` already refuses, so arriving here
    is an internal fault no argument the caller changes would fix.
    """
