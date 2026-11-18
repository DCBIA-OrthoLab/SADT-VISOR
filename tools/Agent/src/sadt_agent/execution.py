"""Running the tool that was chosen -- the one thing this tool does not do by default.

`execute` is off unless the caller turns it on. Upstream gated the same step
behind a `QMessageBox.question` showing the parameters and asking Yes/No
(`Agent/Agent.py:864-875`); a packaged tool has no dialog box, so that approval
had to become something a request carries rather than something that quietly
disappeared with the Qt.

## Why the call does not read `sup.run(...)`

`scripts/describe.py` reads a tool's source for `sup.run("<name>", ...)` and
publishes the names it finds as `calls`, so the server can refuse at startup a
tool that chains to one it does not serve. It requires the name to be a literal
or a module-level constant, and refuses the whole tool otherwise -- exit 2, not
loadable.

**A router has no such name.** Its callee is whatever the model picks out of the
catalogue it was handed, which is the live registry: the set of names it might
call IS the set the server serves, so there is nothing for a startup check to
verify. Writing `supervisor.run(...)` here rather than `sup.run(...)` is what
lets the schema generate; the parameter is named differently on purpose and this
comment is the record of it.

What is given up is the startup check, and what replaces it is a run-time one
that is strictly narrower: `catalog.find_tool` has already resolved the name
against the catalogue, so the only names that reach `supervisor.run` are names
the registry published in this same run. A tool that has been removed since the
catalogue was read fails in the supervisor with its own message naming the
missing venv.
"""

import logging
import os

from .errors import SupervisorRequired

logger = logging.getLogger("Agent")

# Where a supervised run's outputs land, inside this tool's own output
# directory. A subdirectory rather than the directory itself, so the routing
# decision and the run it produced never collide over a file name.
RUN_DIRECTORY = "run"

# The slice of this tool's bar the chosen tool fills. Passed to the supervisor
# as `_progress`, so the server folds the callee's own 0..1 into it and the
# clinician sees one bar rather than the router's and then the callee's. The
# routing before it is a few model round-trips, seconds against the minutes or
# the hour of whatever imaging tool it picks, so the run takes nearly all of it:
# starting the callee at 60%, as the waypoint used to, left the bar parked
# there for almost the whole job.
EXECUTION_SPAN = (0.1, 1.0)

# The argument types that name a file or a folder on this server, as the
# catalogue reads them (`catalog.canonical_type` folds the server's file types
# into these two).
PATH_TYPES = ("path", "list[path]")

# Names the server reads as a hosted bundle it resolves itself: `model`,
# `reference`, and anything ending in `_model` or `_reference`. The rule the
# server applies (VISOR-serve `registry/conventions.is_model`), restated here
# because a tool imports nothing from the server.
MODEL_NAMES = ("model", "reference")

# The job directory the server runs this tool in (VISOR-serve,
# `execution/dispatch.py`), inherited unchanged by a supervised call. Set means
# "on a server", where the folders are confinement roots only inside it.
JOB_DIR_ENV = "SADT_JOB_DIR"


def is_model_named(argument: str) -> bool:
    return any(argument == name or argument.endswith("_" + name) for name in MODEL_NAMES)


def confinement_refusal(tool, arguments: dict, folders) -> str:
    """Why `arguments` may not be run as they stand, or "" when they may.

    The server refuses a file argument sent as a path: a file has to be
    uploaded, or a workstation could point a tool at another run's inputs, at
    the server's own secrets, or at weights of its own making. What the model
    extracted reaches `supervisor.run` without passing that check -- a
    supervised call is not a request -- so a prompt that names a path, or a
    model that invents one, would have the callee read it. Two rules close
    that, checked on the values themselves:

    - a `path` argument must resolve, links followed, inside one of the
      folders this request was given: the only paths the model was shown;
    - a model-named argument must be a bare NAME, never a path. On a
      supervised call nothing resolves it against the server's hosted models
      on the way -- the callee gets the string as it was written, and resolves
      it from its own data root -- so a path there would load whatever weights
      sit at it, and weights are pickles. Not even a path inside the folders:
      those hold what the request is about, not code to run.

    The folders themselves are the CALLER's to name: a list is a plain value
    on the wire, not an upload, so whoever holds the API token chooses them,
    and `folders=["/"]` would make every path one "inside a folder". So on a
    server -- `SADT_JOB_DIR` set -- a folder is a root only if it resolves
    inside the run's own job directory, and any other is ignored: confined to
    no root, nothing runs. Outside a server the folders are the caller's own
    machine, and they stay the roots.

    The reason names the argument and never its value, which is a patient's
    file name.
    """
    roots = _confinement_roots(folders)
    for name, value in arguments.items():
        items = value if isinstance(value, (list, tuple)) else [value]
        if is_model_named(name):
            if not all(_is_bare_name(item) for item in items):
                return (
                    "'{}' names a model by a path. A model is given by its "
                    "name alone, among those the server hosts.".format(name)
                )
            continue
        declared = (tool.get("arguments", {}).get(name) or {}).get("type")
        if declared not in PATH_TYPES:
            continue
        if not all(_is_inside(item, roots) for item in items):
            return (
                "'{}' points outside the folders this request was given. Only "
                "a path inside one of them can be run, and on a server only a "
                "folder inside this run's own job counts.".format(name)
            )
    return ""


def _confinement_roots(folders) -> list:
    """The folders a path may lie in, links resolved: on a server, only those
    inside the job directory (see `confinement_refusal`)."""
    roots = [os.path.realpath(str(folder)) for folder in folders or ()
             if str(folder).strip()]
    job_dir = os.environ.get(JOB_DIR_ENV, "").strip()
    if not job_dir:
        return roots
    return [root for root in roots if _is_inside(root, [os.path.realpath(job_dir)])]


def _is_bare_name(value) -> bool:
    text = str(value)
    return (bool(text.strip()) and text not in (".", "..") and "\0" not in text
            and "/" not in text and "\\" not in text
            and not (os.altsep and os.altsep in text))


def _is_inside(value, roots) -> bool:
    """Whether `value`, every link resolved, is one of `roots` or under one.

    Resolved rather than compared as written: `<folder>/../elsewhere` and a
    link inside a folder pointing out of it both read as inside until they are.
    A relative path resolves against the directory this tool runs in, which is
    none of the folders, so it is refused too.
    """
    resolved = os.path.realpath(str(value))
    return any(os.path.commonpath([root, resolved]) == root for root in roots)


def invoke(supervisor, tool_name: str, arguments: dict, output_dir):
    """Run `tool_name` through the supervisor and return what it returned.

    `output_dir` is ours to give and never the model's to propose: it is taken
    out of the published schema before the model sees it, and filled here with a
    directory inside this run's own output.
    """
    if supervisor is None:
        raise SupervisorRequired(
            "`execute` was asked for, but nothing supplied a supervisor, so no "
            "tool can be run. Drop `execute` to get the proposal and run it "
            "yourself, or call this tool from a server that injects one."
        )

    destination = output_dir / RUN_DIRECTORY
    destination.mkdir(parents=True, exist_ok=True)

    parameters = dict(arguments)
    parameters["output_dir"] = str(destination)

    logger.info("running '%s' through the supervisor", tool_name)
    if hasattr(supervisor, "progress"):
        supervisor.progress(EXECUTION_SPAN[0], "running {}".format(tool_name))

    # `supervisor`, not `sup` -- see this module's docstring. The name is the
    # only thing that differs; this is an ordinary supervised call.
    produced = supervisor.run(tool_name, **parameters, _progress=EXECUTION_SPAN)

    return destination, _describe(produced)


def _describe(produced):
    """What the callee returned, as JSON-safe text.

    A tool returns a `Path` or a `dict[str, Path]`. Both are recorded, because
    the names in the dict are what a later chain would wire on.
    """
    if isinstance(produced, dict):
        return {key: str(value) for key, value in produced.items()}
    if produced is None:
        return None
    return str(produced)
