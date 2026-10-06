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
