#!/usr/bin/env python3
"""Run a tool from the command line, in its own venv, chaining to others.

    scripts/run_tool.py ALI --input scan.nii.gz --model bundle/ --output-dir out/
    scripts/run_tool.py ASO --input cohort/ --reference gold/ --output-dir out/ \
        --automation Fully-Automated --landmark-models ALI_CBCT_Models

One script for every tool: the parser is built from `run()`'s signature, so there
is no per-tool CLI to keep in step with anything. A tool that takes a supervisor
gets one, and its `sup.run("ALI", ...)` re-enters THIS script with ALI's
interpreter -- which is what makes chaining work, and nesting with it:
`AREG -> ASO -> ALI` is three levels of the same recursion, no special case.

**Developer convenience, not the deployment path.** In production the server's
runner invokes tools and sequences them; this exists so the same thing can be
done from a checkout, with no server, no Docker and no network. It is also the
reference a supervisor implementation can be read off.

Two layers, one file:

1. any interpreter -- resolve `tools/<name>/.venv/bin/python` and re-exec;
2. the tool's own venv -- import the package, build the parser, call `run()`.

It also sets `SADT_PROGRESS_FILE`, which is how the SERVER asks a tool for
progress, and echoes every line a tool appends to it. A tool's progress
reporting is therefore exercised from a checkout, on the same channel it will
use in production, rather than only being discovered to be silent once it is
deployed.

Stdlib only and Python 3.9 compatible, for the same reason `describe.py` is: it
has to run inside whichever interpreter a tool pinned. The annotation vocabulary
is imported FROM describe.py rather than restated, so the CLI can never offer an
option the schema does not, or refuse one it does.

It is also the supervisor a desktop client embeds -- the 3D Slicer extension
runs tools through it on the clinician's machine -- so the same code chains
tools on Linux, macOS and Windows, from a terminal or from a GUI:

- `SADT_TOOLS_DIR` points it at a folder of tools other than this checkout's
  `tools/` (a client's download cache), with the same layout;
- `SADT_DATA_DIR` is the DATA root a tool finds its own models under, injected
  as `data_root` exactly as the server injects it;
- `start(tool, params, job_dir)` launches a run without blocking and returns a
  `Job` to poll for progress, read the result from, or cancel -- the whole
  chain of processes below it at once;
- a failure is written to the result file as `{"kind": "error", "type": ...,
  "message": ...}`, the callee's own type and message carried up every level
  of a chain, so the caller can show the reason rather than "see the log".
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import signal
import tempfile
import threading
import time
import traceback
import typing
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS.parent

# Where the tools live. A client that downloads them on demand keeps them in a
# cache of its own and names it here; every nested call inherits the variable,
# so a whole chain resolves its tools in the same place.
TOOLS_DIR_VARIABLE = "SADT_TOOLS_DIR"
TOOLS_DIR = (
    Path(os.environ[TOOLS_DIR_VARIABLE]).resolve()
    if os.environ.get(TOOLS_DIR_VARIABLE)
    else REPO_ROOT / "tools"
)

# The DATA root: `<root>/<tool>/models/...`, the layout `scripts/fetch_data.py`
# writes and the server reads. A tool that declares `data_root` is handed it, as
# the server's runner does, to find ITS OWN models -- so a caller never names
# its neighbour's. Unset, nothing is injected and such a tool asks for its model
# by name.
DATA_DIR_VARIABLE = "SADT_DATA_DIR"

# The chain of tools already running above a call, comma-separated, and how
# deep it may grow. The same variable and the same cap as the server's runner,
# so a chain refused there is refused here, by name, before it starts a process.
SUPERVISOR_CHAIN_VARIABLE = "SADT_SUPERVISOR_CHAIN"
MAX_SUPERVISOR_DEPTH = 10

# The interface a desktop client drives this supervisor through: `start()`, the
# `Job` it returns, the result file and its error record, SADT_TOOLS_DIR and
# SADT_DATA_DIR. A client reads it from a release before offering that release
# as an update, and offers only the levels it knows. Raise it in the commit that
# changes that interface in a way an existing client cannot follow; adding
# something a client may ignore does not need it.
#   1  the command line only (before `start()`)
#   2  start() / Job, structured errors, SADT_TOOLS_DIR, SADT_DATA_DIR
SUPERVISOR_API = 2

sys.path.insert(0, str(SCRIPTS))
import describe  # noqa: E402  -- one annotation vocabulary, not two


class ToolNotBuilt(Exception):
    """The tool has no venv yet, and the message says how to make one."""


class ToolFailed(RuntimeError):
    """A supervised tool failed. Raised under the CALLEE's own class name.

    The server maps a failure by its class name, and a clinician reads the
    message: re-raising a nested failure as a generic error would lose both.
    `failure_class` builds a subclass named after the callee's exception, so
    `ToolInputError` raised three levels down is still `ToolInputError` here.
    """

    tool = None


def failure_class(type_name):
    """A `ToolFailed` subclass named `type_name`, the callee's exception class."""
    name = type_name if str(type_name).isidentifier() else "ToolFailed"
    return type(name, (ToolFailed,), {})


def write_result(path, payload):
    """Write a result file atomically: serialized first, then moved into place.

    A reader polling for it -- a GUI waiting on a run -- must never see half of one.
    """
    path = Path(path)
    body = json.dumps(payload)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(body, encoding="utf-8")
    os.replace(str(temporary), str(path))


def error_payload(error):
    return {"kind": "error", "type": type(error).__name__, "message": str(error)}


# ---------------------------------------------------------------------------
# Layer 1: find the tool's interpreter
# ---------------------------------------------------------------------------

def tool_root(name):
    """The folder of a tool: `tools/<name>`, or `tools/<group>/<name>` for a tool that lives in a grouping
    folder (`ALI_CBCT` is `tools/ALI/ALI_CBCT`). A path such as `ALI/ALI_CBCT` is taken as given. This is how
    the server names a tool, so a tool that calls another one by its name finds it here as well."""
    direct = TOOLS_DIR / name
    if (direct / "pyproject.toml").is_file():
        return direct
    grouped = sorted(path.parent for path in TOOLS_DIR.glob("*/" + name + "/pyproject.toml"))
    return grouped[0] if grouped else direct


def tool_python(name):
    """The interpreter of the tool's own `.venv`."""
    root = tool_root(name)
    if not root.is_dir():
        available = sorted(
            path.name for path in TOOLS_DIR.iterdir()
            if (path / "pyproject.toml").is_file()
        )
        raise ToolNotBuilt(
            "There is no tool called {!r} under tools/. Available: {}.".format(
                name, ", ".join(available) or "none"
            )
        )
    for relative in ("bin/python", "Scripts/python.exe"):
        candidate = root / ".venv" / relative
        if candidate.is_file():
            return candidate
    raise ToolNotBuilt(
        "tools/{0} has no .venv. Build it with `cd tools/{0} && uv sync`.".format(name)
    )


# ---------------------------------------------------------------------------
# Layer 2: the parser, built from run()'s signature
# ---------------------------------------------------------------------------

# Above this many, an argument's options are validated but not spelled into the
# usage line -- ALI's `landmarks` has 119 and argparse prints them all.
_MAX_LISTED_CHOICES = 8


def option_name(argument):
    """`output_dir` -> `--output-dir`."""
    return "--" + argument.replace("_", "-")


def add_argument(parser, name, parameter, hints):
    """One `run()` parameter as one command-line option.

    Everything the schema can express has a command-line spelling, and nothing
    else does: `describe.type_name` decides, so an annotation the server would
    refuse cannot be smuggled in through here.
    """
    kind, choices = describe.type_name(hints[name], "argument '{}'".format(name))
    required = parameter.default is parameter.empty
    default = None if required else parameter.default
    flag = option_name(name)

    if kind == "bool":
        # A flag and its negation, both present, because a tool's default may be
        # either -- `--no-skip-segmented` has to exist for `skip_segmented=True`.
        group = parser.add_mutually_exclusive_group(required=required)
        group.add_argument(flag, dest=name, action="store_true", default=default)
        group.add_argument(
            option_name("no_" + name), dest=name, action="store_false", default=default
        )
        return

    scalar = {"path": str, "str": str, "int": int, "float": float, "bool": str}
    # A long option list is still VALIDATED, just not printed: ALI publishes 119
    # landmarks, and argparse spells every one of them into the usage line.
    metavar = name.upper() if (not choices or len(choices) > _MAX_LISTED_CHOICES) else None
    help_text = (
        "one of: {}".format(", ".join(str(c) for c in choices))
        if choices and metavar
        else None
    )

    if kind.startswith("list["):
        element = kind[len("list["):-1]
        parser.add_argument(
            flag, dest=name, nargs="*", required=required, default=default,
            type=scalar[element], choices=choices, metavar=metavar, help=help_text,
        )
        return

    parser.add_argument(
        flag, dest=name, required=required, default=default,
        type=scalar[kind], choices=choices, metavar=metavar, help=help_text,
    )


def build_parser(run, name):
    signature, hints = describe.inspect.signature(run), typing.get_type_hints(run)
    parser = argparse.ArgumentParser(
        prog="run_tool.py {}".format(name),
        description=(describe.inspect.getdoc(run) or "").strip().splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    for argument, parameter in signature.parameters.items():
        if describe.is_supervisor(argument, parameter, hints):
            continue  # injected, never typed by a human
        add_argument(parser, argument, parameter, hints)
    return parser


def takes_injected(run, argument):
    """Whether `run()` declares the injected `argument`, by describe.py's own rule.

    By NAME: `describe.is_supervisor` answers for every injected argument, so
    asking it of each parameter in turn would take a tool that declares only
    `data_root` for one that wants a supervisor.
    """
    parameters = describe.inspect.signature(run).parameters
    if argument not in parameters:
        return False
    return describe.is_supervisor(argument, parameters[argument], typing.get_type_hints(run))


def takes_supervisor(run):
    """Whether `run()` declares a supervisor."""
    return takes_injected(run, describe.SUPERVISOR)


def takes_data_root(run):
    """Whether `run()` declares the DATA root it finds its own models under."""
    return takes_injected(run, describe.DATA_ROOT)


def coerce(value, annotation):
    """JSON has no path type, so paths travel as strings and become `Path` here.

    The same coercion `sadt_testkit._driver` and the server's runner apply. If
    the three ever disagree, a chain that works from the CLI breaks in
    production, so keep this boring and keep it matching.

    An empty string is ABSENCE and stays a string. `Path("")` is
    `PosixPath(".")` -- the current directory, and truthy -- so coercing the
    "not supplied" default of an optional path turns it into a real directory
    the tool then walks. Calling `run()` directly in Python keeps the `""` the
    signature declares, so coercing it here is also what makes the two paths
    disagree.
    """
    if annotation is Path:
        return Path(value) if value != "" else value
    if typing.get_origin(annotation) is list and typing.get_args(annotation) == (Path,):
        return [Path(item) for item in value]
    return value


# ---------------------------------------------------------------------------
# Progress: the same channel the server uses
# ---------------------------------------------------------------------------

PROGRESS_VARIABLE = "SADT_PROGRESS_FILE"

# How often the echo below looks for new lines. The server has its own knob for
# the same thing (`RUN_EVENT_POLL_SECONDS`); this one only has to be fast
# enough that a developer reads it as live.
PROGRESS_POLL_SECONDS = 0.25


def append_progress(fraction, message, path=None):
    """One event, appended to the file named in the environment. Never raises.

    The same few lines every tool carries in its own `progress.py`, restated
    here rather than imported: this script runs in the TOOL's venv, and
    importing the tool's copy would mean reaching into a package before the
    tool has been resolved. It is short enough that the duplication costs less
    than the coupling.
    """
    path = path or os.environ.get(PROGRESS_VARIABLE)
    if not path:
        return False
    try:
        if fraction is not None:
            fraction = round(min(1.0, max(0.0, float(fraction))), 4)
        line = json.dumps(
            {"fraction": fraction, "message": str(message)[:200]}
        ).encode("utf-8") + b"\n"
        handle = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            os.write(handle, line)
        finally:
            os.close(handle)
        return True
    except Exception:  # noqa: BLE001 -- telemetry must never fail a run
        return False


def fold_fraction(fraction, span):
    """A callee's own 0..1 placed inside the share of its caller's bar it was given.

    No span, no fraction: the callee's 0..1 says nothing about where the CALLER
    is, and passing it on unchanged is what made a chain's bar run backwards
    (0.2 for "predicting landmarks", then 0.0 for ALI's first scan). The message
    still goes through.
    """
    if fraction is None or not span:
        return None
    start, end = float(span[0]), float(span[1])
    return start + (end - start) * min(1.0, max(0.0, float(fraction)))


def relay_progress(source, target, span):
    """Copy the events a callee appends to `source` onto `target`, folded into `span`.

    Returns a callable that relays what is left and stops. Each level of a
    chain relays its own callees, so a fraction is folded once per level and
    lands on the top bar where it belongs.
    """
    stopping = threading.Event()

    def relay():
        seen = 0
        while True:
            done = stopping.is_set()
            try:
                with open(str(source), "r", encoding="utf-8") as handle:
                    text = handle.read()
            except OSError:
                text = ""
            lines = text.split("\n")[:-1]
            for line in lines[seen:]:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                append_progress(
                    fold_fraction(event.get("fraction"), span),
                    event.get("message", ""),
                    path=str(target),
                )
            seen = len(lines)
            if done:
                return
            time.sleep(PROGRESS_POLL_SECONDS)

    thread = threading.Thread(target=relay, daemon=True)
    thread.start()

    def stop():
        stopping.set()
        thread.join(timeout=5)

    return stop


def format_progress(event):
    """One event as one stderr line, in the shape `LocalSupervisor.log` uses."""
    fraction = event.get("fraction")
    share = " --%" if fraction is None else "{:>4.0%}".format(fraction)
    return "[progress]{} {}".format(share, event.get("message", ""))


def watch_progress():
    """Set `SADT_PROGRESS_FILE` and echo what a tool appends to it.

    Returns a callable that drains and stops. It is a no-op when the variable
    is ALREADY set -- a nested `sup.run()` re-enters this script with the
    parent's file in its environment, which is exactly the inheritance the
    server relies on for chain progress, and a second echo would print every
    event once per level of the chain.
    """
    if os.environ.get(PROGRESS_VARIABLE):
        return lambda: None

    directory = tempfile.mkdtemp(prefix="sadt_progress_")
    path = os.path.join(directory, "events.jsonl")
    open(path, "a", encoding="utf-8").close()
    os.environ[PROGRESS_VARIABLE] = path

    stopping = threading.Event()

    def echo():
        seen = 0
        while True:
            done = stopping.is_set()
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    text = handle.read()
            except OSError:
                text = ""
            # COMPLETE lines only: whatever follows the last newline is a write
            # still in flight, and consuming it would lose the event rather
            # than print it late. Each event is one write, so this is at most
            # one poll behind.
            lines = text.split("\n")[:-1]
            for line in lines[seen:]:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue  # not ours, and not worth failing a run over
                sys.stderr.write(format_progress(event) + "\n")
                sys.stderr.flush()
            seen = len(lines)
            if done:
                return
            time.sleep(PROGRESS_POLL_SECONDS)

    # A daemon: an echo of telemetry must never be the reason the process
    # refuses to exit after the tool has returned.
    thread = threading.Thread(target=echo, daemon=True)
    thread.start()

    def stop():
        stopping.set()
        thread.join(timeout=5)
        os.environ.pop(PROGRESS_VARIABLE, None)
        shutil.rmtree(directory, ignore_errors=True)

    return stop


# ---------------------------------------------------------------------------
# The supervisor
# ---------------------------------------------------------------------------

class LocalSupervisor:
    """What a tool receives as `sup`, backed by sibling venvs on this machine.

    Five members and nothing more, matching what the server injects and what the
    tools' tests fake. A tool cannot tell the three apart, which is the point:
    nothing here is imported by a tool, and no shared package exists.

    `run()` re-enters this script rather than importing anything, so the callee
    gets its own interpreter and its own dependency set -- and its own
    supervisor, which is how nesting works without a special case.
    """

    def __init__(self, out, tmp, depth=0, chain=()):
        # Resolved, because `run()` below starts the callee from a NEUTRAL
        # working directory -- which is what makes "writes only under
        # output_dir" testable -- and a relative path handed across that
        # boundary resolves against the wrong place.
        self.out = Path(out).resolve()
        self.tmp = Path(tmp).resolve()
        self._depth = depth
        # The tools running above the callee, this one last.
        self._chain = list(chain)

    def run(self, tool, **params):
        # The caller's span of its own bar (`_progress=(start, end)`). The
        # callee declares no such argument, so it is taken out here exactly as
        # the server's supervisor takes it out, and used to fold the callee's
        # progress into the caller's bar (see `relay_progress`).
        span = params.pop("_progress", None)
        if tool in self._chain:
            raise RecursionError(
                "{} is already running above this call: {}.".format(
                    tool, " -> ".join(self._chain + [tool])
                )
            )
        if len(self._chain) >= MAX_SUPERVISOR_DEPTH:
            raise RecursionError(
                "Supervised calls are nested deeper than {}: {}.".format(
                    MAX_SUPERVISOR_DEPTH, " -> ".join(self._chain + [tool])
                )
            )
        self._say("running {}".format(tool))
        scratch = Path(tempfile.mkdtemp(prefix="sup_", dir=str(self.tmp)))
        # A callee declares `output_dir` as required, and a caller does not name one: where a supervised
        # tool writes is the supervisor's, as on the server. A caller that does name one is obeyed.
        params.setdefault("output_dir", scratch / "out")
        params_file, result_file = scratch / "params.json", scratch / "result.json"
        params_file.write_text(json.dumps(_jsonable(params)), encoding="utf-8")

        environment = dict(os.environ, **{SUPERVISOR_CHAIN_VARIABLE: ",".join(self._chain)})
        stop_relay = None
        if os.environ.get(PROGRESS_VARIABLE):
            # A file of the callee's own, relayed onto ours folded into its span:
            # written straight to ours, its 0..1 would move OUR bar backwards.
            own = scratch / "progress.jsonl"
            own.write_text("", encoding="utf-8")
            environment[PROGRESS_VARIABLE] = str(own)
            stop_relay = relay_progress(own, os.environ[PROGRESS_VARIABLE], span)
        try:
            completed = subprocess.run(
                [
                    sys.executable, str(SCRIPTS / "run_tool.py"), tool,
                    "--params-file", str(params_file),
                    "--result-file", str(result_file),
                    "--depth", str(self._depth + 1),
                ],
                # Not captured: a nested tool's progress is the only sign of life
                # during a run that can take an hour, and swallowing it is how a
                # chain looks hung. Its result never comes back on stdout anyway.
                cwd=str(scratch),
                env=environment,
                **_QUIET
            )
        finally:
            if stop_relay is not None:
                stop_relay()
        result = None
        if result_file.is_file():
            result = json.loads(result_file.read_text(encoding="utf-8"))
        if result is not None and result.get("kind") == "error":
            # The callee's own class name and message, so the level above (and
            # the clinician at the top) reads WHY, not that something below failed.
            error = failure_class(result.get("type", "ToolFailed"))(result.get("message", ""))
            error.tool = tool
            raise error
        if completed.returncode != 0 or result is None:
            raise ToolFailed(
                "{} failed (exit {}). Its error is above.".format(tool, completed.returncode)
            )
        if result["kind"] == "path":
            return Path(result["value"])
        return {key: Path(value) for key, value in result["value"].items()}

    def progress(self, fraction, message):
        # To the file first, because that is where the SERVER's supervisor
        # puts it: the waypoints tools already call this with have to land in
        # the same place as their own `progress.py` events, or a chain would
        # report half of itself on one channel and half on the other. The echo
        # thread prints it, so writing to stderr here as well would double
        # every line.
        if not append_progress(fraction, message):
            self._say("{:.0%} {}".format(fraction, message))

    def log(self, message, level="info", user=False):
        # The server's supervisor writes this to the run's events, for the
        # operator or for the clinician. Here there is only the developer at
        # the terminal, so it is the same line with its level and audience on.
        self._say("{}{}: {}".format("(user) " if user else "", str(level).upper(), message))

    def _say(self, message):
        # stderr, never stdout: stdout carries a result when this script is
        # driven by another copy of itself.
        sys.stderr.write("{}[sup] {}\n".format("  " * self._depth, message))
        sys.stderr.flush()


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def call(name, params, depth):
    """Import the tool, inject a supervisor if it wants one, call `run()`."""
    src = tool_root(name) / "src"
    run = describe.load_run(src, describe.find_package(src))
    hints = typing.get_type_hints(run)
    arguments = {
        key: coerce(value, hints[key]) for key, value in params.items() if key in hints
    }

    if takes_data_root(run) and os.environ.get(DATA_DIR_VARIABLE):
        arguments[describe.DATA_ROOT] = Path(os.environ[DATA_DIR_VARIABLE]).resolve()

    if not takes_supervisor(run):
        return run(**arguments)

    output_dir = Path(arguments.get("output_dir", Path.cwd()))
    output_dir.mkdir(parents=True, exist_ok=True)
    # OUTSIDE output_dir, and removed afterwards. The supervisor's scratch is
    # not the tool's output, and leaving it there would put files in the one
    # directory the caller keeps -- the tool is held to writing only inside it,
    # so whatever drives the tool should not undo that from the other side.
    scratch = Path(tempfile.mkdtemp(prefix="sadt_run_tool_"))
    try:
        above = [item for item in os.environ.get(SUPERVISOR_CHAIN_VARIABLE, "").split(",") if item]
        arguments["sup"] = LocalSupervisor(
            out=output_dir, tmp=scratch, depth=depth, chain=above + [name]
        )
        return run(**arguments)
    finally:
        shutil.rmtree(str(scratch), ignore_errors=True)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0].startswith("-"):
        sys.stderr.write(
            "usage: run_tool.py <TOOL> [--help]\n\ntools: {}\n".format(
                ", ".join(sorted(
                    path.name for path in TOOLS_DIR.iterdir()
                    if (path / "pyproject.toml").is_file()
                ))
            )
        )
        return 2

    name, rest = argv[0], argv[1:]

    # Read in BOTH layers: a failure in either has to reach the result file a
    # caller is waiting on, or a GUI can only say "it failed, see the log".
    machine = argparse.ArgumentParser(add_help=False)
    machine.add_argument("--params-file")
    machine.add_argument("--result-file")
    machine.add_argument("--depth", type=int, default=0)
    known, remaining = machine.parse_known_args(rest)

    def fail(error, code):
        sys.stderr.write("{}: {}\n".format(name, error))
        if known.result_file:
            write_result(known.result_file, error_payload(error))
        return code

    try:
        interpreter = tool_python(name)
    except ToolNotBuilt as error:
        return fail(error, 2)

    # Layer 1 -> layer 2. Compared by VENV, not by interpreter: uv's venvs all
    # symlink `bin/python` to the same underlying CPython, so resolving the
    # binary makes every tool look like every other one -- and the callee would
    # then be imported into its CALLER's environment, which is precisely the
    # dependency mixing this repository exists to prevent. `sys.prefix` is the
    # venv root inside a venv, so it is the thing that actually differs.
    if Path(sys.prefix).resolve() != (tool_root(name) / ".venv").resolve():
        return subprocess.run(
            [str(interpreter), str(Path(__file__).resolve()), name] + rest, **_QUIET
        ).returncode

    # -- layer 2, inside the tool's venv ------------------------------------
    src = tool_root(name) / "src"
    try:
        run = describe.load_run(src, describe.find_package(src))
    except describe.SchemaError as error:
        return fail(error, 2)

    if known.params_file:
        # Driven by a supervisor: arguments as JSON, result to a file. Never
        # parsed off stdout -- these tools print progress bars and nnUNet
        # banners, and anything scraped from stdout breaks the first time a
        # dependency prints something new.
        params = json.loads(Path(known.params_file).read_text(encoding="utf-8"))
    else:
        try:
            params = vars(build_parser(run, name).parse_args(remaining))
        except describe.SchemaError as error:
            return fail(error, 2)

    stop_watching = watch_progress()
    try:
        result = call(name, params, known.depth)
    except Exception as error:  # noqa: BLE001 -- reported, never swallowed
        traceback.print_exc()
        return fail(error, 1)
    finally:
        stop_watching()

    if known.result_file:
        write_result(known.result_file, result_payload(result))
    else:
        print(result if isinstance(result, (str, os.PathLike))
              else json.dumps(_jsonable(result), indent=2))
    return 0


def result_payload(result):
    """What `run()` returned, as the result file carries it.

    A path may come back as a `Path` or as a plain `str` (AREG_CBCT returns its
    output folder as text, and the server accepts both): either is a path. A
    dict maps names to paths. Anything else is refused by name rather than
    crashing after an hour of work with a run that did finish.
    """
    if isinstance(result, (str, os.PathLike)):
        return {"kind": "path", "value": os.fspath(result)}
    if isinstance(result, dict):
        return {"kind": "dict", "value": {str(k): os.fspath(v) if isinstance(v, (str, os.PathLike))
                                          else str(v) for k, v in result.items()}}
    raise TypeError(
        "run() returned {}; a tool returns a path or a dict of paths.".format(type(result).__name__))


# ---------------------------------------------------------------------------
# Driving a run from another program (a GUI)
# ---------------------------------------------------------------------------

# Variables that point an interpreter at ANOTHER Python's library. A GUI that
# embeds Python (3D Slicer does) sets them for its own interpreter; inherited by
# a tool's venv they would load the GUI's standard library into it.
_FOREIGN_PYTHON_VARIABLES = ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONEXECUTABLE")

# How long a cancelled run is given to stop on its own before it is killed.
CANCEL_GRACE_SECONDS = 5.0


class Job:
    """One run started by `start()`: poll it, read its progress and result, cancel it.

    Everything goes through files in `job_dir`, never through pipes, so the
    caller can poll from a GUI timer with no thread of its own and nothing
    blocks when a tool prints a lot:

        params.json     what the tool was called with
        progress.jsonl  one JSON event per line, `{"fraction", "message"}`
        result.json     `{"kind": "path"|"dict"|"error", ...}`, written atomically
        run.log         everything the chain printed, stdout and stderr
    """

    def __init__(self, tool, job_dir, process, log_handle):
        self.tool = tool
        self.job_dir = Path(job_dir)
        self.process = process
        self._log_handle = log_handle
        self._read_events = 0

    @property
    def params_file(self):
        return self.job_dir / "params.json"

    @property
    def result_file(self):
        return self.job_dir / "result.json"

    @property
    def progress_file(self):
        return self.job_dir / "progress.jsonl"

    @property
    def log_file(self):
        return self.job_dir / "run.log"

    def poll(self):
        """The exit code once the whole run has ended, None while it runs."""
        code = self.process.poll()
        if code is not None:
            self._close_log()
        return code

    def wait(self, timeout=None):
        code = self.process.wait(timeout=timeout)
        self._close_log()
        return code

    def new_progress(self):
        """The progress events appended since the last call, oldest first."""
        try:
            text = self.progress_file.read_text(encoding="utf-8")
        except OSError:
            return []
        # Complete lines only: a line still being written is read next time.
        lines = text.split("\n")[:-1]
        events = []
        for line in lines[self._read_events:]:
            try:
                events.append(json.loads(line))
            except ValueError:
                continue
        self._read_events = len(lines)
        return events

    def result(self):
        """What `run()` returned -- a `Path` or a `dict` of them -- or raise its failure.

        Call it once `poll()` is not None. A failure is raised under the tool's
        own exception class name (see `ToolFailed`), with the tool's message.
        """
        if not self.result_file.is_file():
            raise ToolFailed(
                "{} ended (exit {}) without a result. See {}.".format(
                    self.tool, self.process.poll(), self.log_file
                )
            )
        payload = json.loads(self.result_file.read_text(encoding="utf-8"))
        if payload.get("kind") == "error":
            error = failure_class(payload.get("type", "ToolFailed"))(payload.get("message", ""))
            error.tool = self.tool
            raise error
        if payload["kind"] == "path":
            return Path(payload["value"])
        return {key: Path(value) for key, value in payload["value"].items()}

    def cancel(self, grace=CANCEL_GRACE_SECONDS):
        """Stop the run and EVERY process below it: nested tools included.

        The run was started as the leader of its own process group (POSIX) or
        process group / tree (Windows), and every nested `sup.run()` is a
        descendant of it, so one signal reaches the whole chain. It is asked
        to stop first, then killed after `grace` seconds.
        """
        if self.process.poll() is None:
            if os.name == "nt":
                _stop_tree_windows(self.process, grace)
            else:
                _stop_group_posix(self.process, grace)
        self._close_log()

    def _close_log(self):
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None


def _stop_group_posix(process, grace):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except (ProcessLookupError, PermissionError):
            return
        try:
            process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            continue
        # The leader is gone; a descendant that ignored SIGTERM may not be.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        return


def _stop_tree_windows(process, grace):
    # CTRL_BREAK reaches the console process group the run leads; taskkill /T
    # then walks the tree by parent id, which is what reaches nested tools.
    try:
        process.send_signal(signal.CTRL_BREAK_EVENT)
        process.wait(timeout=grace)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass
    subprocess.run(
        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **_QUIET
    )
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass


_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
# For every process this script starts on Windows: no console window of its own.
_QUIET = {"creationflags": _CREATE_NO_WINDOW} if os.name == "nt" else {}


def start(tool, params, job_dir, env=None, data_dir=None):
    """Start `tool` with `params` in the background and return its `Job`.

    The run goes through this same script, in the tool's own venv, so it is
    exactly the path `run_tool.py <TOOL>` takes -- supervisor, chaining and
    progress included. `job_dir` is created and holds every file of the run.

    `env` is the environment to start from (default: this process's). A GUI
    that embeds Python should pass the environment it was launched with, from
    before it set its own library paths; the variables that would load another
    Python's standard library are removed either way. `data_dir` is the DATA
    root (see `SADT_DATA_DIR`), for tools that find their own models.
    """
    job_dir = Path(job_dir).resolve()
    job_dir.mkdir(parents=True, exist_ok=True)
    job = Job(tool, job_dir, None, None)

    job.params_file.write_text(json.dumps(_jsonable(dict(params))), encoding="utf-8")
    job.progress_file.write_text("", encoding="utf-8")
    # A folder used before must not answer for this run.
    if job.result_file.exists():
        job.result_file.unlink()

    environment = dict(os.environ if env is None else env)
    for variable in _FOREIGN_PYTHON_VARIABLES:
        environment.pop(variable, None)
    environment[PROGRESS_VARIABLE] = str(job.progress_file)
    environment[TOOLS_DIR_VARIABLE] = str(TOOLS_DIR)
    if data_dir is not None:
        environment[DATA_DIR_VARIABLE] = str(Path(data_dir).resolve())
    environment.pop(SUPERVISOR_CHAIN_VARIABLE, None)

    # A tool's output goes to a file, and on Windows a file is written in the
    # locale's code page unless told otherwise: one print() of a path in
    # another alphabet would then end the run with a UnicodeEncodeError.
    environment.setdefault("PYTHONUTF8", "1")

    if os.name == "nt":
        # No console window either: a GUI driver (Slicer) has none to share, so
        # each run would open one, and closing it would kill the run.
        group = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | _CREATE_NO_WINDOW}
    else:
        group = {"start_new_session": True}

    log_handle = open(str(job.log_file), "wb")
    try:
        job.process = subprocess.Popen(
            [
                str(tool_python(tool)), str(Path(__file__).resolve()), tool,
                "--params-file", str(job.params_file),
                "--result-file", str(job.result_file),
            ],
            cwd=str(job_dir),
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            **group
        )
    except BaseException:
        log_handle.close()
        raise
    job._log_handle = log_handle
    return job


if __name__ == "__main__":
    sys.exit(main())
