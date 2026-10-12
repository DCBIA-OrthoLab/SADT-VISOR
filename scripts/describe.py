#!/usr/bin/env python3
"""Emit the JSON schema the server publishes for a tool, read from its run().

Run it with the tool's OWN interpreter, because importing the tool needs the
tool's dependencies and nothing in this repository shares an environment:

    tools/AMASSS/.venv/bin/python scripts/describe.py tools/AMASSS

The signature is the single source of truth. There is no second declaration to
keep in step, which is the whole reason the old `ArgSpec` tables are gone: they
drifted from `run()` and the client rendered widgets the tool did not have.

This script deliberately FAILS (exit 2) on anything it cannot represent rather
than emitting a plausible-looking schema. A wrong schema is worse than no
schema: the client builds its form from it, and a silently dropped argument
becomes a run that succeeds with the wrong parameters.

It stays compatible with Python 3.9 -- a tool may pin an old interpreter, and
this must run inside it -- so no tomllib and no match statement here.
"""

import argparse
import ast
import hashlib
import importlib
import inspect
import json
import re
import sys
import typing
from pathlib import Path

# The whole annotation vocabulary. `Path` means "a file or a directory"; the
# rest are scalars. `Literal[...]` narrows a str or int to a fixed set of
# options, which is published as `choices` so the client can render a picker.
# Nothing else is allowed -- no Optional, no unions, no custom marker types,
# nothing imported from the server.
SCALARS = {
    Path: "path",
    str: "str",
    int: "int",
    float: "float",
    bool: "bool",
}

# What a default value may be for each type. `float` accepts an int literal
# (`threshold: float = 0`) and is normalised below so the client gets a float.
ACCEPTED_DEFAULTS = {
    "path": (str, Path),
    "str": (str,),
    "int": (int,),
    "float": (int, float),
    "bool": (bool,),
}


# Literal values may only be these. PEP 586 also allows bytes, None and Enum
# members; none of them survives a JSON round trip as an option a client can
# render, so they are refused rather than silently stringified.
LITERAL_TYPES = (str, int)

# The supervisor: how a tool calls ANOTHER tool. It is keyword-only,
# unannotated, and excluded from the schema -- a client never sends it, because
# it is not data. The runner injects it.
#
# Unannotated is the marker rather than an accident: every other parameter must
# be annotated, so there is nothing else this shape could be, and a tool cannot
# grow a supervisor by forgetting a type. It is duck-typed on purpose -- a tool
# importing a supervisor class would need a package shared with the server,
# which is the coupling this repository exists to remove.
SUPERVISOR = "sup"

# The read-only DATA root, declared and excluded exactly as the supervisor is,
# and for the same reason: it is not data a client sends. A tool takes it to
# find ITS OWN model bundle -- `data_root / "ALI" / "models"` -- which is what
# keeps a caller from having to name its neighbour's weights. The server
# resolves a hosted-model argument on the way in, but a supervised call never
# passes that way, so without this the CALLER was composing the path.
DATA_ROOT = "data_root"

INJECTED = (SUPERVISOR, DATA_ROOT)

# Arguments a tool never declares and the SERVER adds for it. `keep_intermediate`
# appears on any tool that calls another (see `supervised_calls`): collecting what
# a chain produced is the same operation every time, so it is done once in the
# runner rather than written into each orchestrating tool. `stop_after` is the
# same shape -- the server builds it from the checkpoints a chain offers, and
# reads a layout for it out of `injected_layout` exactly as it does for the
# other, so a tool that has no business offering the section can say so.
#
# A layout may still name one. That is the whole point of publishing it here: the
# argument is generic, but "Keep the predicted landmarks" and "Keep the labelled
# meshes" are not, and only the tool knows which it produces.
INJECTED_ARGUMENTS = ("keep_intermediate", "stop_after")

# The docstring section that explains the arguments, in the Google style the
# whole repository already writes. It is the ONLY place that text lives: the
# client shows it under the field, and a panel without it is a column of
# unexplained inputs -- "Register on" tells a clinician nothing, "pick what has
# NOT changed between the two timepoints" tells them everything.
#
# Parsed rather than restated because the alternative was tried: the old
# `ArgSpec` tables carried their own descriptions, drifted from the docstrings
# beside them, and the two disagreed about which arguments were CBCT-only.
DOC_ARGUMENTS = "Args:"

# Presentation, and ONLY presentation. A tool may ship a `layout` module beside
# its package declaring how a client should lay its panel out; these are the
# keys it may set, and nothing here can change what `run()` accepts.
#
# It exists because a schema that says only "119 strings, pick some" makes a
# panel nobody can use, and the client already knows how to render tabs,
# sections and conditional fields -- it just stopped being told. The old
# `ArgSpec` tables carried this and DRIFTED from the code, so the rule is that
# a layout module may only ever DERIVE from the tool's own catalogs, never
# restate them; `layout_for` below refuses anything that names an argument or
# an option the signature does not publish, which is what makes that hold.
LAYOUT_KEYS = ("section", "ui", "groups", "visible_when", "options_when", "label",
               "hidden",
               # A vec2's two axes, low end first (or high end first to mirror
               # the axis). Declared here with the other layout keys because a
               # tool writes them in the same place, but they are NOT
               # presentation: the server refuses a value outside them.
               "x_range", "y_range",
               # Names for the two ends of each axis. Presentation, unlike the
               # ranges: "0.8" says nothing about where that is in a mouth, and
               # "mid"/"out" does.
               "x_labels", "y_labels",
               # MULTICHOICE only: a "select all" / "select none" pair above
               # the options, for a catalogue nobody would tick one box at a
               # time. The client hides the pair when there is only one option,
               # where it would be two buttons for one check box.
               "select_all",
               # How many columns this argument's SECTION is laid out in.
               # Declared per argument because that is where a layout hangs its
               # hints; the client reads it back per section.
               "section_columns",
               # Arguments naming one cell are drawn together in it.
               "cell",
               # What each of a vec2's two numbers is, written beside its box.
               "x_label", "y_label",
               # {option: one line saying what that option is}. For a catalogue
               # whose options are CODES -- ALI publishes `UR1MB` and `Ba` -- the
               # label alone tells a clinician nothing, and the argument's own
               # description covers all of them at once.
               "option_help",
               # How few options a multichoice may be left with; absent means
               # none is a meaningful answer.
               "min_selected")

LAYOUT_MODULE = "layout"


class SchemaError(Exception):
    """The tool cannot be described. Always actionable, always fatal."""


def literal_choices(annotation, where):
    """The options of a `Literal[...]`, with the scalar type they narrow.

    Returns `(type name, choices)`, or `(None, None)` when this is not a
    Literal. A fixed set of options is a property of the argument, so it is
    said once in the annotation rather than declared a second time in a table
    that can drift from it -- which is exactly what the old `ArgSpec.choices`
    did.
    """
    if typing.get_origin(annotation) is not typing.Literal:
        return None, None

    values = list(typing.get_args(annotation))
    if not values:
        raise SchemaError("{}: Literal[] has no options.".format(where))

    kinds = set(type(value) for value in values)
    # bool is a subclass of int, so Literal[True] would otherwise be published
    # as an int option and rendered as a number.
    if any(isinstance(value, bool) for value in values) or len(kinds) != 1:
        raise SchemaError(
            "{}: Literal options must all be str, or all be int. Got "
            "{}.".format(where, ", ".join(sorted(k.__name__ for k in kinds)))
        )
    kind = kinds.pop()
    if kind not in LITERAL_TYPES:
        raise SchemaError(
            "{}: Literal[{}] is not supported. Options must be str or "
            "int.".format(where, kind.__name__)
        )
    # No duplicate check: typing.Literal collapses repeated options itself, so
    # Literal["a", "a"] never reaches here as two.
    return SCALARS[kind], values


def type_name(annotation, where):
    """Map an annotation onto its schema name, or refuse it.

    Returns `(name, choices)`; `choices` is None unless a Literal narrowed it.
    """
    try:
        if annotation in SCALARS:
            return SCALARS[annotation], None
    except TypeError:
        pass  # an unhashable annotation is unsupported by definition

    name, choices = literal_choices(annotation, where)
    if name is not None:
        return name, choices

    # Bare `list` is not a generic alias, so it never reaches the branch below.
    # It is a common enough slip to deserve its own message: without an element
    # type the client has nothing to build a widget from.
    if annotation is list:
        raise SchemaError(
            "{}: bare list is not supported. Say what it holds, e.g. "
            "list[str].".format(where)
        )

    # `tuple[float, float]` is a "vec2": two numbers set together because they
    # are one position rather than two settings. The client renders a 2D pad for
    # it when the layout also says `ui = "joystick"`, and the server validates
    # both numbers against the `x_range`/`y_range` the layout declares -- the one
    # layout key that is not presentation only.
    #
    # Only the two-float form. `tuple[float, float, float]` is a point and would
    # want a different widget; refusing it now is cheaper than publishing a type
    # nothing renders.
    if typing.get_origin(annotation) is tuple:
        args = typing.get_args(annotation)
        if args == (float, float):
            return "vec2", None
        raise SchemaError(
            "{}: tuple[{}] is not supported. Only tuple[float, float], which is "
            "the two-axis 'vec2'.".format(
                where, ", ".join(getattr(a, "__name__", str(a)) for a in args) or "..."
            )
        )

    if typing.get_origin(annotation) is list:
        args = typing.get_args(annotation)
        if len(args) == 1:
            # `list[Literal[...]]` is the multi-select: several of a fixed set.
            # A bare `Literal[...]` is the single-select. The two old schema
            # types, "multichoice" and "choice", fall straight out of that.
            element, choices = literal_choices(args[0], where)
            if element is None and SCALARS.get(args[0]) is not None:
                element = SCALARS[args[0]]
            if element is not None:
                return "list[{}]".format(element), choices
        raise SchemaError(
            "{}: list[{}] is not supported. Use list[str], list[int], "
            "list[float], list[bool], list[Path] or list[Literal[...]].".format(
                where, ", ".join(getattr(a, "__name__", str(a)) for a in args) or "..."
            )
        )

    raise SchemaError(
        "{}: unsupported annotation {!r}. Allowed: Path, str, int, float, bool, "
        "Literal[...] of str or int, tuple[float, float], and list[...] of those. "
        "Optional/Union are "
        "not supported -- an argument is optional because it has a default, not "
        "because it is typed that way.".format(where, annotation)
    )


def return_name(annotation):
    """`Path` for one output, `dict[str, Path]` for several named ones."""
    if annotation is inspect.Signature.empty:
        raise SchemaError("run() has no return annotation. Annotate it `-> Path`.")

    if typing.get_origin(annotation) is dict:
        args = typing.get_args(annotation)
        if len(args) == 2 and args[0] is str and args[1] is Path:
            return "dict[str, path]"
        raise SchemaError(
            "return annotation: only dict[str, Path] is supported for multiple "
            "outputs, got {!r}.".format(annotation)
        )

    name, _choices = type_name(annotation, "return annotation")
    if name != "path":
        raise SchemaError(
            "run() must return a Path (or dict[str, Path]), not {}. Everything a "
            "tool produces is a file under its output directory.".format(name)
        )
    return name


def check_default(name, kind, default, choices=None):
    """Validate a default against its annotation and make it JSON-ready."""
    if default is None:
        raise SchemaError(
            "argument '{}' defaults to None. There is no nullable type here: drop "
            "the default to make the argument required, or give it a real one.".format(name)
        )

    if kind == "vec2":
        # Two numbers, and JSON has no tuple: a default written `(0.5, 0.0)` in
        # the signature has to arrive as a list, the same shape the wire uses.
        if not isinstance(default, (list, tuple)) or len(default) != 2:
            raise SchemaError(
                "argument '{}' is vec2 but defaults to {!r}. A vec2 default is "
                "two numbers.".format(name, default)
            )
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               for value in default):
            raise SchemaError(
                "argument '{}' is vec2 but defaults to {!r}. Both must be "
                "numbers.".format(name, default)
            )
        return [float(value) for value in default]

    if kind.startswith("list["):
        element = kind[len("list[") : -1]
        if not isinstance(default, (list, tuple)):
            raise SchemaError(
                "argument '{}' is {} but defaults to {!r}.".format(name, kind, default)
            )
        return [check_default(name, element, item, choices) for item in default]

    accepted = ACCEPTED_DEFAULTS[kind]
    # bool is a subclass of int, so `count: int = True` passes isinstance and
    # would ship an int argument the client renders as a check box.
    if isinstance(default, bool) != (kind == "bool"):
        raise SchemaError(
            "argument '{}' is {} but defaults to {!r}.".format(name, kind, default)
        )
    if not isinstance(default, accepted):
        raise SchemaError(
            "argument '{}' is {} but defaults to {!r}.".format(name, kind, default)
        )

    # A default outside its own option list is the quiet one: the client shows
    # a picker that cannot produce the value the tool starts from.
    if choices is not None and default not in choices:
        raise SchemaError(
            "argument '{}' defaults to {!r}, which is not one of its options "
            "({}).".format(name, default, ", ".join(repr(c) for c in choices))
        )

    if kind == "path":
        return str(default)
    if kind == "float":
        return float(default)
    return default



def supervised_calls(src_dir):
    """Every tool name this tool asks the supervisor for, sorted.

    Found by READING the source, not by importing it: the call sites are inside
    branches that only a real run reaches, so there is nothing to introspect at
    import time.

    A tool cannot import another tool -- they are separate virtualenvs, which is
    the whole reason the split exists -- so a call name is necessarily a free
    string. That is a deliberate choice rather than an oversight, and it stays.
    What it lacks is verification, and publishing the names is what lets the
    SERVER supply it: a name matching no registered tool then fails at startup
    instead of an hour into a registration.

    Two spellings are resolved, because both are in real use:
        sup.run("ASO", ...)                 a literal
        sup.run(LANDMARK_TOOL, ...)         a module-level string constant

    Anything else is refused. A name this cannot see is a name the server cannot
    check, and a check with a hole in it is worse than no check -- it reads as
    coverage.
    """
    names = set()
    for path in sorted(Path(src_dir).rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError) as error:
            raise SchemaError("{}: {}".format(path, error))

        # Module-level `NAME = "value"`, which is how ASO spells it.
        constants = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) \
                    and isinstance(node.value.value, str):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        constants[target.id] = node.value.value

        # `for name in ("Crown_Seg", "ALI_IOS"): require(sup, name, ...)`, which
        # is how AREG_IOSCBCT states the three tools one mode needs. The names
        # are still literals, written once instead of three times; refusing the
        # form would push a tool towards the more repetitive spelling to satisfy
        # a reader that is only looking for strings.
        loop_names = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.For) or not isinstance(node.target, ast.Name):
                continue
            if not isinstance(node.iter, (ast.Tuple, ast.List)):
                continue
            values = [element.value for element in node.iter.elts
                      if isinstance(element, ast.Constant) and isinstance(element.value, str)]
            if len(values) == len(node.iter.elts):
                loop_names.setdefault(node.target.id, set()).update(values)

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func

            # `require(sup, "AMASSS", ...)` states a dependency without making
            # the call: it is how a tool refuses a mode early, before an hour of
            # work, and it names a tool exactly as `sup.run` does. It was not
            # collected here, so AREG_IOS went on asking for 'ALI' after the
            # split renamed it, and neither the generator nor the server's
            # startup check said anything -- the failure waited for a
            # mucogingival run. That is the hole this function's own docstring
            # warns about, so it is closed rather than documented.
            require_name = function.attr if isinstance(function, ast.Attribute) else (
                function.id if isinstance(function, ast.Name) else None)
            if require_name == "require" and len(node.args) >= 2:
                first, second = node.args[0], node.args[1]
                if isinstance(first, ast.Name) and first.id == SUPERVISOR:
                    if isinstance(second, ast.Constant) and isinstance(second.value, str):
                        names.add(second.value)
                        continue
                    if isinstance(second, ast.Name) and second.id in constants:
                        names.add(constants[second.id])
                        continue
                    if isinstance(second, ast.Name) and second.id in loop_names:
                        names.update(loop_names[second.id])
                        continue
                    raise SchemaError(
                        "{}:{}: require()'s tool name must be a literal or a "
                        "module-level string constant, so the server can check it "
                        "exists. Got {}.".format(
                            path.name, node.lineno, ast.dump(second)[:60])
                    )

            if not isinstance(function, ast.Attribute) or function.attr != "run":
                continue
            if not isinstance(function.value, ast.Name) or function.value.id != SUPERVISOR:
                continue
            if not node.args:
                raise SchemaError(
                    "{}:{}: {}.run() is called with no tool name.".format(
                        path.name, node.lineno, SUPERVISOR)
                )
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                names.add(first.value)
            elif isinstance(first, ast.Name) and first.id in constants:
                names.add(constants[first.id])
            elif isinstance(first, ast.Name) and first.id in loop_names:
                names.update(loop_names[first.id])
            else:
                raise SchemaError(
                    "{}:{}: {}.run()'s tool name must be a literal or a module-level "
                    "string constant, so the server can check it exists. Got {}.".format(
                        path.name, node.lineno, SUPERVISOR, ast.dump(first)[:60])
                )
    return sorted(names)

# What a tool calls to offer a reader a place to stop. Read out of the source
# the same way `sup.run` is, and for the same reason: the call sites are
# inside branches only a real run reaches.
QUALITY_CONTROL = "declareQualityControl"

# What a reader may DO at a checkpoint, and therefore whether it is somewhere
# to come BACK to. The three are the local module's, unchanged: it already
# drew this distinction across two pipelines and it is the right one.
#
#   view          look at the result, then carry on. Nothing to edit here.
#   landmarks     drag the points; they are saved back to their own file.
#   registration  drag the scan; the move is folded into its matrix.
#
# `view` is the default, and that is the conservative direction: a checkpoint
# nobody declared as editable never offers a reader a way back to it, so a
# tool that says nothing cannot promise something it does not support.
REVIEW_VIEW = "view"
REVIEW_KINDS = (REVIEW_VIEW, "landmarks", "registration")


# Which argument holds one entry per case, so a caller can hand this tool a
# SUBSET of them.
CASE_INPUT = "CASE_INPUT"


def case_input(src_dir, arguments):
    """The argument whose folder holds one entry per case, or "".

    **What it is for.** A reader who marked three patients of forty wants
    those three done again and the other thirty-seven left alone. Doing that
    means handing the tool fewer inputs -- and which argument to narrow is
    something no caller can work out: it is `input` here, `scans` there,
    `meshes` elsewhere, and a registration tool has TWO. A server guessing it
    would be guessing about every tool it serves.

    So the tool says, beside the signature it belongs to:

        CASE_INPUT = "scans"

    Absent is the ordinary case and means "cannot be narrowed": the replay is
    then the whole cohort, which is slower and never wrong. That is the
    direction to fail in -- narrowing the wrong argument would silently drop
    patients from a run.
    """
    for path in sorted(Path(src_dir).rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError) as error:
            raise SchemaError("{}: {}".format(path, error))
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if CASE_INPUT not in [t.id for t in node.targets
                                  if isinstance(t, ast.Name)]:
                continue
            if not isinstance(node.value, ast.Constant) or not isinstance(
                    node.value.value, str):
                raise SchemaError(
                    "{}:{}: {} must be a string literal.".format(
                        path.name, node.lineno, CASE_INPUT))
            named = node.value.value
            if named not in arguments:
                raise SchemaError(
                    "{}:{}: {} names {!r}, which run() does not take.".format(
                        path.name, node.lineno, CASE_INPUT, named))
            return named
    return ""


# What a reader may DO with what this tool produced, and therefore whether a
# stop after a call to it is somewhere to come BACK to.
REVIEW_KIND = "REVIEW_KIND"


def review_kind(src_dir):
    """What a reader may do with this tool's output, or "" if it says nothing.

    **Why a tool declares this about ITSELF.** A checkpoint declared mid-run
    carries its own kind -- `sup.declareQualityControl("lm", kind=...)` says
    it on the spot. But most stops are the boundary after a CALL, named after
    the callee, and nothing at that call site knows whether what came back can
    be edited. The caller did not write it.

    The tool that WROTE it did. ALI produces landmarks wherever it is called
    from; ASO produces an orientation. So the fact belongs beside the code
    that writes the files, exactly as `OUTPUT_SUFFIXES` does, and the server
    composes `ASO/ALI_CBCT`'s kind out of ALI_CBCT's own declaration.

    Declared as a module-level string literal anywhere under `src/`:

        REVIEW_KIND = "landmarks"

    Absent is the ordinary case and reads as `view`: a tool that says nothing
    never offers a reader a way back to it, which is the conservative
    direction.
    """
    for path in sorted(Path(src_dir).rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError) as error:
            raise SchemaError("{}: {}".format(path, error))
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if REVIEW_KIND not in [t.id for t in node.targets
                                   if isinstance(t, ast.Name)]:
                continue
            if (not isinstance(node.value, ast.Constant)
                    or node.value.value not in REVIEW_KINDS):
                raise SchemaError(
                    "{}:{}: {} must be one of {}. Got {}.".format(
                        path.name, node.lineno, REVIEW_KIND,
                        ", ".join(REVIEW_KINDS), ast.dump(node.value)[:60])
                )
            return node.value.value
    return ""


# What a tool appends to the name of what it was given. ASO hands back
# `P1_Or.nii.gz` for the `P1.nii.gz` it received; ALI hands back
# `P1_lm_Pred.mrk.json`.
OUTPUT_SUFFIXES = "OUTPUT_SUFFIXES"


def output_suffixes(src_dir):
    """The markers this tool adds to the name of the file it was handed.

    **Why this is published rather than known.** The server has to work out,
    from a folder of results, which of them are about the same patient -- to
    replay a chain for the three cases a clinician marked and leave the other
    thirty-seven alone. Deriving that means stripping the markers tools
    append, and a table of those markers inside the server would be the
    server knowing what ALI and ASO call their outputs. It knows no dental
    tool, and that is the property the whole architecture is arranged around.

    So the tool says. It is the one writing the name, it is the only place
    the fact is not a guess, and a tool that renames its outputs updates one
    tuple beside them instead of a table in another repository.

    Declared as a module-level tuple of string literals anywhere under
    `src/`:

        OUTPUT_SUFFIXES = ("_Or", "_Or_transform", "_lm_Or")

    Read by AST for the same reason `supervised_calls` is: importing a tool
    to ask it a question costs a CUDA stack, and CI asks this on every push.
    """
    found = []
    for path in sorted(Path(src_dir).rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError) as error:
            raise SchemaError("{}: {}".format(path, error))
        # Module-level sequences, so the declaration may POINT at the table a
        # tool already keeps rather than repeat it. ASO derives the patient a
        # file belongs to for its own pairing and has carried that tuple for
        # as long as it has paired anything; making it write the same strings
        # twice is how the two drift.
        sequences = {
            target.id: node.value
            for node in tree.body if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
            and isinstance(node.value, (ast.Tuple, ast.List))
        }
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if OUTPUT_SUFFIXES not in names:
                continue
            value = node.value
            if isinstance(value, ast.Name):
                value = sequences.get(value.id)
                if value is None:
                    raise SchemaError(
                        "{}:{}: {} names {!r}, which is not a tuple or list "
                        "declared at module level in this file.".format(
                            path.name, node.lineno, OUTPUT_SUFFIXES, node.value.id)
                    )
            if not isinstance(value, (ast.Tuple, ast.List)):
                raise SchemaError(
                    "{}:{}: {} must be a tuple or list of string literals.".format(
                        path.name, node.lineno, OUTPUT_SUFFIXES)
                )
            for element in value.elts:
                if not isinstance(element, ast.Constant) or not isinstance(
                        element.value, str):
                    raise SchemaError(
                        "{}:{}: every {} entry must be a string literal, so the "
                        "server can publish it. Got {}.".format(
                            path.name, node.lineno, OUTPUT_SUFFIXES,
                            ast.dump(element)[:60])
                    )
                if not element.value.startswith("_"):
                    # The marker is what SEPARATES it from the name, so it has
                    # to carry the separator. Without it `Or` would cut inside
                    # any patient whose name contains those two letters.
                    raise SchemaError(
                        "{}:{}: {} entry {!r} must begin with '_'.".format(
                            path.name, node.lineno, OUTPUT_SUFFIXES, element.value)
                    )
                if element.value not in found:
                    found.append(element.value)
    # Longest first, so a caller stripping them cannot cut inside a compound:
    # `_lm_Pred` has to be tried before `_lm`.
    return sorted(found, key=len, reverse=True)


def quality_controls(src_dir):
    """Every checkpoint this tool offers a reader, in the order it declares them.

    Each entry is `{"name": ..., "kind": ...}`. The kind says what a reader
    may DO there, and therefore whether it is somewhere to come back to from
    a later stop -- see REVIEW_KINDS.

    `sup.declareQualityControl("landmarks")` says: here is a point where what
    has been produced so far is worth looking at before the rest is built on
    it. A chain already offers one per `sup.run()`; this is for the tool that
    wants one in the MIDDLE of its own work, where no call boundary exists.

    Sorted by nothing -- declaration order is the order they happen in, and a
    reader picking where to stop is reading a sequence, not an index.

    The name must be a literal. The same reasoning as `supervised_calls`: a
    name this cannot see is a name the server cannot publish, and a checkbox
    list with a hole in it reads as coverage.
    """
    found = []
    for path in sorted(Path(src_dir).rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError) as error:
            raise SchemaError("{}: {}".format(path, error))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            if not isinstance(function, ast.Attribute) or function.attr != QUALITY_CONTROL:
                continue
            if not isinstance(function.value, ast.Name) or function.value.id != SUPERVISOR:
                continue
            if not node.args:
                raise SchemaError(
                    "{}:{}: {}.{}() is called with no name.".format(
                        path.name, node.lineno, SUPERVISOR, QUALITY_CONTROL)
                )
            first = node.args[0]
            if not isinstance(first, ast.Constant) or not isinstance(first.value, str):
                raise SchemaError(
                    "{}:{}: {}.{}()'s name must be a literal, so the server can "
                    "publish it. Got {}.".format(
                        path.name, node.lineno, SUPERVISOR, QUALITY_CONTROL,
                        ast.dump(first)[:60])
                )
            kind = REVIEW_VIEW
            for keyword in node.keywords:
                if keyword.arg != "kind":
                    continue
                if (not isinstance(keyword.value, ast.Constant)
                        or keyword.value.value not in REVIEW_KINDS):
                    raise SchemaError(
                        "{}:{}: {}.{}()'s kind must be one of {}. Got {}.".format(
                            path.name, node.lineno, SUPERVISOR, QUALITY_CONTROL,
                            ", ".join(REVIEW_KINDS), ast.dump(keyword.value)[:60])
                    )
                kind = keyword.value.value
            if first.value not in [entry["name"] for entry in found]:
                found.append({"name": first.value, "kind": kind})
    return found


def is_supervisor(name, parameter, hints):
    """Whether this parameter is the supervisor, refusing near-misses.

    A near-miss is worth a hard error rather than a schema entry: `sup` typed
    as a `Path` would be published as a file the client is asked to upload, and
    a positional `sup` would be filled by the first argument the runner passes.
    Both fail far from here.
    """
    if name not in INJECTED:
        return False
    if parameter.kind is not parameter.KEYWORD_ONLY:
        raise SchemaError(
            "'{0}' must be keyword-only: write `*, {0}=None`. Anything else can be "
            "filled positionally by a caller that meant it as data.".format(name)
        )
    if name in hints:
        raise SchemaError(
            "'{0}' must not be annotated. Being unannotated is what marks it as "
            "injected by the runner rather than an argument, and it is duck-typed "
            "so a tool never imports the server's type.".format(name)
        )
    return True


def doc_block(doc):
    """The lines under `Args:`, or None if run() has no such section.

    `inspect.getdoc` has already stripped the common indentation, so the
    section headings sit at column 0 and everything belonging to one is
    indented. The block ends at the next heading -- Returns, Raises, or plain
    prose -- which is the first non-blank line back at that margin.
    """
    lines = doc.splitlines()
    for start, line in enumerate(lines):
        if line.strip() == DOC_ARGUMENTS and not line[:1].isspace():
            break
    else:
        return None

    block = []
    for line in lines[start + 1:]:
        if line.strip() and not line[:1].isspace():
            break
        block.append(line)
    return block


def doc_entries(block):
    """`{argument: text}` from an Args block, joining wrapped lines.

    An entry starts at the block's own margin and looks like `name:`; anything
    indented past it continues the previous entry. A description is one
    paragraph by the time it reaches a client, so the wrapping the source needs
    is undone here rather than shipped as newlines a panel renders literally.
    """
    margins = [len(line) - len(line.lstrip()) for line in block if line.strip()]
    margin = min(margins) if margins else 0

    entries = {}
    name = None
    for line in block:
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        # `name (type): text` is accepted and the type dropped: the annotation
        # is the truth about the type, and a docstring repeating it is the
        # second declaration this script exists to avoid.
        head = re.match(r"(\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)$", line.strip())
        if indent == margin and head:
            name = head.group(1)
            entries[name] = head.group(2).strip()
        elif name is not None:
            entries[name] = (entries[name] + " " + line.strip()).strip()
    return entries


def argument_docs(doc, arguments):
    """The per-argument help out of run()'s docstring, checked against run().

    Every argument must have a line and no line may name something run() does
    not take, both for the same reason the layout is checked: a description
    that quietly goes nowhere is invisible from every end. The old tables were
    dropped precisely because nothing noticed when they stopped matching.
    """
    if not arguments:
        return {}

    block = doc_block(doc)
    if block is None:
        raise SchemaError(
            "run()'s docstring has no '{}' section. Every argument's description "
            "is read from it -- without one the client renders a form of "
            "unexplained fields.".format(DOC_ARGUMENTS)
        )

    entries = doc_entries(block)
    unknown = sorted(set(entries) - set(arguments))
    if unknown:
        raise SchemaError(
            "the '{}' section documents {}, which run() does not take. A "
            "description nothing renders is worse than none: it reads as "
            "current.".format(DOC_ARGUMENTS, ", ".join(unknown))
        )
    undocumented = [name for name in arguments if name not in entries]
    if undocumented:
        raise SchemaError(
            "no description for {}. Add a line under '{}' -- it is what a "
            "clinician reads next to the control, and a control they have to "
            "guess at produces a run that succeeds with the wrong "
            "parameters.".format(", ".join(undocumented), DOC_ARGUMENTS)
        )
    return entries


# What a `Path` argument accepts, declared beside the layout: `ACCEPTS = {"ios":
# SURFACE_EXTENSIONS, "cbct": catalogs.SCAN_EXTENSIONS}`.
#
# Not presentation, which is why it is NOT a LAYOUT key: the server carries it
# to `ArgSpec.accepts` and refuses an upload that contradicts it. It is the
# tool saying what it can read, in one place, and the client narrowing its file
# dialog and its scene entries from the same sentence -- until this existed,
# `ios` offered a folder of CBCT volumes as intraoral surfaces and the mistake
# only surfaced as a refusal a minute into the run.
#
# DERIVED, never retyped: the tables already exist (`pipeline.SURFACE_EXTENSIONS`,
# `catalogs.SCAN_EXTENSIONS`), and a second spelling of them is a second thing
# to keep in step.
ACCEPTS_NAME = "ACCEPTS"


def accepts_for(module, arguments):
    """`{argument: [".vtk", ".stl"]}` from the tool's own `ACCEPTS`, checked.

    Absent is the ordinary case. Present, an argument that does not exist or is
    not a path is a hard error: a filter naming the wrong argument narrows
    nothing and nobody would notice.
    """
    declared = getattr(module, ACCEPTS_NAME, None)
    if declared is None:
        return {}
    if not isinstance(declared, dict):
        raise SchemaError("{} must be a dict of argument -> extensions.".format(ACCEPTS_NAME))

    checked = {}
    for name, extensions in declared.items():
        where = "{}['{}']".format(ACCEPTS_NAME, name)
        if name not in arguments:
            raise SchemaError("{}: run() has no argument '{}'.".format(where, name))
        if arguments[name].get("type") != "path":
            raise SchemaError(
                "{}: only a path argument accepts extensions; '{}' is declared as "
                "{!r}.".format(where, name, arguments[name].get("type"))
            )
        if isinstance(extensions, str) or not extensions:
            raise SchemaError(
                "{}: expected a non-empty sequence of extensions, got {!r}.".format(
                    where, extensions)
            )
        cleaned = []
        for extension in extensions:
            if not isinstance(extension, str) or not extension.startswith("."):
                raise SchemaError(
                    "{}: '{}' is not an extension -- they start with a dot.".format(
                        where, extension)
                )
            cleaned.append(extension.lower())
        checked[name] = sorted(set(cleaned))
    return checked


def layout_for(package, arguments):
    """The tool's optional panel layout, checked against what it publishes.

    Absent is the ordinary case and means no hints: the schema is exactly what
    it was before this existed. Present, every key is verified -- an argument
    that does not exist, an option that is not offered, or a key outside
    LAYOUT_KEYS is a hard error rather than a hint the client silently drops.
    """
    try:
        module = importlib.import_module("{}.{}".format(package, LAYOUT_MODULE))
    except ImportError:
        return {}

    declared = getattr(module, "LAYOUT", None)
    if declared is None:
        raise SchemaError("{}.{} defines no LAYOUT.".format(package, LAYOUT_MODULE))
    if not isinstance(declared, dict):
        raise SchemaError("{}.{}.LAYOUT must be a dict.".format(package, LAYOUT_MODULE))

    for name, hints in declared.items():
        where = "layout for '{}'".format(name)
        if name not in arguments and name not in INJECTED_ARGUMENTS:
            raise SchemaError(
                "{}: run() has no argument '{}'. A layout may only describe "
                "arguments that exist, or one of the injected {}.".format(
                    where, name, ", ".join(INJECTED_ARGUMENTS)
                )
            )
        if not isinstance(hints, dict):
            raise SchemaError("{}: must be a dict of hints.".format(where))
        unknown = sorted(set(hints) - set(LAYOUT_KEYS))
        if unknown:
            raise SchemaError(
                "{}: unknown key(s) {}. A layout sets only {}.".format(
                    where, ", ".join(unknown), ", ".join(LAYOUT_KEYS)
                )
            )
        if name in arguments:
            _check_groups(where, hints, arguments[name])
        _check_visible_when(where, hints, arguments)
    return declared


def _check_groups(where, hints, argument):
    """Every option a group mentions must be one the argument offers.

    This is the check that replaces the drift: the old tables listed options by
    hand and a landmark added to the catalog was unreachable through any tab.
    """
    groups = hints.get("groups")
    if groups is None:
        return
    if not isinstance(groups, dict):
        raise SchemaError("{}: 'groups' must be a dict of tab -> options.".format(where))
    offered = set(argument.get("choices") or ())
    if not offered:
        raise SchemaError(
            "{}: 'groups' needs an argument with choices; this one has none.".format(where)
        )
    for tab, options in groups.items():
        missing = [option for option in options if option not in offered]
        if missing:
            raise SchemaError(
                "{}: tab '{}' lists option(s) the argument does not offer: "
                "{}.".format(where, tab, ", ".join(str(m) for m in missing))
            )


def _check_visible_when(where, hints, arguments):
    """A condition must name a real argument, and a value it can actually take."""
    conditions = hints.get("visible_when")
    if conditions is None:
        return
    if not isinstance(conditions, dict):
        raise SchemaError("{}: 'visible_when' must be a dict of argument -> value.".format(where))
    for other, expected in conditions.items():
        if other not in arguments:
            raise SchemaError(
                "{}: 'visible_when' names '{}', which run() does not take.".format(where, other)
            )
        offered = arguments[other].get("choices")
        if not offered:
            continue
        wanted = expected if isinstance(expected, (list, tuple)) else [expected]
        missing = [value for value in wanted if value not in offered]
        if missing:
            raise SchemaError(
                "{}: 'visible_when' expects '{}' to be {}, which it never is. It "
                "offers: {}.".format(
                    where, other, ", ".join(str(m) for m in missing), ", ".join(offered)
                )
            )


def describe_run(run, package=None):
    """Turn run()'s signature and docstring into the published schema."""
    doc = inspect.getdoc(run)
    if not doc or not doc.strip():
        raise SchemaError(
            "run() has no docstring. Its first line is the tool description the "
            "client shows a clinician, so it cannot be left out."
        )

    signature = inspect.signature(run)
    try:
        hints = typing.get_type_hints(run)
    except Exception as error:  # a forward reference that does not resolve
        raise SchemaError("cannot resolve run()'s annotations: {}".format(error))

    arguments = {}
    supervisor = False
    for name, parameter in signature.parameters.items():
        if is_supervisor(name, parameter, hints):
            # Excluded from `arguments` entirely: it is not something a client
            # can send, so publishing it would put a control on every form.
            supervisor = True
            continue
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            raise SchemaError(
                "run() takes *{}. The runner calls run(**params) from a JSON "
                "object, so every argument must be named.".format(name)
            )
        if parameter.kind is parameter.POSITIONAL_ONLY:
            raise SchemaError(
                "argument '{}' is positional-only and cannot be passed by "
                "name.".format(name)
            )
        if name not in hints:
            raise SchemaError("argument '{}' has no annotation.".format(name))

        kind, choices = type_name(hints[name], "argument '{}'".format(name))
        # The absence of a default is the ONLY thing that makes an argument
        # required. There is no `required=` to contradict it.
        required = parameter.default is parameter.empty
        arguments[name] = {"type": kind, "required": required}
        if not required:
            arguments[name]["default"] = check_default(
                name, kind, parameter.default, choices
            )
        if choices is not None:
            # Last, so adding options to an argument does not reorder the keys
            # a reader is used to. `list[...]` means several may be chosen, a
            # bare scalar means exactly one.
            arguments[name]["choices"] = choices

    described = {
        "description": doc.strip().splitlines()[0].strip(),
        "arguments": arguments,
        "returns": return_name(hints.get("return", signature.return_annotation)),
    }
    # The help text, then presentation, both merged INTO the arguments rather
    # than sitting beside them: a client reads one spec per argument, and a
    # second place to look is a second place to forget.
    for name, text in argument_docs(doc, arguments).items():
        arguments[name]["description"] = text
    # The module `run` was defined in -- `describe_run` is handed the function,
    # not the package, and `ACCEPTS` sits beside it.
    for name, extensions in accepts_for(inspect.getmodule(run), arguments).items():
        arguments[name]["extensions"] = extensions

    injected_layout = {}
    for name, hints in (layout_for(package, arguments).items() if package else []):
        if name in arguments:
            arguments[name].update(hints)
        else:
            # An argument this tool does not declare: the server adds it, so its
            # presentation travels beside the arguments rather than inside them.
            injected_layout[name] = hints
    if injected_layout:
        described["injected_layout"] = injected_layout

    paired = paired_for(inspect.getmodule(run), arguments)
    if paired:
        described["paired"] = paired

    if supervisor:
        # Published so the server can tell, before accepting a job, that this
        # tool needs something to be injected. A deployment whose runner cannot
        # supply one has to refuse the tool rather than call it and have the
        # call fail halfway through.
        described["supervisor"] = True
    return described


# Which path inputs hold the same patients, declared beside `run()`:
# `PAIRED = {"axes": ["t1", "t2"]}`, with a `pairs(**{axis: [names]})` that
# says which listed files go together. Published so a client splitting a
# cohort into batches splits every axis by the SAME patients -- the server
# relays the question to `pairs()` and knows nothing of how a tool pairs.
PAIRED_NAME = "PAIRED"
PAIRS_FUNCTION = "pairs"


def paired_for(module, arguments):
    """`{"axes": [...]}` from the tool's own `PAIRED`, checked, or None."""
    declared = getattr(module, PAIRED_NAME, None)
    if declared is None:
        return None
    if not isinstance(declared, dict) or set(declared) - {"axes"}:
        raise SchemaError("{} must be {{'axes': [argument, ...]}}.".format(PAIRED_NAME))
    axes = declared.get("axes")
    if not isinstance(axes, (list, tuple)) or len(axes) < 2 or len(set(axes)) != len(axes):
        raise SchemaError("{}['axes'] must name two or more distinct arguments.".format(PAIRED_NAME))
    for axis in axes:
        spec = arguments.get(axis)
        if spec is None:
            raise SchemaError("{}: run() has no argument '{}'.".format(PAIRED_NAME, axis))
        if spec.get("type") != "path" or not spec.get("required"):
            raise SchemaError(
                "{}: '{}' must be a required path argument to be split.".format(PAIRED_NAME, axis))
    pairs = getattr(module, PAIRS_FUNCTION, None)
    if not inspect.isfunction(pairs):
        raise SchemaError(
            "{} is declared but the module defines no {}() to answer which files "
            "go together.".format(PAIRED_NAME, PAIRS_FUNCTION))
    missing = [axis for axis in axes if axis not in inspect.signature(pairs).parameters]
    if missing:
        raise SchemaError("{}() must take every axis; it lacks {}.".format(
            PAIRS_FUNCTION, ", ".join(missing)))
    return {"axes": list(axes)}


def source_hash(src_dir):
    """sha256 of src/, so the server can spot a cached schema gone stale.

    The relative path goes into the digest as well as the bytes: renaming a
    module changes what gets imported, so it must change the hash.
    """
    digest = hashlib.sha256()
    for path in sorted(p for p in src_dir.rglob("*") if p.is_file()):
        if "__pycache__" in path.parts or path.suffix in (".pyc", ".pyo"):
            continue
        digest.update(path.relative_to(src_dir).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def find_package(src_dir):
    """The one importable package under src/."""
    if not src_dir.is_dir():
        raise SchemaError("no src/ directory in {}.".format(src_dir.parent))

    candidates = sorted(
        p.name for p in src_dir.iterdir() if p.is_dir() and (p / "__init__.py").is_file()
    )
    if len(candidates) != 1:
        raise SchemaError(
            "expected exactly one package under {}, found {}. Pass --module to "
            "choose.".format(src_dir, ", ".join(candidates) or "none")
        )
    return candidates[0]


def load_run(src_dir, module_name):
    """Import the tool and hand back its run().

    src/ goes on the path so this works whether or not the project itself is
    installed in the venv; the venv is still what provides the dependencies.
    """
    sys.path.insert(0, str(src_dir))
    try:
        module = __import__(module_name)
    except ImportError as error:
        raise SchemaError(
            "cannot import {}: {}.\nIf this names a heavy dependency (torch, "
            "monai, nnunetv2...), the tool imports it at module level. Move it "
            "inside run(): schema generation must not cost a CUDA stack.".format(
                module_name, error
            )
        )

    run = getattr(module, "run", None)
    if run is None:
        raise SchemaError("{} defines no run().".format(module_name))
    if not inspect.isfunction(run):
        raise SchemaError("{}.run is {!r}, not a function.".format(module_name, type(run)))
    return run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "tool_dir",
        nargs="?",
        default=".",
        type=Path,
        help="the tool package directory, e.g. tools/AMASSS (default: cwd)",
    )
    parser.add_argument("--module", help="package to import, if src/ holds more than one")
    parser.add_argument("-o", "--output", type=Path, help="write here instead of stdout")
    args = parser.parse_args(argv)

    tool_dir = args.tool_dir.resolve()
    src_dir = tool_dir / "src"
    try:
        module_name = args.module or find_package(src_dir)
        schema = {"name": tool_dir.name}
        schema.update(describe_run(load_run(src_dir, module_name), module_name))
        # Only for a tool that takes a supervisor: for any other, an attribute
        # called `.run` on something named `sup` would be a coincidence, and
        # publishing a call list it cannot make would have the server check a
        # relationship that does not exist.
        if schema.get("supervisor"):
            calls = supervised_calls(src_dir)
            if calls:
                schema["calls"] = calls
            stops = quality_controls(src_dir)
            if stops:
                schema["quality_controls"] = stops
        # Outside that guard, unlike the two above: what a tool calls its
        # outputs has nothing to do with whether it takes a supervisor, and
        # the tool this matters most for is a LEAF. ALI drives nobody and it
        # is its `_lm_Pred` that says which landmarks belong to which patient.
        suffixes = output_suffixes(src_dir)
        if suffixes:
            schema["output_suffixes"] = suffixes
        kind = review_kind(src_dir)
        if kind:
            schema["review_kind"] = kind
        narrows = case_input(src_dir, schema.get("arguments") or {})
        if narrows:
            schema["case_input"] = narrows
        schema["source_hash"] = source_hash(src_dir)
    except SchemaError as error:
        sys.stderr.write("{}: {}\n".format(tool_dir.name, error))
        return 2

    text = json.dumps(schema, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
