"""Check what the model extracted against what the tool actually declares.

The catalogue is generated from each tool's own `run()` signature, so unlike
upstream's hand-written manifest these checks are against the real thing: a
`choices` list here is a `Literal[...]` in the callee's signature, and a value
outside it is a 422 the caller would have received anyway. That is the whole
argument for taking the catalogue from the registry.

Three checks upstream had are deliberately NOT here:

- **The folder-extension check.** `parameter_validator.py:191` rejected any
  argument whose NAME contained "folder" or "dir" if its value ended in one of
  seven extensions -- a list missing `.nrrd`, `.mha`, `.gipl`, `.dcm`, `.vtp`,
  `.obj` and `.off`, every one of which the tools accept. So it was a guess
  about the argument's meaning, checked against an incomplete guess about the
  file's. The schema says `path` and does not distinguish a file from a
  directory; nothing here pretends otherwise, and the tool itself gives the
  precise error if the path is the wrong kind.
- **`min` / `max`.** `parameter_validator.py:233,242` read fields no manifest
  entry ever set, and `scripts/describe.py` cannot emit them: a numeric bound is
  not part of the published vocabulary. Checking a field that never exists is
  not a check.
- **Filling in defaults.** `utils.complete_with_defaults` copied every optional
  parameter's manifest default into the proposal, keyed by `p.get("name", "")`
  -- so every parameter that happened to lack a name collapsed into a single
  `""` key and was injected as a phantom argument no tool declares. The
  defaults are in the callee's own signature; restating them in the proposal
  adds a second copy that can only drift, and makes it impossible to see which
  values the *user* actually asked for. What is proposed here is what was
  extracted, and nothing else.
"""

from .errors import ToolInputError

TRUE_WORDS = ("true", "yes", "y", "1", "on")
FALSE_WORDS = ("false", "no", "n", "0", "off")


def validate(tool, arguments, extracted):
    """`(values, errors, unknown)` from what the model proposed.

    Never raises on the model's output: a badly extracted parameter is a
    reportable fact about this proposal, not a failure of the run. Errors are
    carried into `routing.json` and the offending value is dropped, so what is
    proposed is only ever what passed.
    """
    if not isinstance(extracted, dict):
        return {}, ["The model's 'extracted' field was not an object."], []

    values = {}
    errors = []
    unknown = []

    for name, value in extracted.items():
        spec = arguments.get(name)
        if spec is None:
            # Dropped, not passed on. The server validates against the same
            # schema and answers 422 for an unknown argument, so forwarding one
            # only moves the failure to a place with less context.
            unknown.append(name)
            continue
        if value is None:
            # `null` is how a model says "not stated". There is no nullable
            # type in the contract, so it is an omission, not a value.
            continue
        try:
            values[name] = coerce(name, spec, value)
        except ToolInputError as exc:
            errors.append(str(exc))

    if unknown:
        errors.append(
            "The model proposed {} for '{}', which it does not take. Dropped.".format(
                ", ".join(repr(name) for name in sorted(unknown)), tool["name"]
            )
        )
    return values, errors, sorted(unknown)


def coerce(name: str, spec, value):
    """One value, as the argument's declared type. Raises `ToolInputError`."""
    declared = spec["type"]
    if declared == "vec2":
        return _vec2(name, spec, value)
    if declared.startswith("list["):
        element = declared[len("list["):-1]
        items = _as_list(name, value)
        coerced = [_scalar(name, element, item) for item in items]
        _check_choices(name, spec, coerced, per_item=True)
        return coerced
    coerced = _scalar(name, declared, value)
    _check_choices(name, spec, coerced, per_item=False)
    return coerced


def _as_list(name: str, value):
    """A list, from a list or from the separated string a model tends to send."""
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        if not text:
            return []
        separator = "," if "," in text else None
        return [item.strip().strip("'\"") for item in text.split(separator) if item.strip()]
    if isinstance(value, (int, float, bool)):
        return [value]
    raise ToolInputError(
        "'{}' takes a list, and the model answered {!r}.".format(name, value)
    )


def _scalar(name: str, declared: str, value):
    if declared == "bool":
        return _boolean(name, value)
    if declared == "int":
        return _integer(name, value)
    if declared == "float":
        return _number(name, value)
    if declared in ("str", "path"):
        if isinstance(value, bool) or isinstance(value, (list, tuple, dict)):
            raise ToolInputError(
                "'{}' takes a {}, and the model answered {!r}.".format(
                    name, declared, value
                )
            )
        text = str(value).strip()
        if declared == "path" and not text:
            raise ToolInputError(
                "'{}' is a path and the model answered an empty string.".format(name)
            )
        return text
    raise ToolInputError(
        "'{}' is of type {!r}, which this agent cannot fill.".format(name, declared)
    )


def _vec2(name: str, spec, value):
    """Exactly two numbers, inside the ranges the tool declared for each axis."""
    items = _as_list(name, value)
    if len(items) != 2:
        raise ToolInputError(
            "'{}' takes exactly two numbers, and the model answered {!r}.".format(
                name, value
            )
        )
    numbers = [_number(name, item) for item in items]
    for number, axis in zip(numbers, ("x_range", "y_range")):
        bounds = spec.get(axis)
        if not bounds or len(bounds) != 2:
            continue
        low, high = min(bounds), max(bounds)
        if not low <= number <= high:
            raise ToolInputError(
                "'{}' must lie within {} on its {} axis. The model answered "
                "{!r}.".format(name, list(bounds), axis[0], number)
            )
    return numbers


def _boolean(name: str, value):
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in TRUE_WORDS:
            return True
        if lowered in FALSE_WORDS:
            return False
    raise ToolInputError(
        "'{}' is a true/false argument, and the model answered {!r}.".format(name, value)
    )


def _integer(name: str, value):
    if isinstance(value, bool):
        raise ToolInputError(
            "'{}' is a whole number, and the model answered {!r}.".format(name, value)
        )
    if isinstance(value, int):
        return value
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        raise ToolInputError(
            "'{}' is a whole number, and the model answered {!r}.".format(name, value)
        )
    if number != int(number):
        raise ToolInputError(
            "'{}' is a whole number, and the model answered {!r}.".format(name, value)
        )
    return int(number)


def _number(name: str, value):
    if isinstance(value, bool):
        raise ToolInputError(
            "'{}' is a number, and the model answered {!r}.".format(name, value)
        )
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        raise ToolInputError(
            "'{}' is a number, and the model answered {!r}.".format(name, value)
        )


def _check_choices(name: str, spec, value, per_item: bool) -> None:
    """A fixed set of options is real now, so it is enforced.

    `choices` comes from a `Literal[...]` in the callee's own signature. Every
    item of a list argument is checked, not just the list: `list[Literal[...]]`
    is several-of, and one bad option is a 422 for the whole request.
    """
    choices = spec.get("choices")
    if not choices:
        return
    offending = [item for item in value if item not in choices] if per_item else (
        [] if value in choices else [value]
    )
    if offending:
        raise ToolInputError(
            "'{}' must be {}. The model answered {}.".format(
                name,
                "one of: " + ", ".join(repr(choice) for choice in choices),
                ", ".join(repr(item) for item in offending),
            )
        )
