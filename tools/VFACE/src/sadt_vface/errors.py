"""The exceptions that cross the process boundary.

Duplicated in every tool rather than shared, and deliberately: errors cross by
exception class NAME -- the runner records the name, the server maps it to an
HTTP status -- so a shared base class is not merely unnecessary, it is not the
mechanism. See CONTRIBUTING.md.

Below them, the few helpers that turn a batch's failures into one short line:
the server keeps only the start of an error, and deletes the run report with
the job when the run fails, so the cause has to be said in the error itself.
"""

import re


class ToolInputError(ValueError):
    """What the caller sent cannot be worked with. Answered as a 422."""


class SupervisorRequired(RuntimeError):
    """This mode needs another tool, and no supervisor was injected.

    Answered as a 501. It names the mode that works without one, because
    "deploy a tool" is not an answer a clinician can act on.
    """


_QUOTED_PATH = re.compile(r"""(["'])[^"'\n]*[/\\][^"'\n]*\1""")
_BARE_PATH = re.compile(r"(?<![\w.])/[^\s\"',;:]+")


def describe_failure(failure) -> str:
    """`Type: first line of the message`, or the text itself for a recorded reason.

    Only the first line, and cut short: the server keeps the start of an error
    and drops the rest, so the cause has to come first and come briefly.
    """
    if isinstance(failure, BaseException):
        lines = [line.strip() for line in str(failure).splitlines() if line.strip()]
        # SimpleITK and ITK open with a header naming the C++ source line and
        # put the cause on the LAST line; a first line ending in a colon is
        # that header, and the cause is what follows it.
        if len(lines) > 1 and lines[0].endswith(":"):
            text = lines[-1]
        else:
            text = lines[0] if lines else ""
        text = f"{type(failure).__name__}: {text}" if text else type(failure).__name__
    else:
        text = str(failure)
    # Paths out, before counting: they would make every scan's identical
    # failure a different string -- so "most common" would always be 1 of N --
    # and they carry the scan's file name, which no line here may.
    text = _QUOTED_PATH.sub("<path>", text)
    text = _BARE_PATH.sub("<path>", text)
    return text if len(text) <= 180 else text[:177] + "..."


def most_common_failure(failures) -> str:
    """The commonest of `failures`, with how many of them it accounts for."""
    from collections import Counter

    counts = Counter(describe_failure(failure) for failure in failures)
    if not counts:
        return "none recorded"
    text, count = counts.most_common(1)[0]
    return f"{text} ({count} of {len(failures)})"


def nothing_succeeded(total: int, what: str, failures) -> Exception:
    """The error for a batch where not one of `total` items came through.

    A `ToolInputError` only when every failure was the caller's own input --
    the server answers that as a 422, "your fault" -- and a `RuntimeError`
    otherwise, because a batch that died on the server's side is not something
    the caller can fix by sending it again.
    """
    message = f"0 of {total} {what}; most common failure: {most_common_failure(failures)}"
    caller_fault = failures and all(
        isinstance(failure, (ValueError, FileNotFoundError)) for failure in failures
    )
    return ToolInputError(message) if caller_fault else RuntimeError(message)
