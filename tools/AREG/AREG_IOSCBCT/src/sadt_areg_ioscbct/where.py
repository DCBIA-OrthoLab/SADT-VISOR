"""Which patient and which arch a log line is about.

The registration's own lines -- "Pre-alignment accepted", "Only 40% of the
intraoral scan is labelled as crowns" -- are written deep in `geometry` and
`volume`, which know nothing of the batch. Read on the operator page they said
WHAT happened and never to WHICH of forty arches. Threading a LoggerAdapter
through every function down there would change a dozen signatures for one
prefix, so the position is set once, by the loop that knows it, and a filter on
each module's logger puts it in front of the message.

A position, never a name: "patient 3/12, arch U", not the file it came from.
"""

import contextlib
import contextvars
import logging

_position = contextvars.ContextVar("sadt_areg_ioscbct_position", default="")


class _Prefix(logging.Filter):
    """Puts the current position in front of every record of one logger."""

    def filter(self, record):
        position = _position.get()
        if position and isinstance(record.msg, str):
            record.msg = f"{position}: {record.msg}"
        return True


_PREFIX = _Prefix()


def attach(logger: logging.Logger) -> logging.Logger:
    """Prefix `logger`'s lines with the position. Idempotent."""
    if _PREFIX not in logger.filters:
        logger.addFilter(_PREFIX)
    return logger


@contextlib.contextmanager
def at(position: str):
    """Every line logged inside the block carries `position`."""
    token = _position.set(position)
    try:
        yield
    finally:
        _position.reset(token)
