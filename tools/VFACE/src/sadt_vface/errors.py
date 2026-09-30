"""The exceptions that cross the process boundary.

Duplicated in every tool rather than shared, and deliberately: errors cross by
exception class NAME -- the runner records the name, the server maps it to an
HTTP status -- so a shared base class is not merely unnecessary, it is not the
mechanism. See CONTRIBUTING.md.
"""


class ToolInputError(ValueError):
    """What the caller sent cannot be worked with. Answered as a 422."""


class SupervisorRequired(RuntimeError):
    """This mode needs another tool, and no supervisor was injected.

    Answered as a 501. It names the mode that works without one, because
    "deploy a tool" is not an answer a clinician can act on.
    """
