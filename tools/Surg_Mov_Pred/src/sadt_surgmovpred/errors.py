"""The failure a caller cannot fix by changing the request.

The server maps by exception class NAME: `ValueError`/`FileNotFoundError` reach
the caller as a 422 with the message passed through, `ToolUnavailableError` as a
503. Defined here rather than imported, since no package is shared between
tools -- see CONTRIBUTING.md.
"""


class ToolUnavailableError(RuntimeError):
    """The tool is installed but the models it ships with are not.

    A separate class because the fix is a deployment one, not a request one:
    no argument the caller changes makes a missing installed bundle appear.
    """
