"""The failures this tool raises on purpose, by who can fix them.

The server maps by exception class NAME -- `ToolInputError`/`ValueError`/
`FileNotFoundError` to 422 with the message passed through, and
`ToolUnavailableError` to 503 -- so the class is chosen by whose fault the
failure is, and every message is written to be read by whoever sent the
request. Nothing here imports the server.
"""


class ToolInputError(ValueError):
    """An argument the tool cannot work with, phrased for whoever sent it."""


class ToolUnavailableError(RuntimeError):
    """The tool is installed but something it needs is not staged.

    A separate class because the fix is a deployment one: no argument the
    caller changes will make a missing backbone appear. The server answers 503
    rather than 422.
    """
