"""The failures a caller or an operator can do something about.

Anything else raised by this tool is a bug and should surface as one. The
server maps by exception class NAME -- `ToolInputError`/`ValueError` to 422
with the message passed through, `ToolUnavailableError` to 503 -- so every
message raised with these is written to be read by whoever sent the request.
"""


class ToolInputError(ValueError):
    """The request cannot work, and the message says why to whoever sent it."""


class ToolUnavailableError(RuntimeError):
    """The tool is installed but the hardware it needs is not there.

    A separate class because the fix is a deployment one, not a request one: no
    argument the caller changes will make a CUDA device appear on a server that
    has none. The server answers 503 rather than 422.
    """
