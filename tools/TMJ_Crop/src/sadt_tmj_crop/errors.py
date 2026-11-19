"""The errors this tool raises on purpose."""


class ToolInputError(ValueError):
    """An input the tool cannot work with, phrased for whoever sent it.

    The runner turns this into a 422 for the client. Anything else surfaces as
    an internal error, so raise this for bad inputs and let genuine bugs crash.
    """


class SupervisorRequired(ToolInputError):
    """The tool needs ASO and AMASSS, and nothing here can run them.

    Its own class because nothing about the request is wrong: whatever runs the
    tool has to inject a supervisor, which a plain `uv run` never does.
    """
