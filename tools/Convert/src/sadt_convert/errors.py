"""The failures a caller can do something about."""


class ToolInputError(ValueError):
    """An argument the tool cannot work with, phrased for whoever sent it."""
