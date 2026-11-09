"""Exceptions shared across the harness."""


class RunRejected(ValueError):
    """The request was refused before a run started. ``code`` is stable; the text comes from Messages."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class Cancelled(Exception):
    pass


class ToolError(Exception):
    """Expected tool failure; the message is safe to show to the model and the user."""
