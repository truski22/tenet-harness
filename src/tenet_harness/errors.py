"""Exceptions shared across the harness."""


class HarnessError(Exception):
    """Base of errors raised by the harness itself; ``code`` is stable, the message is user-facing."""
    code = "HARNESS_ERROR"


class BusyError(HarnessError):
    """The runtime cannot accept the request right now (capacity, conversation busy, exclusive use)."""
    code = "BUSY"


class LoopDetected(HarnessError):
    """The agent is repeating itself or failing repeatedly; the run is stopped."""
    code = "LOOP_DETECTED"


class BudgetExceeded(HarnessError):
    """The run used more model tokens than its budget allows."""
    code = "BUDGET_EXCEEDED"


class RunRejected(HarnessError, ValueError):
    """The request was refused before a run started. ``code`` is stable; the text comes from Messages."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class Cancelled(Exception):
    pass


class ToolError(Exception):
    """Expected tool failure; the message is safe to show to the model and the user."""
