"""User-facing runtime texts, injectable so applications choose the language."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Messages:
    """User-facing runtime texts. Applications inject their own language; codes stay stable."""
    empty_message: str = "The message is empty."
    idempotency_conflict: str = "The run ID was already used with a different message."
    other_agent: str = "The conversation belongs to another agent."
    busy: str = "A response is already in progress."
    wait_for_run: str = "Wait for the current response to finish."
    run_failed: str = "The response could not be completed. Check the model provider and execution limits."
    cancelled: str = "Stopped. Actions already performed are kept."
    memory_queue_full: str = "Memory not saved: queue is full."
    memory_failed: str = "The memory for this message could not be saved."
