"""Run-level types: the request, the event stream and the per-run context handed to tools."""
from dataclasses import dataclass, field
from threading import Event
from time import monotonic
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field

from .errors import Cancelled


TERMINAL = frozenset({"completed", "failed", "cancelled", "interrupted"})


class RunEvent(BaseModel):
    version: Literal[1] = 1
    sequence: int
    run_id: str
    thread_id: str
    type: Literal["start", "reset", "token", "tool_start", "tool_result", "reply",
                  "memory_queued", "memory_saved", "memory_failed", "warning", "error", "done"]
    text: str | None = None
    message: str | None = None
    name: str | None = None
    status: str | None = None
    data: Any = None


class RunRequest(BaseModel):
    run_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,80}$")
    thread_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,80}$")
    agent_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,80}$")
    text: str = Field(min_length=1, max_length=16000)


@dataclass
class RunContext:
    run_id: str
    thread_id: str
    user_text: str
    emit: Callable[..., None]
    cancel: Event
    deadline: float
    allowed_reminders: set[str] = field(default_factory=set)
    tool_call_id: str = ""
    successful_tools: set[str] = field(default_factory=set)

    def check(self):
        if self.cancel.is_set():
            raise Cancelled("Run cancelled")
        if monotonic() >= self.deadline:
            raise TimeoutError("Run deadline exceeded")
