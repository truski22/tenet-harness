from dataclasses import dataclass, field
from enum import StrEnum
from threading import Event
from time import monotonic
from typing import Any, Callable, Literal, Protocol

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    status: Literal["success", "clarification_required", "error"]
    data: Any = None
    message: str = ""


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


class RunRejected(ValueError):
    """The request was refused before a run started. ``code`` is stable; the text comes from Messages."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


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


class Provider(Protocol):
    def chat(self, structured: bool = False) -> BaseChatModel: ...


class Cancelled(Exception):
    pass


class ToolError(Exception):
    """Expected tool failure; the message is safe to show to the model and the user."""


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


class Effect(StrEnum):
    READ = "read"
    WRITE = "write"
    DESTRUCTIVE = "destructive"


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    schema: type[BaseModel]
    execute: Callable[[RunContext, BaseModel], ToolResult]
    effect: Effect = Effect.READ


@dataclass(frozen=True)
class Toolset:
    """Unit of extension: a cohesive group of tools exposed to the model together.

    ``always_on`` toolsets are bound on every turn; the rest are selected per turn by the
    agent's router so the prompt only carries relevant schemas. ``external`` marks systems
    outside this machine: their results are untrusted data and their writes need approval.
    """
    id: str
    summary: str
    tools: tuple[Tool, ...]
    always_on: bool = False
    external: bool = False
    instructions: str = ""
    keywords: tuple[str, ...] = ()
    examples: tuple[str, ...] = ()  # sample user requests; drive semantic routing
    max_result_chars: int = 4000
    available: Callable[[], bool] = lambda: True


class Router(Protocol):
    def select(self, text: str, candidates: list[Toolset]) -> list[Toolset]: ...


@dataclass(frozen=True)
class AgentDefinition:
    id: str
    name: str
    description: str
    toolsets: tuple[Toolset, ...]
    context: Callable[[RunContext], SystemMessage]
    memorize: Callable[[dict], list[dict]] | None = None
    preflight: Callable[[RunContext], str | None] | None = None
    finalize: Callable[[RunContext, str], str] | None = None
    router: Router | None = None

    @property
    def tools(self) -> tuple[Tool, ...]:
        return tuple(tool for toolset in self.toolsets for tool in toolset.tools)


META_TOOL = "use_toolset"


class Registry:
    def __init__(self):
        self.agents: dict[str, AgentDefinition] = {}

    def register(self, agent: AgentDefinition):
        if agent.id in self.agents:
            raise ValueError(f"Duplicate agent: {agent.id}")
        if len({toolset.id for toolset in agent.toolsets}) != len(agent.toolsets):
            raise ValueError("Duplicate toolset ids")
        if len({tool.name for tool in agent.tools}) != len(agent.tools):
            raise ValueError("Duplicate tool names")
        if any(tool.name == META_TOOL for tool in agent.tools):
            raise ValueError(f"{META_TOOL} is reserved by the runtime")
        for toolset in agent.toolsets:
            if toolset.external and any(tool.effect != Effect.READ for tool in toolset.tools):
                # Writes to external systems must pass human approval; not available yet.
                raise ValueError(f"External toolset {toolset.id} may only expose read tools.")
        self.agents[agent.id] = agent

    def get(self, agent_id: str) -> AgentDefinition:
        return self.agents[agent_id]

    def list(self):
        return [{"id": a.id, "name": a.name, "description": a.description,
                 "tools": [t.name for t in a.tools],
                 "toolsets": [{"id": ts.id, "summary": ts.summary, "external": ts.external,
                               "available": ts.available()} for ts in a.toolsets]}
                for a in self.agents.values()]
