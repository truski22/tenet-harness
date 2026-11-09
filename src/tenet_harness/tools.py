"""Tool contracts: the unit the model can call (Tool) and the unit of extension (Toolset)."""
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable, Literal

from pydantic import BaseModel

from .runs import RunContext


class ToolResult(BaseModel):
    status: Literal["success", "clarification_required", "error"]
    data: Any = None
    message: str = ""


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
