"""Agent definition and the registry that validates it."""
from dataclasses import dataclass
from typing import Callable

from langchain_core.messages import SystemMessage

from .runs import RunContext
from .routing.base import Router
from .tools import Tool, Toolset


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
        self.agents[agent.id] = agent

    def get(self, agent_id: str) -> AgentDefinition:
        return self.agents[agent_id]

    def list(self):
        return [{"id": a.id, "name": a.name, "description": a.description,
                 "tools": [t.name for t in a.tools],
                 "toolsets": [{"id": ts.id, "summary": ts.summary, "external": ts.external,
                               "available": ts.available()} for ts in a.toolsets]}
                for a in self.agents.values()]
