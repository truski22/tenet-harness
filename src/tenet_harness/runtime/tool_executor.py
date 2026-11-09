"""Executing the tool calls the model requested, within budgets and output limits."""
import json
import logging

from langchain_core.messages import ToolMessage
from pydantic import ValidationError

from ..agents import META_TOOL
from ..errors import Cancelled, ToolError
from ..runs import RunContext
from ..tools import Tool, ToolResult, Toolset
from .binding import ToolsetBinder

log = logging.getLogger(__name__)


def bounded(result: ToolResult, limit: int) -> ToolResult:
    """Keep tool output within a character budget so one call cannot flood the context."""
    if len(result.model_dump_json()) <= limit:
        return result
    if isinstance(result.data, list):
        kept = list(result.data)
        while kept:
            kept.pop()
            candidate = result.model_copy(update={"data": {
                "items": kept, "truncated": True, "omitted_items": len(result.data) - len(kept)}})
            if len(candidate.model_dump_json()) <= limit:
                return candidate
    partial = json.dumps(result.data, ensure_ascii=False)[:max(0, limit - len(result.message) - 200)]
    return result.model_copy(update={"data": {"truncated": True, "partial_json": partial}})


def execute_tool(context: RunContext, tool: Tool, raw_args) -> ToolResult:
    """Validate arguments and run one tool; expected failures become results, never exceptions."""
    try:
        return tool.execute(context, tool.schema.model_validate(raw_args))
    except (ValidationError, ValueError) as exc:
        return ToolResult(status="clarification_required", message=str(exc))
    except ToolError as exc:
        return ToolResult(status="error", message=str(exc))
    except (Cancelled, TimeoutError):
        raise
    except Exception as exc:
        log.error("Tool %s failed (%s)", tool.name, type(exc).__name__)
        return ToolResult(status="error", message="The tool failed unexpectedly. Do not retry it.")


class ToolExecutor:
    """Runs the tool calls of one model turn and turns results into tool messages."""

    def __init__(self, context: RunContext, binder: ToolsetBinder, toolsets: dict[str, Toolset],
                 max_tools: int):
        self.context, self.binder, self.max_tools = context, binder, max_tools
        self.owner = {t.name: ts for ts in toolsets.values() for t in ts.tools}
        self.used = 0

    def run(self, tool_calls: list[dict]) -> list[ToolMessage]:
        context, emit = self.context, self.context.emit
        results = []
        for call in tool_calls:
            context.check()
            self.used += 1
            if self.used > self.max_tools:
                raise ValueError("Tool execution budget exhausted.")
            context.tool_call_id = call["id"]
            emit("tool_start", name=call["name"])
            source = self.owner.get(call["name"])
            if call["name"] == META_TOOL:
                result = self.binder.activate(call["args"])
            elif call["name"] not in self.binder.binding.tools:
                result = ToolResult(status="error", message="Tool is not permitted for this agent.")
            else:
                result = execute_tool(context, self.binder.binding.tools[call["name"]], call["args"])
            if source:
                result = bounded(result, source.max_result_chars)
            emit("tool_result", name=call["name"], status=result.status, data=result.model_dump())
            if result.status == "success":
                context.successful_tools.add(call["name"])
            content = result.model_dump_json()
            if source and source.external:
                content = json.dumps({"source": source.id, "untrusted_data": True, **result.model_dump()},
                                     ensure_ascii=False)
            results.append(ToolMessage(content=content, tool_call_id=call["id"], name=call["name"],
                                       status="success" if result.status == "success" else "error"))
        return results
