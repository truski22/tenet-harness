"""Executing the tool calls the model requested, within budgets and output limits."""
import json
import logging

from langchain_core.messages import ToolMessage
from pydantic import BaseModel, ValidationError

from ..agents import META_TOOL
from ..errors import Cancelled, ToolError
from ..policy import Decision
from ..runs import RunContext
from ..tools import Tool, ToolResult, Toolset
from .approvals import APPROVED
from .binding import ToolsetBinder
from .governance import Governance
from .loops import LoopDetector

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
        parsed = tool.schema.model_validate(raw_args)
    except (ValidationError, ValueError) as exc:
        return ToolResult(status="clarification_required", message=str(exc))
    return run_tool(context, tool, parsed)


def run_tool(context: RunContext, tool: Tool, parsed: BaseModel) -> ToolResult:
    """Run a tool with already-validated arguments."""
    try:
        return tool.execute(context, parsed)
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
                 max_tools: int, governance: Governance):
        self.context, self.binder, self.max_tools = context, binder, max_tools
        self.governance = governance
        self.loops = LoopDetector(governance.loops) if governance.loops else None
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
            executed_args = None
            if call["name"] == META_TOOL:
                result = self.binder.activate(call["args"])
            elif call["name"] not in self.binder.binding.tools:
                result = ToolResult(status="error", message="Tool is not permitted for this agent.")
            else:
                result, executed_args = self._call(self.binder.binding.tools[call["name"]], source, call["args"])
            if source:
                result = bounded(result, source.max_result_chars)
            emit("tool_result", name=call["name"], status=result.status, data=result.model_dump())
            if executed_args is not None and self.loops:
                self.loops.record(call["name"], executed_args, result.status)
            if result.status == "success":
                context.successful_tools.add(call["name"])
            content = result.model_dump_json()
            if source and source.external:
                content = json.dumps({"source": source.id, "untrusted_data": True, **result.model_dump()},
                                     ensure_ascii=False)
            results.append(ToolMessage(content=content, tool_call_id=call["id"], name=call["name"],
                                       status="success" if result.status == "success" else "error"))
        return results

    def _call(self, tool: Tool, toolset: Toolset | None, raw_args) -> tuple[ToolResult, dict | None]:
        """Validate, apply loop guard and policy (with approval if required), then execute.

        Returns the result and, if the call counts as executed, its arguments (for loop accounting).
        """
        context, governance = self.context, self.governance
        try:
            parsed = tool.schema.model_validate(raw_args)
        except (ValidationError, ValueError) as exc:
            return ToolResult(status="clarification_required", message=str(exc)), None
        args = parsed.model_dump(mode="json")
        safe = governance.redact(args)
        audit = governance.store.audit
        if self.loops and (refusal := self.loops.check(tool.name, args)):
            context.emit("loop_detected", name=tool.name, message=refusal)
            audit(context.run_id, context.thread_id, "loop_blocked", tool.name, {"args": safe})
            return ToolResult(status="error", message=refusal), None
        decision = governance.policy.evaluate(tool, toolset, args)
        audit(context.run_id, context.thread_id, "tool_decision", tool.name,
              {"effect": tool.effect.value, "decision": decision.decision.value, "rule_id": decision.rule_id,
               "reason": decision.reason, "args": safe})
        if decision.decision == Decision.DENY:
            result = ToolResult(status="error", message=f"Blocked by policy: {decision.reason} Do not retry it.")
        elif decision.decision == Decision.REQUIRE_APPROVAL:
            status = governance.approvals.request(context, tool, safe, decision)
            audit(context.run_id, context.thread_id, "approval_" + status, tool.name, {"rule_id": decision.rule_id})
            if status == APPROVED:
                result = run_tool(context, tool, parsed)
            else:
                result = ToolResult(status="error", message=f"The action was not approved ({status}). "
                                    "Do not retry it or look for a way around it; tell the user.")
        else:
            result = run_tool(context, tool, parsed)
        audit(context.run_id, context.thread_id, "tool_outcome", tool.name, {"status": result.status})
        return result, args
