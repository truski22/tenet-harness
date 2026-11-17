"""Execution of a single run, from accepted request to a finished, persisted outcome."""
import logging
from threading import Event
from typing import Any, Callable
from time import monotonic

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from ..agents import AgentDefinition, Registry
from ..errors import BudgetExceeded, Cancelled, LoopDetected, Suspended
from ..messages import Messages
from ..providers import Provider
from ..runs import RunContext, RunRequest
from ..storage import RunStore
from .binding import ToolsetBinder, select_toolsets
from .budget import UsageMeter
from .events import EventEmitter
from .gate import ModelGate
from .governance import Governance
from .graph import ModelNode, build_graph
from .history import last_assistant_reply, thread_messages
from .tool_executor import OUTCOME_UNKNOWN, ToolExecutor

log = logging.getLogger(__name__)


class RunExecutor:
    """Runs one request to completion. Stateless between runs, so many can execute concurrently."""

    def __init__(self, registry: Registry, provider: Provider, store: RunStore, checkpointer, gate: ModelGate,
                 messages: Messages, config_for: Callable[[str], RunnableConfig], max_tools: int, run_timeout: float, context_window: int,
                 governance: Governance, token_limit: int | None = None):
        self.registry, self.provider, self.store, self.checkpointer = registry, provider, store, checkpointer
        self.gate, self.messages, self.config_for = gate, messages, config_for
        self.max_tools, self.run_timeout, self.context_window = max_tools, run_timeout, context_window
        self.governance, self.token_limit = governance, token_limit

    def run(self, request: RunRequest, cancel: Event, resume: dict | None = None) -> None:
        """Execute a run. With ``resume`` (a decided approval record) continue a suspended run instead."""
        run_id, thread_id = request.run_id, request.thread_id
        emit = EventEmitter(self.store, run_id, thread_id)
        context = RunContext(run_id, thread_id, request.text, emit, cancel, monotonic() + self.run_timeout)
        meter = UsageMeter(self.token_limit, self.store.run(run_id) if resume else None)
        try:
            self.store.set_status(run_id, "running")
            emit("resumed" if resume else "start")
            reply, previous = self._execute(request, context, meter, resume)
            emit("reply", text=reply)
            agent = self.registry.get(request.agent_id)
            if agent.memorize:
                self._queue_memory(agent, request, previous, emit)
            emit("done", status="completed", data=meter.summary())
            self.store.finish(run_id, "completed", reply=reply)
        except Suspended:
            emit("suspended")  # the run stays awaiting_approval; resolving the approval resumes it
        except Cancelled:
            emit("done", status="cancelled", message=self.messages.cancelled, data=meter.summary())
            self.store.finish(run_id, "cancelled")
        except (LoopDetected, BudgetExceeded) as exc:
            kind, text = (("loop_detected", self.messages.loop_detected) if isinstance(exc, LoopDetected)
                          else ("budget_exceeded", self.messages.budget_exceeded))
            log.warning("Run %s stopped: %s (%s)", run_id, exc.code, exc)
            self.store.audit(run_id, thread_id, kind, None, {"detail": str(exc)})
            emit(kind, message=text)
            emit("error", message=text)
            emit("done", status="failed", data=meter.summary())
            self.store.finish(run_id, "failed", error=text)
        except Exception as exc:
            # Exception messages may contain prompt fragments; retain type and correlation ID only.
            log.error("Run %s failed (%s)", run_id, type(exc).__name__)
            emit("error", message=self.messages.run_failed)
            emit("done", status="failed", data=meter.summary())
            self.store.finish(run_id, "failed", error=self.messages.run_failed)
        finally:
            self.store.record_usage(run_id, meter.input_tokens, meter.output_tokens, meter.calls)

    def _execute(self, request: RunRequest, context: RunContext, meter: UsageMeter,
                 resume: dict | None = None) -> tuple[str, str]:
        """Prepare the turn, run the agent graph; returns (final reply, previous assistant reply)."""
        agent = self.registry.get(request.agent_id)
        config = self.config_for(request.thread_id)
        history = thread_messages(self.checkpointer, config)
        turn = next((i for i, m in enumerate(history) if m.id == request.run_id), len(history))
        previous = last_assistant_reply(history[:turn])
        context.check()
        policy_reply = agent.preflight(context) if agent.preflight and not resume else None
        base_prompt = "Runtime policy response."
        if not policy_reply:
            prompt = agent.context(context).content
            if not isinstance(prompt, str):
                raise TypeError("Agent context must be plain text.")
            base_prompt = prompt
        toolsets = {ts.id: ts for ts in agent.toolsets if ts.always_on or ts.available()}
        binder = ToolsetBinder(self.provider, toolsets, base_prompt, self.context_window,
                               expose_meta_tool=not policy_reply)
        active = [] if policy_reply else select_toolsets(agent, toolsets, request.text, history)
        binder.fit(active)
        log.info("Run %s toolsets: %s", request.run_id, ",".join(binder.binding.active) or "-")

        executor = ToolExecutor(context, binder, toolsets, self.max_tools, self.governance)
        graph = build_graph(ModelNode(context, binder, agent, self.gate, policy_reply, meter), executor,
                            self.checkpointer)
        if resume:
            self._answer_open_calls(graph, config, executor, resume, context)
            return self._stream(graph, config, None, context), previous
        self._repair_interrupted(graph, config)
        return self._stream(graph, config, {"messages": [HumanMessage(content=request.text, id=request.run_id)]},
                            context), previous

    @staticmethod
    def _outstanding(graph, config: RunnableConfig) -> dict[Any, Any]:
        """Tool calls in the checkpoint that no tool message has answered, in call order."""
        outstanding: dict[Any, Any] = {}
        for message in graph.get_state(config).values.get("messages", []):
            if isinstance(message, AIMessage):
                outstanding.update({c["id"]: c for c in message.tool_calls})
            elif isinstance(message, ToolMessage):
                outstanding.pop(message.tool_call_id, None)
        return outstanding

    @classmethod
    def _repair_interrupted(cls, graph, config: RunnableConfig) -> None:
        """Close tool calls left open by a crashed run so the model never replays a side effect."""
        outstanding = cls._outstanding(graph, config)
        if outstanding:
            graph.update_state(config, {"messages": [
                ToolMessage(content=OUTCOME_UNKNOWN, tool_call_id=key, status="error")
                for key in outstanding]}, as_node="tools")

    @classmethod
    def _answer_open_calls(cls, graph, config: RunnableConfig, executor: ToolExecutor, approval: dict,
                           context: RunContext) -> None:
        """Apply a decided approval to the suspended turn so the graph can continue from its checkpoint."""
        outstanding = cls._outstanding(graph, config)
        if approval["tool_call_id"] not in outstanding:
            raise ValueError("The suspended tool call is no longer in the conversation.")
        context.emit("approval_resolved", name=approval["tool"], status=approval["status"],
                     data={"approval_id": approval["id"], "decided_by": approval["decided_by"]})
        graph.update_state(config, {"messages": executor.resume(list(outstanding.values()), approval)},
                           as_node="tools")

    @staticmethod
    def _stream(graph, config: RunnableConfig, payload: dict | None, context: RunContext) -> str:
        current = None
        for chunk, meta in graph.stream(payload, config, stream_mode="messages"):
            context.check()
            if meta.get("langgraph_node") != "agent" or not isinstance(chunk, AIMessageChunk):
                continue
            if chunk.id != current:
                current = chunk.id
                context.emit("reset")
            if isinstance(chunk.content, str) and chunk.content:
                context.emit("token", text=chunk.content)
        final = graph.get_state(config).values["messages"][-1]
        if not isinstance(final, AIMessage) or final.tool_calls \
                or not isinstance(final.content, str) or not final.content.strip():
            raise ValueError("No final text answer returned.")
        context.check()
        return final.content.strip()

    def _queue_memory(self, agent: AgentDefinition, request: RunRequest, previous: str, emit) -> None:
        try:
            self.store.queue_memory(request.run_id, agent.id, request.thread_id, request.text, previous)
            emit("memory_queued")
        except ValueError:
            log.warning("Memory queue full for run %s", request.run_id)
            emit("memory_failed", message=self.messages.memory_queue_full)
