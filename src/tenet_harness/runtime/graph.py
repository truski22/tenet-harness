"""The agent loop as a LangGraph graph: model node, tools node, and the edge between them."""
from langchain_core.messages import AIMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langgraph.graph import END, START, MessagesState, StateGraph

from ..agents import AgentDefinition
from ..runs import RunContext
from .binding import ToolsetBinder
from .budget import UsageMeter
from .gate import ModelGate
from .tool_executor import ToolExecutor


class ModelNode:
    """One model call over the trimmed history, limited by the shared model gate."""

    def __init__(self, context: RunContext, binder: ToolsetBinder, agent: AgentDefinition,
                 gate: ModelGate, policy_reply: str | None, meter: UsageMeter):
        self.context, self.binder, self.agent = context, binder, agent
        self.gate, self.policy_reply, self.meter = gate, policy_reply, meter

    def __call__(self, state: MessagesState):
        context = self.context
        context.check()
        if self.policy_reply:
            return {"messages": [AIMessage(content=self.policy_reply)]}
        binding = self.binder.binding
        history = trim_messages(state["messages"], max_tokens=binding.budget,
                                token_counter=count_tokens_approximately, start_on="human", strategy="last")
        if not history or history[-1] != state["messages"][-1]:
            raise ValueError("The latest message exceeds the context budget.")
        self.meter.check()
        prompt = [binding.system, *history]
        with self.gate.slot(context.check):
            result = binding.model.invoke(prompt)
        context.check()
        self.meter.add(prompt, result)
        if result.tool_calls:
            self.meter.check()  # a final answer is never discarded for having crossed the limit
        if self.agent.finalize and not result.tool_calls and isinstance(result.content, str):
            checked = self.agent.finalize(context, result.content)
            if checked != result.content:
                context.emit("reset")
                result = result.model_copy(update={"content": checked})
        return {"messages": [result]}


def build_graph(model_node: ModelNode, tools: ToolExecutor, checkpointer):
    graph = StateGraph(MessagesState)
    graph.add_node("agent", model_node)
    graph.add_node("tools", lambda state: {"messages": tools.run(state["messages"][-1].tool_calls)})
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", lambda s: "tools" if s["messages"][-1].tool_calls else END)
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer)
