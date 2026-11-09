"""Reading conversation state back from LangGraph checkpoints."""
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig


def thread_messages(checkpointer, config: RunnableConfig) -> list:
    checkpoint = checkpointer.get(config)
    return checkpoint.get("channel_values", {}).get("messages", []) if checkpoint else []


def visible_history(messages: list) -> list[dict]:
    """User messages and final assistant answers; tool traffic is not part of the visible history."""
    return [{"id": m.id, "role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
            for m in messages
            if isinstance(m, HumanMessage) or (isinstance(m, AIMessage) and m.content and not m.tool_calls)]


def last_assistant_reply(messages: list) -> str:
    return next((m["content"] for m in reversed(visible_history(messages)) if m["role"] == "assistant"), "")
