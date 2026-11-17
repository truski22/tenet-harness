"""Ask a model that has no JSON mode for JSON, and clean what comes back."""
import re
from typing import Any

from langchain_core.messages import AIMessage, SystemMessage

INSTRUCTION = "Respond with a single valid JSON object and nothing else: no prose, no code fences."
_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def strip_fences(text: str) -> str:
    match = _FENCE.match(text)
    return match.group(1) if match else text.strip()


class JsonOnly:
    """Wraps a chat model: appends a JSON-only instruction to the prompt and unwraps fenced output.

    It only exposes ``invoke``, which is all the structured calls (memory extraction and similar) need.
    """

    def __init__(self, model: Any):
        self.model = model

    def invoke(self, messages: list) -> AIMessage:
        prompt = [*messages, SystemMessage(content=INSTRUCTION)]
        reply = self.model.invoke(prompt)
        if isinstance(reply.content, str):
            return reply.model_copy(update={"content": strip_fences(reply.content)})
        return reply
