"""Deterministic fakes for testing agents without a real model. Part of the public API."""
from langchain_core.messages import AIMessage


class FakeModel:
    def __init__(self, responses=None):
        self.responses = responses or [AIMessage(content="Respuesta local.")]
        self.calls = []

    def bind_tools(self, tools):
        self.tools = tools
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        return self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]


class FakeProvider:
    def __init__(self, responses=None):
        self.model_instance = FakeModel(responses)
        self.model = "fake"

    def chat(self, structured=False):
        return FakeModel([AIMessage(content='{"facts": []}')]) if structured else self.model_instance
