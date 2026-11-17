"""Provider adapters are built and configured without any network access."""
from langchain_core.messages import AIMessage, HumanMessage

from tenet_harness import (AgentDefinition, AnthropicProvider, Capabilities, Effect, OllamaProvider,
                           OpenAICompatibleProvider, Registry, RunRequest, Runtime, Tool, ToolResult, Toolset,
                           capabilities_of)
from tenet_harness.providers.json_only import INSTRUCTION, JsonOnly, strip_fences
from tenet_harness.testing import FakeProvider, InMemoryStore
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage
from pydantic import BaseModel


def test_providers_without_declared_capabilities_get_permissive_defaults():
    assert capabilities_of(object()) == Capabilities()
    assert capabilities_of(FakeProvider()).tool_calling is True


def test_ollama_declares_its_context_window():
    caps = capabilities_of(OllamaProvider("m", "http://localhost:11434", 5, context_window=16384))
    assert caps.context_window == 16384 and caps.json_mode


def test_openai_adapter_builds_a_configured_chat_model():
    provider = OpenAICompatibleProvider("gpt-x", base_url="http://localhost:8000/v1", api_key="k", timeout=7,
                                        context_window=32000)
    model = provider.chat()
    assert (model.model_name, model.openai_api_base, model.request_timeout, model.temperature) == \
        ("gpt-x", "http://localhost:8000/v1", 7, 0)
    assert model.stream_usage is True
    assert model.bind_tools([{"name": "t", "description": "d", "parameters": {"type": "object"}}])
    assert capabilities_of(provider).context_window == 32000


def test_openai_structured_calls_request_json_mode_when_supported():
    assert OpenAICompatibleProvider("m", api_key="k").chat(structured=True).model_kwargs == \
        {"response_format": {"type": "json_object"}}
    wrapped = OpenAICompatibleProvider("m", api_key="k", json_mode=False).chat(structured=True)
    assert isinstance(wrapped, JsonOnly)


def test_anthropic_adapter_builds_a_chat_model_and_wraps_structured_calls():
    provider = AnthropicProvider("claude-x", api_key="k", timeout=9, max_tokens=512)
    model = provider.chat()
    assert (model.model, model.max_tokens, model.default_request_timeout) == ("claude-x", 512, 9)
    assert isinstance(provider.chat(structured=True), JsonOnly)
    caps = capabilities_of(provider)
    assert caps.json_mode is False and caps.tool_calling and caps.context_window == 200_000


def test_strip_fences_handles_fenced_and_plain_json():
    assert strip_fences('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert strip_fences('```\n{"a": 1}\n```') == '{"a": 1}'
    assert strip_fences('  {"a": 1} ') == '{"a": 1}'


def test_json_only_appends_the_instruction_and_unwraps_the_reply():
    class Model:
        def invoke(self, messages):
            self.seen = messages
            return AIMessage(content='```json\n{"facts": []}\n```')

    model = Model()
    reply = JsonOnly(model).invoke([HumanMessage(content="hi")])
    assert reply.content == '{"facts": []}'
    assert model.seen[-1].content == INSTRUCTION and model.seen[0].content == "hi"


class Ping(BaseModel):
    pass


def agent():
    return AgentDefinition("a", "A", "", (Toolset("t", "T", (
        Tool("ping", "Ping", Ping, lambda ctx, args: ToolResult(status="success"), Effect.READ),),
        always_on=True),), lambda ctx: SystemMessage(content="Test."))


def runtime(provider, **options):
    registry = Registry()
    registry.register(agent())
    return Runtime(registry, provider, InMemoryStore(), checkpointer=MemorySaver(), **options)


def test_the_runtime_takes_its_context_window_from_the_provider():
    rt = runtime(FakeProvider(capabilities=Capabilities(context_window=32000)))
    try:
        assert rt._executor.context_window == 32000
    finally:
        rt.close()
    rt = runtime(FakeProvider(capabilities=Capabilities(context_window=32000)), context_window=4096)
    try:
        assert rt._executor.context_window == 4096  # an explicit setting wins
    finally:
        rt.close()
    rt = runtime(FakeProvider())
    try:
        assert rt._executor.context_window == 8192
    finally:
        rt.close()


def test_a_model_without_tool_calling_is_refused_for_an_agent_with_tools():
    outcomes = {}
    for supported in (True, False):
        rt = runtime(FakeProvider([AIMessage(content="hi")], Capabilities(tool_calling=supported)))
        try:
            rt.submit(RunRequest(run_id="r", thread_id="t", agent_id="a", text="go"))
            assert rt.wait("r", 5)
            outcomes[supported] = rt.store.run("r")["status"]
        finally:
            rt.close()
    assert outcomes == {True: "completed", False: "failed"}
