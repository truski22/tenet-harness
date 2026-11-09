import json

import pytest
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from pydantic import BaseModel

from tenet_harness import (AgentDefinition, Effect, Registry, RunRequest, Tool, ToolError,
                                     ToolResult, Toolset)
from tenet_harness.routing import CombinedRouter, EmbeddingRouter, KeywordRouter
from tenet_harness.runtime import Runtime
from tenet_harness.storage import Store
from conftest import FakeProvider


class Empty(BaseModel):
    pass


def call(name, args=None, key="c1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args or {}, "id": key}])


def build(tmp_path, responses, external=None, available=True):
    calls = []

    def read(ctx, args):
        calls.append(ctx.user_text)
        return external(ctx, args) if external else ToolResult(status="success", data=[{"n": 1}])
    core = Toolset("core", "Core", (Tool("ping", "Ping", Empty, lambda c, a: ToolResult(status="success")),),
                   always_on=True)
    ext = Toolset("ext", "External code host", (Tool("ext_read", "Read", Empty, read),), external=True,
                  keywords=("gitlab", "merge request"), max_result_chars=400, instructions="Ext rules.",
                  available=lambda: available)
    registry = Registry()
    registry.register(AgentDefinition("a", "A", "", (core, ext), lambda ctx: SystemMessage(content="Base."),
                                      router=KeywordRouter()))
    provider = FakeProvider(responses)
    return Runtime(registry, provider, Store(tmp_path / "t.sqlite")), provider, calls


def run(runtime, text, run_id="r1", thread_id="t"):
    runtime.submit(RunRequest(run_id=run_id, thread_id=thread_id, agent_id="a", text=text))
    assert runtime.wait(run_id, 10)
    return runtime.store.run(run_id)


def bound_names(provider):
    return [schema["name"] for schema in provider.model_instance.tools]


def tool_messages(provider):
    return [m for m in provider.model_instance.calls[-1] if isinstance(m, ToolMessage)]


def test_unrelated_turn_only_binds_core_and_meta_tool(tmp_path):
    runtime, provider, _ = build(tmp_path, [AIMessage(content="Hola.")])
    try:
        assert run(runtime, "Hola, ¿qué tal?")["status"] == "completed"
        assert bound_names(provider) == ["ping", "use_toolset"]
        assert "Ext rules" not in provider.model_instance.calls[-1][0].content
    finally:
        runtime.close()


def test_keyword_routing_bounds_and_marks_external_results(tmp_path):
    many = lambda ctx, args: ToolResult(status="success", data=[{"title": "x" * 40}] * 50)  # noqa: E731
    runtime, provider, calls = build(tmp_path, [call("ext_read"), AIMessage(content="Listo.")], many)
    try:
        assert run(runtime, "¿Qué merge requests tengo en GitLab?")["status"] == "completed"
        assert "ext_read" in bound_names(provider) and "use_toolset" not in bound_names(provider)
        system = provider.model_instance.calls[-1][0].content
        assert "Ext rules." in system and "untrusted_data" in system
        content = json.loads(tool_messages(provider)[0].content)
        assert content["untrusted_data"] is True and content["source"] == "ext"
        assert content["data"]["truncated"] is True and content["data"]["omitted_items"] > 0
        assert len(tool_messages(provider)[0].content) < 600
    finally:
        runtime.close()


def test_meta_tool_enables_toolset_mid_run_and_followups_stay_routed(tmp_path):
    runtime, provider, calls = build(tmp_path, [
        call("ext_read", key="c0"), call("use_toolset", {"toolset": "ext"}), call("ext_read", key="c2"),
        AIMessage(content="Hecho."), AIMessage(content="Sigue."),
    ])
    try:
        assert run(runtime, "Revisa mi trabajo pendiente")["status"] == "completed"
        # The first call was rejected because ext was not active yet; the second ran after enabling it.
        assert len(calls) == 1
        assert run(runtime, "¿Y el segundo?", run_id="r2")["status"] == "completed"
        assert "ext_read" in bound_names(provider)
    finally:
        runtime.close()


def test_unavailable_toolset_is_never_bound(tmp_path):
    runtime, provider, _ = build(tmp_path, [AIMessage(content="No conectado.")], available=False)
    try:
        assert run(runtime, "mis merge request de gitlab")["status"] == "completed"
        assert bound_names(provider) == ["ping"]
    finally:
        runtime.close()


@pytest.mark.parametrize("failure, message", [
    (ToolError("Token caducado."), "Token caducado."),
    (RuntimeError("secret internals"), "unexpectedly"),
])
def test_tool_failures_become_error_results(tmp_path, failure, message):
    def fail(ctx, args):
        raise failure
    runtime, provider, _ = build(tmp_path, [call("ext_read"), AIMessage(content="Falló.")], fail)
    try:
        assert run(runtime, "pipeline de gitlab")["status"] == "completed"
        result = [e for e in runtime.store.events("r1") if e["type"] == "tool_result"][0]
        assert result["status"] == "error" and message in result["data"]["message"]
        assert "secret internals" not in tool_messages(provider)[0].content
    finally:
        runtime.close()


def test_registry_rejects_external_writes_and_reserved_names():
    write = Tool("w", "Write", Empty, lambda c, a: ToolResult(status="success"), Effect.WRITE)
    context = lambda ctx: SystemMessage(content="x")  # noqa: E731
    with pytest.raises(ValueError, match="read tools"):
        Registry().register(AgentDefinition("a", "A", "", (Toolset("e", "E", (write,), external=True),), context))
    meta = Tool("use_toolset", "x", Empty, lambda c, a: ToolResult(status="success"))
    with pytest.raises(ValueError, match="reserved"):
        Registry().register(AgentDefinition("a", "A", "", (Toolset("e", "E", (meta,)),), context))


def test_routers():
    gitlab = Toolset("gitlab", "code reviews", (), keywords=("mr", "merge request"),
                     examples=("¿MRs abiertos?", "¿falló el CI?"))
    calendar = Toolset("calendar", "meetings", ())
    assert KeywordRouter().select("¿Tengo MRs? mi MR", [gitlab, calendar]) == [gitlab]
    assert KeywordRouter().select("hmrm", [gitlab, calendar]) == []
    vectors = {"¿MRs abiertos?": [1.0, 0.0], "¿falló el CI?": [0.9, 0.3], "meetings": [0.0, 1.0],
               "¿reuniones hoy?": [0.1, 0.9], "¿se rompió el build?": [0.6, 0.8]}
    router = EmbeddingRouter(lambda texts: [vectors[t] for t in texts])
    assert router.select("¿reuniones hoy?", [gitlab, calendar]) == [calendar]
    # Best-matching example wins, and both toolsets may qualify.
    assert router.select("¿se rompió el build?", [gitlab, calendar]) == [gitlab, calendar]

    def offline(texts):
        raise ConnectionError
    combined = CombinedRouter(KeywordRouter(), EmbeddingRouter(offline))
    assert combined.select("merge request", [gitlab, calendar]) == [gitlab]
