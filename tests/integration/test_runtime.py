import time

import pytest
from langchain_core.messages import AIMessage, SystemMessage
from pydantic import BaseModel

from tenet_harness import AgentDefinition, Registry, RunRequest, Tool, ToolResult, Toolset
from tenet_harness.runtime import Runtime
from tenet_harness.storage import Store
from conftest import FakeProvider


def wait(runtime, run_id):
    runtime.run_thread.join(timeout=10)
    assert not runtime.run_thread.is_alive()
    return runtime.store.run(run_id)


def test_second_agent_run_idempotency_and_isolation(tmp_path, provider):
    registry = Registry()
    registry.register(AgentDefinition("echo", "Echo", "Test", (), lambda ctx: SystemMessage(content="Echo.")))
    store = Store(tmp_path / "test.sqlite")
    runtime = Runtime(registry, provider, store)
    try:
        request = RunRequest(run_id="r1", thread_id="t1", agent_id="echo", text="hello")
        runtime.submit(request)
        assert wait(runtime, "r1")["status"] == "completed"
        assert runtime.submit(request)["id"] == "r1"
        assert len(provider.model_instance.calls) == 1
        assert runtime.history("t1")[0]["content"] == "hello"
        assert store.events("r1")[-1]["type"] == "done"
        with pytest.raises(ValueError):
            runtime.submit(request.model_copy(update={"text": "changed"}))
        registry.register(AgentDefinition("other", "Other", "", (), lambda ctx: SystemMessage(content="Other.")))
        with pytest.raises(ValueError):
            runtime.submit(request.model_copy(update={"run_id": "r2", "agent_id": "other"}))
    finally:
        runtime.close()


def test_tools_budget_and_denied_unknown_tool(tmp_path):
    class Empty(BaseModel):
        pass
    count = []
    tool = Tool("ping", "Ping", Empty, lambda ctx, args: (count.append(1) or ToolResult(status="success")))
    registry = Registry()
    registry.register(AgentDefinition("test", "Test", "", (Toolset("t", "Test", (tool,), always_on=True),),
                                      lambda ctx: SystemMessage(content="Test")))
    provider = FakeProvider([
        AIMessage(content="", tool_calls=[{"name": "forbidden", "args": {}, "id": "c0"}]),
        AIMessage(content="", tool_calls=[{"name": "ping", "args": {}, "id": "c1"}]),
    ])
    runtime = Runtime(registry, provider, Store(tmp_path / "test.sqlite"), max_tools=1)
    try:
        runtime.submit(RunRequest(run_id="r", thread_id="t", agent_id="test", text="hello"))
        assert wait(runtime, "r")["status"] == "failed"
        assert count == []
        events = runtime.store.events("r")
        assert any(e["type"] == "tool_result" and e["status"] == "error" for e in events)
        assert [e["sequence"] for e in events] == list(range(1, len(events) + 1))
    finally:
        runtime.close()


def test_memory_failure_is_visible(tmp_path, provider):
    def fail(job):
        raise ValueError("bad extraction")
    registry = Registry()
    registry.register(AgentDefinition("test", "Test", "", (), lambda ctx: SystemMessage(content="Test"), fail))
    runtime = Runtime(registry, provider, Store(tmp_path / "test.sqlite"))
    try:
        runtime.submit(RunRequest(run_id="r", thread_id="t", agent_id="test", text="hello"))
        assert wait(runtime, "r")["status"] == "completed"
        for _ in range(100):
            if runtime.store.memory_status("r") == "failed":
                break
            time.sleep(.02)
        assert runtime.store.memory_status("r") == "failed"
        assert any(e["type"] == "memory_failed" for e in runtime.store.events("r"))
    finally:
        runtime.close()


def test_restart_marks_interrupted(tmp_path):
    path = tmp_path / "test.sqlite"
    store = Store(path)
    store.new_run("r", "a", "t", "hello")
    assert Store(path).run("r")["status"] == "interrupted"
