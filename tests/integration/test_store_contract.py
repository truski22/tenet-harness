import pytest
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver

from tenet_harness import AgentDefinition, Registry, RunRequest, Runtime, Store
from tenet_harness.storage import RunStore
from tenet_harness.testing import FakeProvider, InMemoryStore


@pytest.fixture(params=["sqlite", "memory"])
def store(request, tmp_path):
    return Store(tmp_path / "t.sqlite") if request.param == "sqlite" else InMemoryStore()


def test_both_stores_satisfy_the_protocol(store):
    assert isinstance(store, RunStore)


def test_store_contract(store):
    store.touch("t", "a", "  hello   world ")
    assert store.conversation("t")["title"] == "hello world"
    assert store.conversations()[0]["id"] == "t"
    store.new_run("r", "a", "t", "hello")
    assert store.run("r")["status"] == "accepted"
    store.set_status("r", "running")
    assert store.run("r")["status"] == "running"
    first = store.event("r", "t", "start")
    second = store.event("r", "t", "token", text="x")
    assert (first["sequence"], second["sequence"]) == (1, 2)
    assert [e["sequence"] for e in store.events("r", after=1)] == [2]
    store.finish("r", "completed", reply="done")
    assert store.run("r")["reply"] == "done" and store.run("r")["finished_at"]
    store.queue_memory("r", "a", "t", "hello", "")
    assert store.memory_status("r") == "pending"
    job = store.next_memory()
    assert job["run_id"] == "r" and store.next_memory() is None
    store.finish_memory("r", "completed")
    assert store.memory_status("r") == "completed"
    store.delete_thread("t")
    assert store.conversation("t") is None and store.run("r") is None and store.events("r") == []


def test_memory_queue_is_bounded(store):
    for i in range(32):
        store.queue_memory(f"r{i}", "a", "t", "x", "")
    with pytest.raises(ValueError):
        store.queue_memory("overflow", "a", "t", "x", "")


def test_runtime_runs_on_the_in_memory_store():
    registry = Registry()
    registry.register(AgentDefinition("a", "A", "", (), lambda ctx: SystemMessage(content="Test.")))
    store = InMemoryStore()
    runtime = Runtime(registry, FakeProvider([AIMessage(content="Hi there")]), store, checkpointer=MemorySaver())
    try:
        runtime.submit(RunRequest(run_id="r", thread_id="t", agent_id="a", text="hello"))
        assert runtime.wait("r", 5)
        assert store.run("r")["reply"] == "Hi there"
        assert [m["content"] for m in runtime.history("t")] == ["hello", "Hi there"]
    finally:
        runtime.close()


def test_sqlite_checkpointer_is_required_for_other_stores():
    with pytest.raises(ValueError):
        Runtime(Registry(), FakeProvider(), InMemoryStore())


def test_approvals_are_resolved_exactly_once(store):
    record = store.create_approval("a1", "r", "t", "close", "write", {"x": 1}, "rule", "why")
    assert record["status"] == "pending" and record["args"] == {"x": 1}
    assert [a["id"] for a in store.approvals(status="pending")] == ["a1"]
    decided = store.decide_approval("a1", "approved", "alice", "ok")
    assert decided["status"] == "approved" and decided["decided_by"] == "alice" and decided["decided_at"]
    assert store.decide_approval("a1", "rejected", "bob") is None  # already resolved
    assert store.approval("a1")["status"] == "approved" and store.approval("missing") is None
    assert store.approvals(run_id="other") == [] and store.approvals(status="pending") == []


def test_audit_log_keeps_order_and_filters_by_run(store):
    store.audit("r1", "t", "tool_decision", "close", {"decision": "allow"})
    store.audit("r2", "t", "tool_outcome", "close", {"status": "success"})
    store.audit("r1", "t", "tool_outcome", "close", {"status": "error"})
    entries = store.audit_entries("r1")
    assert [(e["kind"], e["data"]) for e in entries] == [("tool_decision", {"decision": "allow"}),
                                                          ("tool_outcome", {"status": "error"})]
    assert len(store.audit_entries()) == 3 and len(store.audit_entries(limit=2)) == 2


def test_usage_is_recorded_per_run_and_summed_by_date(store):
    store.new_run("r1", "a", "t", "hello")
    store.new_run("r2", "a", "t", "hello")
    store.record_usage("r1", 100, 20, 2)
    store.record_usage("r2", 50, 5, 1)
    assert store.run("r1")["input_tokens"] == 100
    total = store.usage_since("1970-01-01T00:00:00+00:00")
    assert total == {"input_tokens": 150, "output_tokens": 25, "model_calls": 3, "runs": 2}
    assert store.usage_since("2999-01-01T00:00:00+00:00")["runs"] == 0
