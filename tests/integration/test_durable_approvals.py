"""A run waiting for approval survives the runtime (and the process) that started it."""
import sqlite3
import time

import pytest
from langchain_core.messages import AIMessage, SystemMessage
from pydantic import BaseModel

from tenet_harness import (AgentDefinition, Effect, Registry, RunRejected, RunRequest, Runtime, Store, Tool,
                           ToolResult, Toolset)
from tenet_harness.testing import FakeProvider


class CloseArgs(BaseModel):
    ticket: str
    token: str = ""


class NoArgs(BaseModel):
    pass


def call(name, args, call_id):
    return {"name": name, "id": call_id, "args": args}


def message(*calls):
    return AIMessage(content="", tool_calls=list(calls))


def wait_for(condition, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(0.02)
    return False


class Site:
    """One database file opened by successive runtimes, like an application restarting."""

    def __init__(self, tmp_path):
        self.path = tmp_path / "t.sqlite"
        self.closed: list[tuple[str, str]] = []
        self.reads = 0
        self.runtimes: list[Runtime] = []

    def open(self, responses, **options) -> Runtime:
        def close(ctx, args):
            self.closed.append((args.ticket, args.token))
            return ToolResult(status="success", message=f"closed {args.ticket}")

        def read(ctx, args):
            self.reads += 1
            return ToolResult(status="success", data=self.reads)

        registry = Registry()
        registry.register(AgentDefinition("a", "A", "", (Toolset("tickets", "Tickets", (
            Tool("close_ticket", "Close", CloseArgs, close, Effect.WRITE),
            Tool("read_status", "Read", NoArgs, read, Effect.READ)), always_on=True, external=True),),
            lambda ctx: SystemMessage(content="Test.")))
        runtime = Runtime(registry, FakeProvider(responses), Store(self.path), **options)
        self.runtimes.append(runtime)
        return runtime

    def close_all(self):
        for runtime in self.runtimes:
            runtime.close()


@pytest.fixture
def site(tmp_path):
    site = Site(tmp_path)
    yield site
    site.close_all()


def submit(runtime, run_id="r1", thread="t1"):
    runtime.submit(RunRequest(run_id=run_id, thread_id=thread, agent_id="a", text="go"))


def suspend(site, responses):
    """Start a run that asks for approval, then stop the runtime while it waits."""
    first = site.open(responses)
    submit(first)
    assert wait_for(lambda: first.pending_approvals())
    first.close()
    assert first.wait("r1", 5)
    return first.store.approvals()[0]["id"]


CLOSE = call("close_ticket", {"ticket": "T-1", "token": "s3cret"}, "c1")


def test_shutdown_suspends_the_run_and_keeps_the_request_pending(site):
    approval_id = suspend(site, [message(CLOSE), AIMessage(content="Closed it.")])
    store = Store(site.path)
    assert store.run("r1")["status"] == "awaiting_approval"
    assert store.approval(approval_id)["status"] == "pending" and site.closed == []
    types = [e["type"] for e in store.events("r1")]
    assert "suspended" in types and "done" not in types


def test_approving_after_a_restart_runs_the_action_once_and_finishes_the_run(site):
    approval_id = suspend(site, [message(CLOSE), AIMessage(content="Closed it.")])
    second = site.open([AIMessage(content="Closed it.")])
    assert [a["id"] for a in second.pending_approvals()] == [approval_id]
    second.resolve_approval(approval_id, True, actor="alice")
    assert second.wait("r1", 5)
    run = second.store.run("r1")
    assert run["status"] == "completed" and run["reply"] == "Closed it."
    assert site.closed == [("T-1", "s3cret")]  # the real, unredacted arguments from the checkpoint
    types = [e["type"] for e in second.store.events("r1")]
    assert types.count("start") == 1 and "resumed" in types and types[-1] == "done"
    kinds = [e["kind"] for e in second.audit_log("r1")]
    assert kinds == ["tool_decision", "approval_approved", "tool_outcome"]
    assert "s3cret" not in str(second.audit_log("r1"))


def test_rejecting_after_a_restart_tells_the_model_and_runs_nothing(site):
    approval_id = suspend(site, [message(CLOSE), AIMessage(content="x")])
    second = site.open([AIMessage(content="I did not close it.")])
    second.resolve_approval(approval_id, False, actor="bob")
    assert second.wait("r1", 5)
    assert site.closed == [] and second.store.run("r1")["status"] == "completed"
    assert "not approved (rejected)" in str(second.provider.model_instance.calls)


def test_sibling_calls_of_the_same_turn_are_never_replayed(site):
    turn = message(call("read_status", {}, "c0"), CLOSE, call("read_status", {}, "c2"))
    approval_id = suspend(site, [turn, AIMessage(content="x")])
    assert site.reads == 1  # the first read ran live; its result was lost with the process
    second = site.open([AIMessage(content="Done.")])
    second.resolve_approval(approval_id, True)
    assert second.wait("r1", 5)
    assert site.reads == 1 and site.closed == [("T-1", "s3cret")]  # not replayed
    assert "outcome unknown" in str(second.provider.model_instance.calls)


def test_usage_accumulates_across_the_suspension(site):
    reply = message(CLOSE)
    reply.usage_metadata = {"input_tokens": 50, "output_tokens": 10, "total_tokens": 60}
    approval_id = suspend(site, [reply, AIMessage(content="x")])
    first = Store(site.path).run("r1")
    assert (first["input_tokens"], first["output_tokens"], first["model_calls"]) == (50, 10, 1)
    final = AIMessage(content="Done.")
    final.usage_metadata = {"input_tokens": 70, "output_tokens": 5, "total_tokens": 75}
    second = site.open([final])
    second.resolve_approval(approval_id, True)
    assert second.wait("r1", 5)
    run = second.store.run("r1")
    assert (run["input_tokens"], run["output_tokens"], run["model_calls"]) == (120, 15, 2)


def test_a_conversation_with_a_suspended_run_does_not_accept_new_messages(site):
    suspend(site, [message(CLOSE), AIMessage(content="x")])
    second = site.open([AIMessage(content="Hi.")])
    with pytest.raises(RunRejected) as rejected:
        submit(second, "r2", "t1")
    assert rejected.value.code == "APPROVAL_PENDING"
    submit(second, "r3", "t-other")  # other conversations are unaffected
    assert second.wait("r3", 5)


def test_old_requests_cannot_be_approved_after_a_restart(site):
    approval_id = suspend(site, [message(CLOSE), AIMessage(content="x")])
    second = site.open([AIMessage(content="I could not close it.")], approval_ttl=0)
    with pytest.raises(RunRejected) as stale:
        second.resolve_approval(approval_id, True)
    assert stale.value.code == "APPROVAL_STALE"
    assert second.wait("r1", 5)
    assert site.closed == [] and second.store.approval(approval_id)["status"] == "expired"
    assert second.store.run("r1")["status"] == "completed"


def test_a_decided_approval_is_resumed_when_the_process_died_before_continuing(site):
    approval_id = suspend(site, [message(CLOSE), AIMessage(content="x")])
    Store(site.path).decide_approval(approval_id, "approved", "alice")  # decided, then the process died
    second = site.open([AIMessage(content="Closed it.")])
    assert second.wait("r1", 5)
    assert site.closed == [("T-1", "s3cret")] and second.store.run("r1")["status"] == "completed"


def test_deleting_the_conversation_cancels_its_pending_requests(site):
    approval_id = suspend(site, [message(CLOSE), AIMessage(content="x")])
    second = site.open([AIMessage(content="x")])
    second.delete_conversation("t1")
    assert second.store.approval(approval_id)["status"] == "cancelled"
    with pytest.raises(RunRejected):
        second.resolve_approval(approval_id, True)
    assert site.closed == []


def test_a_crash_between_marking_and_requesting_leaves_nothing_stuck(tmp_path):
    path = tmp_path / "t.sqlite"
    store = Store(path)
    store.new_run("r", "a", "t", "hi")
    store.set_status("r", "awaiting_approval")  # killed before the approval row was written
    assert Store(path).run("r")["status"] == "interrupted"


def test_requests_written_before_durable_approvals_expire(tmp_path):
    path = tmp_path / "t.sqlite"
    store = Store(path)
    store.new_run("r", "a", "t", "hi")
    store.set_status("r", "awaiting_approval")
    store.create_approval("a1", "r", "t", "close", "write", {}, None, "why")  # no tool call id: cannot resume
    again = Store(path)
    assert again.approval("a1")["status"] == "expired" and again.run("r")["status"] == "interrupted"
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT version FROM harness_schema").fetchone()[0] == 3
