import time

import pytest
from langchain_core.messages import AIMessage, SystemMessage
from pydantic import BaseModel

from tenet_harness import (AgentDefinition, Budget, Effect, LoopPolicy, Policy, Registry, RunRejected,
                           RunRequest, Runtime, Store, Tool, ToolError, ToolResult, Toolset)
from tenet_harness.testing import FakeProvider


class CloseArgs(BaseModel):
    ticket: str
    token: str = ""


class NoArgs(BaseModel):
    pass


class Counter(BaseModel):
    n: int = 0


def call(name, args, call_id):
    return AIMessage(content="", tool_calls=[{"name": name, "id": call_id, "args": args}])


def wait_for(condition, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(0.02)
    return False


class World:
    """An agent with an external write tool, a local write tool and two read tools."""

    def __init__(self, tmp_path, responses, policy=None, external=True, **runtime_options):
        self.closed, self.local_writes, self.reads = [], [], 0

        def close(ctx, args):
            self.closed.append(args.ticket)
            return ToolResult(status="success", message=f"closed {args.ticket}")

        def note(ctx, args):
            self.local_writes.append(args.ticket)
            return ToolResult(status="success")

        def read(ctx, args):
            self.reads += 1
            return ToolResult(status="success", data=self.reads)

        def flaky(ctx, args):
            raise ToolError("backend unavailable")

        tools = (Tool("close_ticket", "Close", CloseArgs, close, Effect.WRITE),
                 Tool("read_status", "Read", NoArgs, read, Effect.READ),
                 Tool("flaky", "Flaky", Counter, flaky, Effect.READ))
        local = (Tool("save_note", "Save", CloseArgs, note, Effect.WRITE),)
        registry = Registry()
        registry.register(AgentDefinition("a", "A", "", (
            Toolset("tickets", "Tickets", tools, always_on=True, external=external),
            Toolset("notes", "Notes", local, always_on=True)), lambda ctx: SystemMessage(content="Test.")))
        self.provider = FakeProvider(responses)
        self.runtime = Runtime(registry, self.provider, Store(tmp_path / "t.sqlite"), policy=policy,
                               **runtime_options)

    def submit(self, run_id="r1"):
        self.runtime.submit(RunRequest(run_id=run_id, thread_id=f"t-{run_id}", agent_id="a", text="go"))

    def run(self, run_id="r1"):
        return self.runtime.store.run(run_id)

    def events(self, run_id="r1"):
        return [e["type"] for e in self.runtime.store.events(run_id)]

    def audit(self, run_id="r1"):
        return [(e["kind"], e["tool"]) for e in self.runtime.audit_log(run_id)]


@pytest.fixture
def world_factory(tmp_path):
    worlds = []

    def make(responses, **options):
        world = World(tmp_path, responses, **options)
        worlds.append(world)
        return world

    yield make
    for world in worlds:
        world.runtime.close()


CLOSE = call("close_ticket", {"ticket": "T-1", "token": "s3cret"}, "c1")


def test_external_write_waits_for_approval_then_runs(world_factory):
    world = world_factory([CLOSE, AIMessage(content="Closed it.")])
    world.submit()
    assert wait_for(lambda: world.runtime.pending_approvals())
    pending = world.runtime.pending_approvals()[0]
    assert (pending["tool"], pending["effect"], pending["rule_id"]) == ("close_ticket", "write", "baseline:external-write")
    assert pending["args"] == {"ticket": "T-1", "token": "[REDACTED]"}  # secrets never reach the approver
    assert world.run()["status"] == "awaiting_approval" and world.closed == []
    resolved = world.runtime.resolve_approval(pending["id"], True, actor="alice", note="ok")
    assert (resolved["status"], resolved["decided_by"]) == ("approved", "alice")
    assert world.runtime.wait("r1", 5)
    assert world.run()["status"] == "completed" and world.closed == ["T-1"]
    assert "approval_required" in world.events() and "approval_resolved" in world.events()
    assert [k for k, _ in world.audit()] == ["tool_decision", "approval_approved", "tool_outcome"]
    assert "s3cret" not in str(world.runtime.audit_log("r1"))


def test_rejection_blocks_the_action_and_tells_the_model(world_factory):
    world = world_factory([CLOSE, AIMessage(content="I could not close it.")])
    world.submit()
    assert wait_for(lambda: world.runtime.pending_approvals())
    world.runtime.resolve_approval(world.runtime.pending_approvals()[0]["id"], False, actor="bob")
    assert world.runtime.wait("r1", 5)
    assert world.closed == [] and world.run()["status"] == "completed"
    seen = str([m.content for call_messages in world.provider.model_instance.calls for m in call_messages])
    assert "not approved (rejected)" in seen
    assert world.runtime.store.approvals(run_id="r1")[0]["status"] == "rejected"


def test_unanswered_approval_expires(world_factory):
    world = world_factory([CLOSE, AIMessage(content="Timed out.")], approval_timeout=0.3)
    world.submit()
    assert world.runtime.wait("r1", 5)
    approval = world.runtime.store.approvals(run_id="r1")[0]
    assert approval["status"] == "expired" and world.closed == []
    assert world.run()["status"] == "completed"


def test_cancel_while_waiting_for_approval(world_factory):
    world = world_factory([CLOSE, AIMessage(content="x")])
    world.submit()
    assert wait_for(lambda: world.runtime.pending_approvals())
    world.runtime.cancel("r1")
    assert world.runtime.wait("r1", 5)
    assert world.run()["status"] == "cancelled" and world.closed == []
    assert world.runtime.store.approvals(run_id="r1")[0]["status"] == "cancelled"


def test_waiting_for_a_human_does_not_consume_the_run_deadline(world_factory):
    world = world_factory([CLOSE, AIMessage(content="Done.")], run_timeout=0.6)
    world.submit()
    assert wait_for(lambda: world.runtime.pending_approvals())
    time.sleep(1.0)  # longer than the run timeout
    world.runtime.resolve_approval(world.runtime.pending_approvals()[0]["id"], True)
    assert world.runtime.wait("r1", 5)
    assert world.run()["status"] == "completed" and world.closed == ["T-1"]


def test_resolving_twice_or_unknown_is_rejected(world_factory):
    world = world_factory([CLOSE, AIMessage(content="Done.")])
    world.submit()
    assert wait_for(lambda: world.runtime.pending_approvals())
    approval_id = world.runtime.pending_approvals()[0]["id"]
    world.runtime.resolve_approval(approval_id, True)
    with pytest.raises(RunRejected) as twice:
        world.runtime.resolve_approval(approval_id, False)
    assert twice.value.code == "APPROVAL_NOT_PENDING"
    with pytest.raises(RunRejected) as unknown:
        world.runtime.resolve_approval("nope", True)
    assert unknown.value.code == "APPROVAL_NOT_FOUND"
    assert world.runtime.wait("r1", 5)


def test_deny_rule_blocks_without_asking(world_factory):
    policy = Policy.from_dict({"rules": [{"id": "no-t1", "tools": ["close_*"], "decision": "deny",
                                          "reason": "Closing is disabled.",
                                          "when": [{"arg": "ticket", "op": "eq", "value": "T-1"}]}]})
    world = world_factory([CLOSE, AIMessage(content="Blocked.")], policy=policy)
    world.submit()
    assert world.runtime.wait("r1", 5)
    assert world.closed == [] and world.runtime.store.approvals() == []
    assert world.audit()[0] == ("tool_decision", "close_ticket")
    assert "Blocked by policy: Closing is disabled." in str(world.provider.model_instance.calls)


def test_explicit_allow_rule_lifts_the_baseline(world_factory):
    policy = Policy.from_dict({"rules": [{"id": "ok", "tools": ["close_ticket"], "decision": "allow"}]})
    world = world_factory([CLOSE, AIMessage(content="Done.")], policy=policy)
    world.submit()
    assert world.runtime.wait("r1", 5)
    assert world.closed == ["T-1"] and world.runtime.store.approvals() == []


def test_local_writes_and_reads_need_no_approval_by_default(world_factory):
    world = world_factory([call("save_note", {"ticket": "n"}, "c1"), call("read_status", {}, "c2"),
                           AIMessage(content="Done.")])
    world.submit()
    assert world.runtime.wait("r1", 5)
    assert world.local_writes == ["n"] and world.reads == 1 and world.runtime.store.approvals() == []


def test_identical_calls_are_blocked_then_the_run_is_stopped(world_factory):
    world = world_factory([call("read_status", {}, f"c{i}") for i in range(8)])
    world.submit()
    assert world.runtime.wait("r1", 5)
    assert world.reads == 3  # the fourth identical call never executed
    assert world.run()["status"] == "failed"
    events = world.events()
    assert events.count("loop_detected") >= 3 and "error" in events
    assert ("loop_detected", None) in world.audit()


def test_blocked_repeat_lets_a_sensible_agent_recover(world_factory):
    responses = [call("read_status", {}, f"c{i}") for i in range(4)] + [AIMessage(content="Enough.")]
    world = world_factory(responses)
    world.submit()
    assert world.runtime.wait("r1", 5)
    assert world.run()["status"] == "completed" and world.reads == 3


def test_consecutive_failures_stop_the_run(world_factory):
    world = world_factory([call("flaky", {"n": i}, f"c{i}") for i in range(8)])
    world.submit()
    assert world.runtime.wait("r1", 5)
    assert world.run()["status"] == "failed" and "loop_detected" in world.events()


def test_loop_detection_can_be_disabled_or_tuned(world_factory):
    off = world_factory([call("read_status", {}, f"c{i}") for i in range(5)] + [AIMessage(content="ok")],
                        loop_policy=None)
    off.submit()
    assert off.runtime.wait("r1", 5) and off.run()["status"] == "completed" and off.reads == 5
    tight = world_factory([call("read_status", {}, f"c{i}") for i in range(4)] + [AIMessage(content="ok")],
                          loop_policy=LoopPolicy(max_identical_calls=1, max_blocked_calls=5))
    tight.submit("r2")
    assert tight.runtime.wait("r2", 5) and tight.reads == 1


def usage(message, tokens_in, tokens_out):
    message.usage_metadata = {"input_tokens": tokens_in, "output_tokens": tokens_out,
                              "total_tokens": tokens_in + tokens_out}
    return message


def test_run_token_budget_stops_the_run_and_usage_is_recorded(world_factory):
    responses = [usage(call("read_status", {}, "c1"), 60, 20), usage(call("save_note", {"ticket": "x"}, "c2"), 60, 20),
                 usage(AIMessage(content="never reached"), 60, 20)]
    world = world_factory(responses, budget=Budget(max_tokens_per_run=100))
    world.submit()
    assert world.runtime.wait("r1", 5)
    run = world.run()
    assert run["status"] == "failed" and "budget_exceeded" in world.events()
    assert (run["input_tokens"], run["output_tokens"], run["model_calls"]) == (120, 40, 2)
    assert world.local_writes == []  # no further work after the limit was crossed


def test_final_answer_is_kept_even_if_it_crosses_the_limit(world_factory):
    world = world_factory([usage(AIMessage(content="Long answer."), 500, 500)], budget=Budget(max_tokens_per_run=100))
    world.submit()
    assert world.runtime.wait("r1", 5)
    assert world.run()["status"] == "completed" and world.run()["reply"] == "Long answer."
    done = [e for e in world.runtime.store.events("r1") if e["type"] == "done"][0]
    assert done["data"]["input_tokens"] == 500 and done["data"]["estimated_calls"] == 0


def test_missing_provider_usage_is_estimated(world_factory):
    world = world_factory([AIMessage(content="Hello there")])
    world.submit()
    assert world.runtime.wait("r1", 5)
    run = world.run()
    assert run["model_calls"] == 1 and run["input_tokens"] > 0 and run["output_tokens"] > 0
    done = [e for e in world.runtime.store.events("r1") if e["type"] == "done"][0]
    assert done["data"]["estimated_calls"] == 1


def test_daily_token_budget_rejects_new_runs(world_factory):
    world = world_factory([usage(AIMessage(content="One."), 80, 40)], budget=Budget(max_tokens_per_day=100))
    world.submit("r1")
    assert world.runtime.wait("r1", 5)
    with pytest.raises(RunRejected) as rejected:
        world.submit("r2")
    assert rejected.value.code == "DAILY_BUDGET_EXCEEDED"
    assert world.runtime.store.run("r2") is None
