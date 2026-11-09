"""An agent that wants to close a ticket in an external system, under policy and human approval.

    uv run python examples/approval_agent.py

A scripted model asks to close a ticket. The baseline policy requires approval for writes to external
systems, so the run pauses until a "person" (a thread here) approves it. Swap FakeProvider for
OllamaProvider to go live; resolve approvals from a CLI, a web hook or a chat bot in a real deployment.
"""
import tempfile
import threading
import time
from pathlib import Path

from langchain_core.messages import AIMessage, SystemMessage
from pydantic import BaseModel

from tenet_harness import (AgentDefinition, Effect, Policy, Registry, RunRequest, Runtime, Store, Tool, ToolResult,
                           Toolset)
from tenet_harness.testing import FakeProvider

CLOSED: list[str] = []


class CloseInput(BaseModel):
    ticket: str
    api_token: str = ""  # secrets in arguments are masked before an approver or the audit log sees them


def close_ticket(context, args: CloseInput) -> ToolResult:
    CLOSED.append(args.ticket)
    return ToolResult(status="success", message=f"Ticket {args.ticket} closed.")


agent = AgentDefinition(
    id="tickets", name="Tickets", description="Closes tickets in an external tracker.",
    toolsets=(Toolset("tracker", "Ticket tracker", (Tool("close_ticket", "Close a ticket.", CloseInput,
                                                         close_ticket, Effect.WRITE),),
                      always_on=True, external=True),),
    context=lambda context: SystemMessage(content="Close tickets when asked."),
)


def approver(runtime: Runtime) -> None:
    while not (pending := runtime.pending_approvals()):
        time.sleep(0.05)
    request = pending[0]
    print(f"Approval needed: {request['tool']} {request['args']} ({request['reason']})")
    runtime.resolve_approval(request["id"], approve=True, actor="alice")


if __name__ == "__main__":
    registry = Registry()
    registry.register(agent)
    model = FakeProvider([
        AIMessage(content="", tool_calls=[{"name": "close_ticket", "id": "c1",
                                           "args": {"ticket": "OPS-42", "api_token": "s3cret"}}]),
        AIMessage(content="Ticket OPS-42 is closed."),
    ])
    with tempfile.TemporaryDirectory() as tmp:
        store = Store(Path(tmp) / "harness.sqlite")
        runtime = Runtime(registry, model, store, policy=Policy())
        threading.Thread(target=approver, args=(runtime,), daemon=True).start()
        runtime.submit(RunRequest(run_id="r1", thread_id="t1", agent_id="tickets", text="Close OPS-42"))
        runtime.wait("r1", 30)
        print("Reply:", store.run("r1")["reply"], "| closed:", CLOSED)
        for entry in runtime.audit_log("r1"):
            print(" audit:", entry["kind"], entry["tool"], entry["data"])
        runtime.close()
