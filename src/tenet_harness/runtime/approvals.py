"""Human-in-the-loop checkpoints: a run pauses on a risky tool call until a person decides."""
from __future__ import annotations

import uuid
from threading import Event, Lock
from time import monotonic
from typing import Any

from ..errors import Cancelled, RunRejected
from ..messages import Messages
from ..policy import PolicyDecision
from ..runs import RunContext
from ..storage import RunStore
from ..tools import Tool

APPROVED, REJECTED, EXPIRED = "approved", "rejected", "expired"


class ApprovalBroker:
    """Creates approval requests, blocks the requesting run and wakes it when someone resolves them.

    Waiting is in-process: if the process stops, pending requests are marked expired on restart and the
    interrupted tool call is never replayed. While waiting, the run's deadline does not run.
    """

    def __init__(self, store: RunStore, messages: Messages, timeout: float = 600.0):
        self.store, self.messages, self.timeout = store, messages, timeout
        self._waiters: dict[str, Event] = {}
        self._lock = Lock()

    def request(self, context: RunContext, tool: Tool, args: dict[str, Any], decision: PolicyDecision) -> str:
        """Block until the request is approved, rejected, expired or the run is cancelled."""
        approval_id = uuid.uuid4().hex
        # Register the waiter and the visible state before the request exists, so that whoever sees
        # it pending can resolve it immediately without the run missing the wake-up.
        waiter = Event()
        with self._lock:
            self._waiters[approval_id] = waiter
        self.store.set_status(context.run_id, "awaiting_approval")
        try:
            self.store.create_approval(approval_id, context.run_id, context.thread_id, tool.name,
                                       tool.effect.value, args, decision.rule_id, decision.reason)
        except Exception:
            with self._lock:
                self._waiters.pop(approval_id, None)
            self.store.set_status(context.run_id, "running")
            raise
        context.emit("approval_required", name=tool.name, message=decision.reason,
                     data={"approval_id": approval_id, "effect": tool.effect.value, "args": args,
                           "rule_id": decision.rule_id})
        started = monotonic()
        try:
            while not waiter.wait(0.1):
                current = self.store.approval(approval_id)
                if current and current["status"] != "pending":
                    break  # resolved through the store (for example by another process)
                if context.cancel.is_set():
                    self.store.decide_approval(approval_id, "cancelled", "system")
                    raise Cancelled("Run cancelled while awaiting approval")
                if monotonic() - started >= self.timeout:
                    self.store.decide_approval(approval_id, EXPIRED, "system", "Approval timed out.")
                    break
        finally:
            with self._lock:
                self._waiters.pop(approval_id, None)
            context.deadline += monotonic() - started
            self.store.set_status(context.run_id, "running")
        record = self.store.approval(approval_id)
        status = record["status"] if record else EXPIRED
        context.emit("approval_resolved", name=tool.name, status=status,
                     data={"approval_id": approval_id, "decided_by": record and record["decided_by"]})
        return status

    def resolve(self, approval_id: str, approve: bool, actor: str = "human", note: str = "") -> dict:
        """Approve or reject a pending request. Exactly one caller can resolve it."""
        record = self.store.decide_approval(approval_id, APPROVED if approve else REJECTED, actor, note)
        if record is None:
            if self.store.approval(approval_id) is None:
                raise RunRejected("APPROVAL_NOT_FOUND", self.messages.approval_not_found)
            raise RunRejected("APPROVAL_NOT_PENDING", self.messages.approval_not_pending)
        with self._lock:
            waiter = self._waiters.get(approval_id)
        if waiter:
            waiter.set()
        return record
