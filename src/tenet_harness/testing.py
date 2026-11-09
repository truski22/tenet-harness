"""Deterministic fakes for testing agents without a real model or database. Part of the public API."""
from threading import Lock
from typing import Any

from langchain_core.messages import AIMessage

from .storage.sqlite import timestamp


class FakeModel:
    def __init__(self, responses=None):
        self.responses = responses or [AIMessage(content="Local reply.")]
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


class InMemoryStore:
    """RunStore kept in process memory: no files, same behaviour as the SQLite Store.

    Pair it with ``langgraph.checkpoint.memory.MemorySaver`` as the Runtime checkpointer.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._conversations: dict[str, dict] = {}
        self._runs: dict[str, dict] = {}
        self._events: dict[str, list[dict]] = {}
        self._memory: dict[str, dict] = {}
        self._approvals: dict[str, dict] = {}
        self._audit: list[dict] = []

    def conversations(self) -> list[dict]:
        with self._lock:
            rows = sorted(self._conversations.values(), key=lambda c: c["updated_at"], reverse=True)
            return [{"id": c["thread_id"], "title": c["title"], "updated_at": c["updated_at"],
                     "agent_id": c["agent_id"]} for c in rows]

    def conversation(self, thread_id: str) -> dict | None:
        with self._lock:
            row = self._conversations.get(thread_id)
            return dict(row) if row else None

    def touch(self, thread_id: str, agent_id: str, text: str) -> None:
        with self._lock:
            now = timestamp()
            row = self._conversations.get(thread_id)
            if row:
                row["updated_at"] = now
            else:
                self._conversations[thread_id] = {
                    "thread_id": thread_id, "title": " ".join(text.split())[:60] or "Conversation",
                    "created_at": now, "updated_at": now, "agent_id": agent_id}

    def delete_thread(self, thread_id: str) -> None:
        with self._lock:
            for run_id in [r["id"] for r in self._runs.values() if r["thread_id"] == thread_id]:
                self._runs.pop(run_id)
                self._events.pop(run_id, None)
            self._memory = {k: v for k, v in self._memory.items() if v["thread_id"] != thread_id}
            self._conversations.pop(thread_id, None)

    def run(self, run_id: str) -> dict | None:
        with self._lock:
            row = self._runs.get(run_id)
            return dict(row) if row else None

    def new_run(self, run_id: str, agent_id: str, thread_id: str, text: str) -> None:
        with self._lock:
            if run_id in self._runs:
                raise ValueError("Duplicate run id.")
            self._runs[run_id] = {"id": run_id, "thread_id": thread_id, "agent_id": agent_id, "user_text": text,
                                  "status": "accepted", "started_at": timestamp(), "finished_at": None,
                                  "reply": None, "error": None, "input_tokens": 0, "output_tokens": 0,
                                  "model_calls": 0}

    def set_status(self, run_id: str, status: str) -> None:
        with self._lock:
            self._runs[run_id]["status"] = status

    def record_usage(self, run_id: str, input_tokens: int, output_tokens: int, calls: int) -> None:
        with self._lock:
            self._runs[run_id].update(input_tokens=input_tokens, output_tokens=output_tokens, model_calls=calls)

    def usage_since(self, since: str) -> dict:
        with self._lock:
            runs = [r for r in self._runs.values() if r["started_at"] >= since]
            return {"input_tokens": sum(r.get("input_tokens", 0) for r in runs),
                    "output_tokens": sum(r.get("output_tokens", 0) for r in runs),
                    "model_calls": sum(r.get("model_calls", 0) for r in runs), "runs": len(runs)}

    def create_approval(self, approval_id: str, run_id: str, thread_id: str, tool: str, effect: str,
                        args: dict, rule_id: str | None, reason: str) -> dict:
        with self._lock:
            record = {"id": approval_id, "run_id": run_id, "thread_id": thread_id, "tool": tool,
                      "effect": effect, "args": dict(args), "rule_id": rule_id, "reason": reason,
                      "status": "pending", "requested_at": timestamp(), "decided_at": None,
                      "decided_by": None, "note": None}
            self._approvals[approval_id] = record
            return dict(record)

    def decide_approval(self, approval_id: str, status: str, actor: str, note: str = "") -> dict | None:
        with self._lock:
            record = self._approvals.get(approval_id)
            if record is None or record["status"] != "pending":
                return None
            record.update(status=status, decided_at=timestamp(), decided_by=actor, note=note)
            return dict(record)

    def approval(self, approval_id: str) -> dict | None:
        with self._lock:
            record = self._approvals.get(approval_id)
            return dict(record) if record else None

    def approvals(self, run_id: str | None = None, status: str | None = None) -> list[dict]:
        with self._lock:
            return [dict(a) for a in self._approvals.values()
                    if (run_id is None or a["run_id"] == run_id) and (status is None or a["status"] == status)]

    def audit(self, run_id: str, thread_id: str, kind: str, tool: str | None = None,
              data: dict | None = None) -> None:
        with self._lock:
            self._audit.append({"id": len(self._audit) + 1, "ts": timestamp(), "run_id": run_id,
                                "thread_id": thread_id, "kind": kind, "tool": tool, "data": dict(data or {})})

    def audit_entries(self, run_id: str | None = None, limit: int = 500) -> list[dict]:
        with self._lock:
            return [dict(e) for e in self._audit if run_id is None or e["run_id"] == run_id][:limit]

    def finish(self, run_id: str, status: str, reply: str | None = None, error: str | None = None) -> None:
        with self._lock:
            self._runs[run_id].update(status=status, reply=reply, error=error, finished_at=timestamp())

    def event(self, run_id: str, thread_id: str, kind: str, **data: Any) -> dict:
        with self._lock:
            events = self._events.setdefault(run_id, [])
            event = {"version": 1, "sequence": len(events) + 1, "run_id": run_id,
                     "thread_id": thread_id, "type": kind, **data}
            events.append(event)
            return event

    def events(self, run_id: str, after: int = 0) -> list[dict]:
        with self._lock:
            return [e for e in self._events.get(run_id, []) if e["sequence"] > after][:200]

    def queue_memory(self, run_id: str, agent_id: str, thread_id: str, text: str, previous: str) -> None:
        with self._lock:
            if sum(j["status"] in {"pending", "running"} for j in self._memory.values()) >= 32:
                raise ValueError("Memory queue is full.")
            self._memory[run_id] = {"run_id": run_id, "agent_id": agent_id, "thread_id": thread_id,
                                    "user_text": text, "previous_reply": previous, "status": "pending"}

    def next_memory(self) -> dict | None:
        with self._lock:
            for job in self._memory.values():
                if job["status"] == "pending":
                    job["status"] = "running"
                    return dict(job)
            return None

    def memory_status(self, run_id: str) -> str | None:
        with self._lock:
            job = self._memory.get(run_id)
            return job["status"] if job else None

    def finish_memory(self, run_id: str, status: str) -> None:
        with self._lock:
            self._memory[run_id]["status"] = status
