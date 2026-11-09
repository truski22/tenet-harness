"""SQLite run store."""
import json
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


SCHEMA_VERSION = 2


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, path: Path, legacy_agent_id: str = "legacy"):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", legacy_agent_id):
            raise ValueError("Invalid legacy agent id.")
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("CREATE TABLE IF NOT EXISTS harness_schema (version INTEGER NOT NULL)")
            row = db.execute("SELECT version FROM harness_schema").fetchone()
            if row and row[0] > SCHEMA_VERSION:
                raise ValueError("Database schema is newer than this application.")
            db.execute(
                "CREATE TABLE IF NOT EXISTS conversations (thread_id TEXT PRIMARY KEY, "
                "title TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)"
            )
            columns = {r[1] for r in db.execute("PRAGMA table_info(conversations)")}
            if "agent_id" not in columns:
                db.execute(
                    f"ALTER TABLE conversations ADD COLUMN agent_id TEXT NOT NULL DEFAULT '{legacy_agent_id}'"
                )
            db.execute(
                "CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, thread_id TEXT NOT NULL, "
                "agent_id TEXT NOT NULL, user_text TEXT NOT NULL, status TEXT NOT NULL, "
                "started_at TEXT NOT NULL, finished_at TEXT, reply TEXT, error TEXT)"
            )
            run_columns = {r[1] for r in db.execute("PRAGMA table_info(runs)")}
            for column in ("input_tokens", "output_tokens", "model_calls"):
                if column not in run_columns:
                    db.execute(f"ALTER TABLE runs ADD COLUMN {column} INTEGER NOT NULL DEFAULT 0")
            db.execute(
                "CREATE TABLE IF NOT EXISTS approvals (id TEXT PRIMARY KEY, run_id TEXT NOT NULL, "
                "thread_id TEXT NOT NULL, tool TEXT NOT NULL, effect TEXT NOT NULL, args TEXT NOT NULL, "
                "rule_id TEXT, reason TEXT NOT NULL, status TEXT NOT NULL, requested_at TEXT NOT NULL, "
                "decided_at TEXT, decided_by TEXT, note TEXT)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS audit_log (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, "
                "run_id TEXT NOT NULL, thread_id TEXT NOT NULL, kind TEXT NOT NULL, tool TEXT, data TEXT NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS run_events (run_id TEXT NOT NULL, sequence INTEGER NOT NULL, "
                "payload TEXT NOT NULL, PRIMARY KEY(run_id,sequence))"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS memory_jobs (run_id TEXT PRIMARY KEY, agent_id TEXT NOT NULL, "
                "thread_id TEXT NOT NULL, user_text TEXT NOT NULL, previous_reply TEXT NOT NULL, "
                "status TEXT NOT NULL)"
            )
            db.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            if row is None:
                db.execute("INSERT INTO harness_schema VALUES (?)", (SCHEMA_VERSION,))
            elif row[0] < SCHEMA_VERSION:
                db.execute("UPDATE harness_schema SET version=?", (SCHEMA_VERSION,))
            # A pending approval belongs to a run that no longer exists in this process.
            db.execute("UPDATE approvals SET status='expired',decided_at=?,decided_by='system' "
                       "WHERE status='pending'", (timestamp(),))
            db.execute(
                "UPDATE runs SET status='interrupted',finished_at=? "
                "WHERE status IN ('accepted','running')", (timestamp(),)
            )
            db.execute("UPDATE memory_jobs SET status='pending' WHERE status='running'")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def conversations(self):
        with self.connect() as db:
            return [dict(row) for row in db.execute(
                "SELECT thread_id AS id,title,updated_at,agent_id FROM conversations ORDER BY updated_at DESC"
            )]

    def conversation(self, thread_id: str):
        with self.connect() as db:
            row = db.execute("SELECT * FROM conversations WHERE thread_id=?", (thread_id,)).fetchone()
            return dict(row) if row else None

    def touch(self, thread_id: str, agent_id: str, text: str):
        with self.connect() as db:
            db.execute(
                "INSERT INTO conversations (thread_id,title,created_at,updated_at,agent_id) VALUES (?,?,?,?,?) "
                "ON CONFLICT(thread_id) DO UPDATE SET updated_at=excluded.updated_at",
                (thread_id, " ".join(text.split())[:60] or "Conversation", timestamp(), timestamp(), agent_id),
            )

    def run(self, run_id: str):
        with self.connect() as db:
            row = db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
            return dict(row) if row else None

    def new_run(self, run_id: str, agent_id: str, thread_id: str, text: str):
        with self.connect() as db:
            db.execute(
                "INSERT INTO runs (id,thread_id,agent_id,user_text,status,started_at) VALUES (?,?,?,?,?,?)",
                (run_id, thread_id, agent_id, text, "accepted", timestamp()),
            )

    def set_status(self, run_id: str, status: str):
        with self.connect() as db:
            db.execute("UPDATE runs SET status=? WHERE id=?", (status, run_id))

    def record_usage(self, run_id: str, input_tokens: int, output_tokens: int, calls: int):
        with self.connect() as db:
            db.execute("UPDATE runs SET input_tokens=?,output_tokens=?,model_calls=? WHERE id=?",
                       (input_tokens, output_tokens, calls, run_id))

    def usage_since(self, since: str) -> dict:
        """Token totals over runs started at or after ``since`` (ISO-8601 UTC)."""
        with self.connect() as db:
            row = db.execute("SELECT COALESCE(SUM(input_tokens),0), COALESCE(SUM(output_tokens),0), "
                             "COALESCE(SUM(model_calls),0), COUNT(*) FROM runs WHERE started_at>=?",
                             (since,)).fetchone()
            return {"input_tokens": row[0], "output_tokens": row[1], "model_calls": row[2], "runs": row[3]}

    def create_approval(self, approval_id: str, run_id: str, thread_id: str, tool: str, effect: str,
                        args: dict, rule_id: str | None, reason: str) -> dict:
        with self.connect() as db:
            db.execute("INSERT INTO approvals (id,run_id,thread_id,tool,effect,args,rule_id,reason,status,"
                       "requested_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                       (approval_id, run_id, thread_id, tool, effect, json.dumps(args, ensure_ascii=False),
                        rule_id, reason, "pending", timestamp()))
        record = self.approval(approval_id)
        assert record is not None
        return record

    def decide_approval(self, approval_id: str, status: str, actor: str, note: str = "") -> dict | None:
        """Resolve a pending approval exactly once. Returns None if it was not pending."""
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            changed = db.execute("UPDATE approvals SET status=?,decided_at=?,decided_by=?,note=? "
                                 "WHERE id=? AND status='pending'",
                                 (status, timestamp(), actor, note, approval_id)).rowcount
        return self.approval(approval_id) if changed else None

    def approval(self, approval_id: str) -> dict | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM approvals WHERE id=?", (approval_id,)).fetchone()
        return self._approval(row) if row else None

    def approvals(self, run_id: str | None = None, status: str | None = None) -> list[dict]:
        query = "SELECT * FROM approvals WHERE 1=1"
        params: list[str] = []
        if run_id:
            query, params = query + " AND run_id=?", [*params, run_id]
        if status:
            query, params = query + " AND status=?", [*params, status]
        with self.connect() as db:
            return [self._approval(r) for r in db.execute(query + " ORDER BY requested_at", params)]

    @staticmethod
    def _approval(row) -> dict:
        record = dict(row)
        record["args"] = json.loads(record["args"])
        return record

    def audit(self, run_id: str, thread_id: str, kind: str, tool: str | None = None,
              data: dict | None = None):
        with self.connect() as db:
            db.execute("INSERT INTO audit_log (ts,run_id,thread_id,kind,tool,data) VALUES (?,?,?,?,?,?)",
                       (timestamp(), run_id, thread_id, kind, tool, json.dumps(data or {}, ensure_ascii=False)))

    def audit_entries(self, run_id: str | None = None, limit: int = 500) -> list[dict]:
        query = "SELECT * FROM audit_log"
        params: list[str] = []
        if run_id:
            query, params = query + " WHERE run_id=?", [run_id]
        with self.connect() as db:
            rows = db.execute(query + " ORDER BY id LIMIT ?", [*params, limit]).fetchall()
        return [{**dict(r), "data": json.loads(r["data"])} for r in rows]

    def delete_thread(self, thread_id: str):
        """Remove a conversation and everything recorded for it (not the LangGraph checkpoints)."""
        with self.connect() as db:
            db.execute("DELETE FROM memory_jobs WHERE thread_id=?", (thread_id,))
            db.execute("DELETE FROM run_events WHERE run_id IN "
                       "(SELECT id FROM runs WHERE thread_id=?)", (thread_id,))
            db.execute("DELETE FROM runs WHERE thread_id=?", (thread_id,))
            db.execute("DELETE FROM conversations WHERE thread_id=?", (thread_id,))

    def finish(self, run_id: str, status: str, reply: str | None = None, error: str | None = None):
        with self.connect() as db:
            db.execute("UPDATE runs SET status=?,reply=?,error=?,finished_at=? WHERE id=?",
                       (status, reply, error, timestamp(), run_id))

    def event(self, run_id: str, thread_id: str, kind: str, **data):
        with self.connect() as db:
            # Immediate transaction serializes model/worker event writers.
            db.execute("BEGIN IMMEDIATE")
            sequence = db.execute(
                "SELECT COALESCE(MAX(sequence),0)+1 FROM run_events WHERE run_id=?", (run_id,)
            ).fetchone()[0]
            event = {"version": 1, "sequence": sequence, "run_id": run_id,
                     "thread_id": thread_id, "type": kind, **data}
            db.execute("INSERT INTO run_events VALUES (?,?,?)",
                       (run_id, sequence, json.dumps(event, ensure_ascii=False)))
            return event

    def events(self, run_id: str, after: int = 0):
        with self.connect() as db:
            return [json.loads(row[0]) for row in db.execute(
                "SELECT payload FROM run_events WHERE run_id=? AND sequence>? ORDER BY sequence LIMIT 200",
                (run_id, after),
            )]

    def queue_memory(self, run_id: str, agent_id: str, thread_id: str, text: str, previous: str):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            count = db.execute("SELECT COUNT(*) FROM memory_jobs WHERE status IN ('pending','running')").fetchone()[0]
            if count >= 32:
                raise ValueError("Memory queue is full.")
            db.execute("INSERT INTO memory_jobs VALUES (?,?,?,?,?,?)",
                       (run_id, agent_id, thread_id, text, previous, "pending"))

    def next_memory(self):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM memory_jobs WHERE status='pending' ORDER BY rowid LIMIT 1").fetchone()
            if row:
                db.execute("UPDATE memory_jobs SET status='running' WHERE run_id=?", (row["run_id"],))
            return dict(row) if row else None

    def memory_status(self, run_id: str):
        with self.connect() as db:
            row = db.execute("SELECT status FROM memory_jobs WHERE run_id=?", (run_id,)).fetchone()
            return row[0] if row else None

    def finish_memory(self, run_id: str, status: str):
        with self.connect() as db:
            db.execute("UPDATE memory_jobs SET status=? WHERE run_id=?", (status, run_id))

    def get_settings(self) -> dict:
        with self.connect() as db:
            return {row[0]: row[1] for row in db.execute("SELECT key,value FROM settings")}

    def save_settings(self, settings: dict[str, str]):
        with self.connect() as db:
            db.executemany("INSERT OR REPLACE INTO settings VALUES (?,?)", settings.items())
