import sqlite3

import pytest

from tenet_harness import Store
from tenet_harness.storage.sqlite import SCHEMA_VERSION


def make_v1_database(path):
    """The schema written by 0.1 before governance existed, with one finished run."""
    db = sqlite3.connect(path)
    db.executescript("""
        CREATE TABLE harness_schema (version INTEGER NOT NULL);
        INSERT INTO harness_schema VALUES (1);
        CREATE TABLE conversations (thread_id TEXT PRIMARY KEY, title TEXT NOT NULL, created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL, agent_id TEXT NOT NULL DEFAULT 'app');
        CREATE TABLE runs (id TEXT PRIMARY KEY, thread_id TEXT NOT NULL, agent_id TEXT NOT NULL,
            user_text TEXT NOT NULL, status TEXT NOT NULL, started_at TEXT NOT NULL, finished_at TEXT,
            reply TEXT, error TEXT);
        CREATE TABLE run_events (run_id TEXT NOT NULL, sequence INTEGER NOT NULL, payload TEXT NOT NULL,
            PRIMARY KEY(run_id,sequence));
        CREATE TABLE memory_jobs (run_id TEXT PRIMARY KEY, agent_id TEXT NOT NULL, thread_id TEXT NOT NULL,
            user_text TEXT NOT NULL, previous_reply TEXT NOT NULL, status TEXT NOT NULL);
        CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO runs VALUES ('old', 't', 'app', 'hi', 'completed', '2026-01-01T00:00:00+00:00',
            '2026-01-01T00:00:01+00:00', 'hello', NULL);
    """)
    db.commit()
    db.close()


def test_version_1_database_is_migrated_in_place(tmp_path):
    path = tmp_path / "old.sqlite"
    make_v1_database(path)
    store = Store(path)
    old = store.run("old")
    assert old["reply"] == "hello" and (old["input_tokens"], old["output_tokens"], old["model_calls"]) == (0, 0, 0)
    store.record_usage("old", 5, 3, 1)
    assert store.usage_since("2026-01-01T00:00:00+00:00")["input_tokens"] == 5
    store.audit("old", "t", "tool_outcome", "x", {})
    assert len(store.audit_entries("old")) == 1
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT version FROM harness_schema").fetchone()[0] == SCHEMA_VERSION


def test_reopening_a_migrated_database_is_a_no_op(tmp_path):
    path = tmp_path / "t.sqlite"
    Store(path)
    Store(path)  # must not try to add columns twice


def test_pending_approvals_expire_when_the_store_reopens(tmp_path):
    path = tmp_path / "t.sqlite"
    store = Store(path)
    store.create_approval("a1", "r", "t", "close", "write", {}, None, "why")
    again = Store(path)
    assert again.approval("a1")["status"] == "expired" and again.approval("a1")["decided_by"] == "system"


def test_database_from_a_newer_version_is_refused(tmp_path):
    path = tmp_path / "future.sqlite"
    Store(path)
    with sqlite3.connect(path) as db:
        db.execute("UPDATE harness_schema SET version=?", (SCHEMA_VERSION + 1,))
    with pytest.raises(ValueError, match="newer"):
        Store(path)
