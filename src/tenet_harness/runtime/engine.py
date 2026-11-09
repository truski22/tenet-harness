"""Runtime: accepts runs, tracks the active ones and exposes control operations."""
import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Event, Lock, Thread

from langgraph.checkpoint.sqlite import SqliteSaver

from ..agents import Registry
from ..errors import RunRejected
from ..messages import Messages
from ..providers import Provider
from ..runs import RunRequest
from ..storage import Store
from .executor import RunExecutor
from .gate import ModelGate
from .history import thread_messages, visible_history
from .memory import MemoryWorker

log = logging.getLogger(__name__)


class BusyError(Exception):
    code = "BUSY"


@dataclass
class RunHandle:
    """A run that has been accepted and has not finished yet."""
    run_id: str
    thread_id: str
    cancel: Event = field(default_factory=Event)
    thread: Thread | None = None


class Runtime:
    """Facade over the run machinery.

    ``max_concurrent_runs`` bounds simultaneous runs (one per conversation at a time);
    ``model_concurrency`` bounds simultaneous model calls across runs and background jobs.
    The defaults (1 and 1) keep the original single-user behaviour.
    """

    def __init__(self, registry: Registry, provider: Provider, store: Store,
                 max_tools: int = 20, run_timeout: float = 300, context_window: int = 8192,
                 messages: Messages = Messages(), recursion_limit: int = 45,
                 max_concurrent_runs: int = 1, model_concurrency: int = 1):
        if max_concurrent_runs < 1:
            raise ValueError("max_concurrent_runs must be at least 1.")
        self.registry, self.provider, self.store = registry, provider, store
        self.messages, self.recursion_limit = messages, recursion_limit
        self.max_concurrent_runs = max_concurrent_runs
        self.gate = ModelGate(model_concurrency)
        self.stopping = Event()
        self._lock = Lock()
        self._runs: dict[str, RunHandle] = {}
        self._exclusive = False
        self.connection = sqlite3.connect(store.path, check_same_thread=False, timeout=30)
        self.checkpointer = SqliteSaver(self.connection)
        self.checkpointer.setup()  # create tables now: lazy setup races when runs start together
        self._executor = RunExecutor(registry, provider, store, self.checkpointer, self.gate, messages,
                                     self.config, max_tools, run_timeout, context_window)
        self._memory = MemoryWorker(store, registry, self.gate, messages, lambda: bool(self._runs))
        self._memory.start()

    # --- conversations -------------------------------------------------------------------------

    def config(self, thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id}, "recursion_limit": self.recursion_limit}

    def history(self, thread_id: str) -> list[dict]:
        return visible_history(thread_messages(self.checkpointer, self.config(thread_id)))

    def adopt_thread(self, thread_id: str, agent_id: str, title: str) -> bool:
        """Register a checkpointed thread that predates conversation metadata. Returns True if adopted."""
        if self.store.conversation(thread_id) or not self.checkpointer.get(self.config(thread_id)):
            return False
        self.store.touch(thread_id, agent_id, title)
        return True

    def delete_conversation(self, thread_id: str) -> None:
        with self.exclusive():
            self.checkpointer.delete_thread(thread_id)
            with self.store.connect() as db:
                db.execute("DELETE FROM memory_jobs WHERE thread_id=?", (thread_id,))
                db.execute("DELETE FROM run_events WHERE run_id IN "
                           "(SELECT id FROM runs WHERE thread_id=?)", (thread_id,))
                db.execute("DELETE FROM runs WHERE thread_id=?", (thread_id,))
                db.execute("DELETE FROM conversations WHERE thread_id=?", (thread_id,))

    # --- runs ----------------------------------------------------------------------------------

    @property
    def active_run_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._runs)

    def submit(self, request: RunRequest) -> dict:
        if not request.text.strip():
            raise RunRejected("EMPTY_MESSAGE", self.messages.empty_message)
        existing = self.store.run(request.run_id)
        if existing:
            if any(existing[key] != getattr(request, key) for key in ("agent_id", "thread_id")) \
                    or existing["user_text"] != request.text:
                raise RunRejected("IDEMPOTENCY_CONFLICT", self.messages.idempotency_conflict)
            return existing
        self.registry.get(request.agent_id)
        conversation = self.store.conversation(request.thread_id)
        if conversation and conversation["agent_id"] != request.agent_id:
            raise RunRejected("OTHER_AGENT", self.messages.other_agent)
        with self._lock:
            if self._exclusive or self.stopping.is_set() or len(self._runs) >= self.max_concurrent_runs \
                    or any(h.thread_id == request.thread_id for h in self._runs.values()):
                raise BusyError(self.messages.busy)
            handle = RunHandle(request.run_id, request.thread_id)
            try:
                self.store.touch(request.thread_id, request.agent_id, request.text)
                self.store.new_run(request.run_id, request.agent_id, request.thread_id, request.text)
                self._runs[request.run_id] = handle
                handle.thread = Thread(target=self._run, args=(request, handle), name=f"run-{request.run_id}",
                                       daemon=True)
                handle.thread.start()
            except Exception:
                self._runs.pop(request.run_id, None)
                raise
        return self.store.run(request.run_id)

    def cancel(self, run_id: str) -> None:
        with self._lock:
            handle = self._runs.get(run_id)
        if handle:
            handle.cancel.set()

    def wait(self, run_id: str, timeout: float | None = None) -> bool:
        """Block until the run's worker thread ends. Returns True if the run is no longer active."""
        with self._lock:
            handle = self._runs.get(run_id)
        if handle is None or handle.thread is None:
            return True
        handle.thread.join(timeout)
        return not handle.thread.is_alive()

    def _run(self, request: RunRequest, handle: RunHandle) -> None:
        try:
            self._executor.run(request, handle.cancel)
        finally:
            with self._lock:
                self._runs.pop(request.run_id, None)

    @contextmanager
    def exclusive(self):
        """Exclusive use of the runtime: no run active, none accepted and no model call in flight.

        Raises BusyError if a run is active. Use it to switch models or delete state safely.
        """
        with self._lock:
            if self._runs or self._exclusive:
                raise BusyError(self.messages.wait_for_run)
            self._exclusive = True
        try:
            with self.gate.drain():
                yield
        finally:
            with self._lock:
                self._exclusive = False

    def close(self) -> None:
        self.stopping.set()
        with self._lock:
            handles = list(self._runs.values())
        for handle in handles:
            handle.cancel.set()
        for handle in handles:
            if handle.thread:
                handle.thread.join()
        self._memory.stop()
        self.connection.close()
