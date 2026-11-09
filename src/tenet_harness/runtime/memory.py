"""Background worker that runs an agent's post-run memory extraction, one job at a time."""
import logging
from threading import Event, Thread
from typing import Callable

from ..agents import Registry
from ..errors import Cancelled
from ..messages import Messages
from ..storage import RunStore
from .gate import ModelGate

log = logging.getLogger(__name__)


class MemoryWorker:
    """Polls the store for queued memory jobs. Yields to live runs: it only starts a job when idle."""

    def __init__(self, store: RunStore, registry: Registry, gate: ModelGate, messages: Messages,
                 is_busy: Callable[[], bool]):
        self.store, self.registry, self.gate, self.messages, self.is_busy = store, registry, gate, messages, is_busy
        self.stopping = Event()
        self.thread = Thread(target=self._loop, name="memory-worker", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stopping.set()
        self.thread.join()

    def _check(self) -> None:
        if self.stopping.is_set():
            raise Cancelled("Memory worker stopped")

    def _loop(self) -> None:
        while not self.stopping.wait(0.2):
            if self.is_busy():
                continue
            try:
                with self.gate.slot(self._check):
                    job = self.store.next_memory()
                    if job is not None:
                        self._process(job)
            except Cancelled:
                return

    def _process(self, job: dict) -> None:
        try:
            agent = self.registry.get(job["agent_id"])
            if agent.memorize is None:
                raise ValueError("Agent memory hook unavailable")
            saved = agent.memorize(job)
            self.store.event(job["run_id"], job["thread_id"], "memory_saved", data=saved)
            self.store.finish_memory(job["run_id"], "completed")
        except Exception as exc:
            log.error("Memory job %s failed (%s)", job["run_id"], type(exc).__name__)
            self.store.event(job["run_id"], job["thread_id"], "memory_failed", message=self.messages.memory_failed)
            self.store.finish_memory(job["run_id"], "failed")
