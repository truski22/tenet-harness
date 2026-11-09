"""Event emission for one run, with token coalescing."""
from time import monotonic

from ..storage import RunStore


class EventEmitter:
    """Persists run events; consecutive tokens are merged to keep the event log small."""

    def __init__(self, store: RunStore, run_id: str, thread_id: str):
        self.store, self.run_id, self.thread_id = store, run_id, thread_id
        self._pending = ""
        self._last_token = monotonic()

    def __call__(self, kind: str, **kwargs) -> None:
        if kind == "token":
            self._pending += kwargs["text"]
            if len(self._pending) < 64 and monotonic() - self._last_token < 0.05:
                return
        if self._pending:
            self.store.event(self.run_id, self.thread_id, "token", text=self._pending)
            self._pending = ""
            self._last_token = monotonic()
        if kind == "token":
            return
        self.store.event(self.run_id, self.thread_id, kind, **kwargs)
