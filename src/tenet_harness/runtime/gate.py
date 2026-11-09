"""Concurrency limit for model calls, shared by runs and background jobs."""
from contextlib import contextmanager
from threading import Condition
from typing import Callable


class ModelGate:
    """Counting gate: at most ``limit`` model calls at once.

    ``drain`` blocks new calls and waits for running ones, giving a caller exclusive use of the model
    (for example to switch models safely). Waiting is cooperative: ``check`` may raise to abort.
    """

    def __init__(self, limit: int = 1):
        if limit < 1:
            raise ValueError("Model concurrency must be at least 1.")
        self.limit = limit
        self._cond = Condition()
        self._in_use = 0
        self._draining = False

    @contextmanager
    def slot(self, check: Callable[[], None] = lambda: None):
        with self._cond:
            while self._draining or self._in_use >= self.limit:
                self._cond.wait(0.1)
                check()
            self._in_use += 1
        try:
            yield
        finally:
            with self._cond:
                self._in_use -= 1
                self._cond.notify_all()

    @contextmanager
    def drain(self):
        with self._cond:
            while self._draining:
                self._cond.wait(0.1)
            self._draining = True
            while self._in_use:
                self._cond.wait(0.1)
        try:
            yield
        finally:
            with self._cond:
                self._draining = False
                self._cond.notify_all()
