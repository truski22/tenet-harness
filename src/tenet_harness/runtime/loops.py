"""Detection of agents that repeat themselves or keep failing."""
import json
from dataclasses import dataclass

from ..errors import LoopDetected


@dataclass(frozen=True)
class LoopPolicy:
    """Thresholds, per run.

    An identical call (same tool, same arguments) may execute ``max_identical_calls`` times; further
    attempts are refused and the model is told to stop. After ``max_blocked_calls`` refusals, or
    ``max_consecutive_errors`` failed tool calls in a row, the run is stopped with LoopDetected.
    """
    max_identical_calls: int = 3
    max_blocked_calls: int = 2
    max_consecutive_errors: int = 4


class LoopDetector:
    def __init__(self, policy: LoopPolicy):
        self.policy = policy
        self._executed: dict[str, int] = {}
        self._blocked = 0
        self._error_streak = 0

    @staticmethod
    def _signature(name: str, args: dict) -> str:
        return name + json.dumps(args, sort_keys=True, default=str)

    def check(self, name: str, args: dict) -> str | None:
        """Return a message for the model if this call must not run; raise LoopDetected to stop the run."""
        if self._executed.get(self._signature(name, args), 0) < self.policy.max_identical_calls:
            return None
        self._blocked += 1
        if self._blocked > self.policy.max_blocked_calls:
            raise LoopDetected(f"{name} repeated with identical arguments.")
        return (f"This exact call to {name} was already made {self.policy.max_identical_calls} times in this "
                "run. Do not repeat it; answer with the information you already have.")

    def record(self, name: str, args: dict, status: str) -> None:
        """Register an executed call and its result status."""
        key = self._signature(name, args)
        self._executed[key] = self._executed.get(key, 0) + 1
        self._error_streak = self._error_streak + 1 if status == "error" else 0
        if self._error_streak >= self.policy.max_consecutive_errors:
            raise LoopDetected(f"{self._error_streak} consecutive tool failures.")
