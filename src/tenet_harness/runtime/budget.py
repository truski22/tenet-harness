"""Token accounting and budgets."""
from dataclasses import dataclass
from datetime import datetime, timezone

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.messages.utils import count_tokens_approximately

from ..errors import BudgetExceeded


@dataclass(frozen=True)
class Budget:
    """Limits on model tokens (input + output). ``None`` means unlimited.

    The per-run limit stops a run after the model call that crosses it. The per-day limit is checked
    when a run is submitted, over all runs started since 00:00 UTC.
    """
    max_tokens_per_run: int | None = None
    max_tokens_per_day: int | None = None


def start_of_day_utc() -> str:
    now = datetime.now(timezone.utc)
    return now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()


class UsageMeter:
    """Tokens consumed by one run. Uses provider-reported usage; estimates it when absent."""

    def __init__(self, limit: int | None = None):
        self.limit = limit
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0
        self.estimated_calls = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    def add(self, prompt: list[BaseMessage], reply: AIMessage) -> None:
        usage = reply.usage_metadata
        if usage and (usage.get("input_tokens") or usage.get("output_tokens")):
            self.input_tokens += usage.get("input_tokens", 0)
            self.output_tokens += usage.get("output_tokens", 0)
        else:
            self.input_tokens += count_tokens_approximately(prompt)
            self.output_tokens += count_tokens_approximately([reply])
            self.estimated_calls += 1
        self.calls += 1

    def check(self) -> None:
        if self.limit is not None and self.total >= self.limit:
            raise BudgetExceeded(f"Run used {self.total} tokens (limit {self.limit}).")

    def summary(self) -> dict:
        return {"input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                "model_calls": self.calls, "estimated_calls": self.estimated_calls}
