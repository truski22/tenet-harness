"""Provider contract: how the harness obtains a chat model, and what that model can do."""
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from langchain_core.language_models.chat_models import BaseChatModel


@dataclass(frozen=True)
class Capabilities:
    """What a provider's model supports. Adapters declare it; the runtime checks it instead of assuming.

    ``context_window`` is the number of tokens the application may use for prompt plus history
    (``None`` if unknown). ``json_mode`` means ``chat(structured=True)`` constrains output to JSON;
    without it the adapter only asks for JSON and strips code fences. ``reports_usage`` says whether
    token usage comes from the provider (otherwise the harness estimates it).
    """
    tool_calling: bool = True
    streaming: bool = True
    json_mode: bool = True
    reports_usage: bool = True
    context_window: int | None = None


@runtime_checkable
class Provider(Protocol):
    def chat(self, structured: bool = False) -> BaseChatModel: ...


def capabilities_of(provider: object) -> Capabilities:
    """Declared capabilities, or the permissive defaults for providers that declare none."""
    declared = getattr(provider, "capabilities", None)
    return declared if isinstance(declared, Capabilities) else Capabilities()
