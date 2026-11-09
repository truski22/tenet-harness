"""Provider contract: how the harness obtains a chat model."""
from typing import Protocol

from langchain_core.language_models.chat_models import BaseChatModel


class Provider(Protocol):
    def chat(self, structured: bool = False) -> BaseChatModel: ...
