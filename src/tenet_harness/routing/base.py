"""Router contract: choose which optional toolsets a turn needs."""
from typing import Protocol

from ..tools import Toolset


class Router(Protocol):
    def select(self, text: str, candidates: list[Toolset]) -> list[Toolset]: ...
