"""Deterministic whole-word keyword routing."""
import re

from ..tools import Toolset


class KeywordRouter:
    """Deterministic match of whole words/phrases from ``Toolset.keywords``."""

    def select(self, text: str, candidates: list[Toolset]) -> list[Toolset]:
        folded = text.casefold()
        return [ts for ts in candidates if any(
            re.search(rf"(?<!\w){re.escape(keyword.casefold())}(?!\w)", folded) for keyword in ts.keywords)]
