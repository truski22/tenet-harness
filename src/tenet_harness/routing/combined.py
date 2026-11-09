"""Union of routers that tolerates a failing member."""
import logging

from ..tools import Toolset
from .base import Router

log = logging.getLogger(__name__)


class CombinedRouter:
    """Union of routers; a failing router (e.g. embeddings offline) is skipped."""

    def __init__(self, *routers: Router):
        self.routers = routers

    def select(self, text: str, candidates: list[Toolset]) -> list[Toolset]:
        selected: dict[str, Toolset] = {}
        for router in self.routers:
            try:
                selected.update((ts.id, ts) for ts in router.select(text, candidates))
            except Exception as exc:
                log.warning("Router %s unavailable (%s)", type(router).__name__, type(exc).__name__)
        return [ts for ts in candidates if ts.id in selected]
