"""Semantic routing by similarity between the message and each toolset's example requests."""
import math
from typing import Callable, Sequence

from ..tools import Toolset


class EmbeddingRouter:
    """Semantic match: best cosine similarity between the message and each toolset's examples.

    Example utterances separate far better than a one-line summary (measured with bge-m3:
    in-domain questions 0.69-0.93, unrelated chat <= 0.58), hence the default threshold.
    """

    def __init__(self, embed: Callable[[list[str]], list[list[float]]], threshold: float = 0.62,
                 top_k: int = 2):
        self.embed, self.threshold, self.top_k = embed, threshold, top_k
        self._cache: dict[str, list[float]] = {}

    @staticmethod
    def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
        norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
        return sum(x * y for x, y in zip(a, b)) / norm if norm else 0.0

    def select(self, text: str, candidates: list[Toolset]) -> list[Toolset]:
        samples = {ts.id: ts.examples or (ts.summary,) for ts in candidates}
        missing = list(dict.fromkeys(s for group in samples.values() for s in group if s not in self._cache))
        query, *vectors = self.embed([text, *missing])
        self._cache.update(zip(missing, vectors))
        scored = sorted(((max(self._cosine(query, self._cache[s]) for s in samples[ts.id]), ts)
                         for ts in candidates), key=lambda pair: pair[0], reverse=True)
        return [ts for score, ts in scored[:self.top_k] if score >= self.threshold]
