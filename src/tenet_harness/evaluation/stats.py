"""Small statistics helpers for repeated-run evaluation."""
import math


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval for a pass rate. Wide on purpose: it shows how little a small suite proves."""
    if total == 0:
        return 0.0, 0.0
    p, d = successes / total, 1 + z * z / total
    centre, margin = p + z * z / (2 * total), z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return round((centre - margin) / d, 3), round((centre + margin) / d, 3)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1)], 2)
