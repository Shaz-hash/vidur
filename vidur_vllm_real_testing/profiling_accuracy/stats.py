from __future__ import annotations

import math
import statistics
from typing import Iterable


def timing_stats(values_ms: Iterable[float]) -> dict[str, float]:
    values = [float(value) for value in values_ms]
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("timings must be a non-empty sequence of finite non-negative values")
    mean = statistics.fmean(values)
    return {
        "min": min(values),
        "max": max(values),
        "mean": mean,
        "median": statistics.median(values),
        "std": statistics.pstdev(values),
    }


def percentile(values: Iterable[float], percentile_value: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot calculate a percentile of an empty sequence")
    if not 0 <= percentile_value <= 100:
        raise ValueError("percentile must be in [0, 100]")
    position = (len(ordered) - 1) * percentile_value / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction
