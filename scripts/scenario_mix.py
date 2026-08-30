"""Deterministic scenario quotas for the official evaluator mix.

The evaluator's official proportions are 40% Buying, 40% Browsing, 15% Intent Override,
and 5% Boundary.  For finite generated sets the quotas are allocated with the largest-
remainder method, then shuffled so a prefix is not scenario-biased.
"""
from __future__ import annotations

import random


OFFICIAL_SCENARIOS: tuple[tuple[str, float], ...] = (
    ("buying", 0.40),
    ("browsing", 0.40),
    ("intent_override", 0.15),
    ("boundary", 0.05),
)


def largest_remainder_counts(n: int) -> dict[str, int]:
    """Return exact integer quotas summing to *n* using largest remainder.

    Ties are resolved in the published scenario order, making the result reproducible.
    """
    if n < 0:
        raise ValueError("scenario count must be non-negative")
    raw = [(name, n * weight) for name, weight in OFFICIAL_SCENARIOS]
    counts = {name: int(value) for name, value in raw}
    remaining = n - sum(counts.values())
    ranked_remainders = sorted(
        enumerate(raw), key=lambda item: (-(item[1][1] - int(item[1][1])), item[0]))
    for index, (name, _value) in ranked_remainders[:remaining]:
        counts[name] += 1
    if sum(counts.values()) != n:
        raise AssertionError("largest-remainder quotas do not sum to requested count")
    return counts


def scenario_schedule(n: int, seed: int) -> list[str]:
    """Build a shuffled schedule with the official finite-sample quotas."""
    counts = largest_remainder_counts(n)
    schedule = [name for name, _weight in OFFICIAL_SCENARIOS for _ in range(counts[name])]
    random.Random(seed).shuffle(schedule)
    return schedule


def assert_official_mix(scenarios: list[str], requested: int) -> None:
    """Fail loudly if a generator emitted the wrong number or scenario composition."""
    from collections import Counter

    expected = largest_remainder_counts(requested)
    actual = Counter(scenarios)
    if len(scenarios) != requested or any(actual[name] != count for name, count in expected.items()):
        raise AssertionError(f"scenario mix mismatch: expected {expected}, got {dict(actual)}")
