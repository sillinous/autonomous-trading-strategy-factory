from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .research_director import ResearchSignal


@dataclass(frozen=True)
class ResearchBudgetPolicy:
    """Bounded allocation policy for research effort."""

    total_units: int = 100
    min_units: int = 1
    max_units_per_request: int = 60

    def __post_init__(self) -> None:
        if self.total_units <= 0:
            raise ValueError("total_units must be positive")
        if self.min_units <= 0:
            raise ValueError("min_units must be positive")
        if self.max_units_per_request < self.min_units:
            raise ValueError("max_units_per_request cannot be below min_units")


@dataclass(frozen=True)
class ResearchAllocation:
    request_id: str
    units: int
    score: float


def allocate_budget(
    signals: list[ResearchSignal],
    *,
    request_ids: list[str] | None = None,
    policy: ResearchBudgetPolicy | None = None,
) -> tuple[ResearchAllocation, ...]:
    """Allocate finite research units proportionally, deterministically, and fail closed."""
    active = policy or ResearchBudgetPolicy()
    if request_ids is not None and len(request_ids) != len(signals):
        raise ValueError("request_ids must match signals")
    ids = request_ids or [
        f"research:{signal.reason.value}:{signal.strategy_id or 'global'}"
        for signal in signals
    ]
    scores: list[float] = []
    for signal in signals:
        values = (
            signal.fitness,
            signal.robustness,
            signal.novelty,
            signal.uncertainty,
            signal.capacity_gap,
        )
        if any(not isfinite(value) or value < 0 for value in values):
            raise ValueError("research signals must be finite and non-negative for budgeting")
        score = sum(values)
        scores.append(score)

    if not signals:
        return ()
    total_score = sum(scores)
    if total_score <= 0:
        scores = [1.0] * len(signals)
        total_score = float(len(signals))

    raw = [active.total_units * score / total_score for score in scores]
    units = [max(active.min_units, min(active.max_units_per_request, int(value))) for value in raw]

    while sum(units) > active.total_units:
        candidates = [i for i, value in enumerate(units) if value > active.min_units]
        if not candidates:
            break
        index = min(candidates, key=lambda i: (scores[i] / max(units[i], 1), ids[i]))
        units[index] -= 1

    while sum(units) < active.total_units:
        candidates = [i for i, value in enumerate(units) if value < active.max_units_per_request]
        if not candidates:
            break
        index = max(candidates, key=lambda i: (scores[i] / max(units[i], 1), -i))
        units[index] += 1

    return tuple(
        ResearchAllocation(request_id=request_id, units=units[i], score=scores[i])
        for i, request_id in enumerate(ids)
    )
