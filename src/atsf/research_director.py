from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .research_queue import ResearchReason, ResearchRequest


@dataclass(frozen=True)
class ResearchSignal:
    """Evidence used to prioritize the next autonomous research request."""

    strategy_id: str | None
    reason: ResearchReason
    fitness: float = 0.0
    robustness: float = 0.0
    novelty: float = 0.0
    uncertainty: float = 0.0
    capacity_gap: float = 0.0


@dataclass(frozen=True)
class ResearchDirectorPolicy:
    """Deterministic weights and limits for research-budget allocation."""

    fitness_weight: float = 0.20
    robustness_weight: float = 0.15
    novelty_weight: float = 0.25
    uncertainty_weight: float = 0.25
    capacity_weight: float = 0.15
    max_requests: int = 3

    def __post_init__(self) -> None:
        weights = (
            self.fitness_weight,
            self.robustness_weight,
            self.novelty_weight,
            self.uncertainty_weight,
            self.capacity_weight,
        )
        if any(not isfinite(weight) or weight < 0 for weight in weights):
            raise ValueError("research weights must be finite and non-negative")
        if sum(weights) <= 0:
            raise ValueError("research weights must not all be zero")
        if self.max_requests <= 0:
            raise ValueError("max_requests must be positive")


def _priority(signal: ResearchSignal, policy: ResearchDirectorPolicy) -> float:
    values = (
        signal.fitness,
        signal.robustness,
        signal.novelty,
        signal.uncertainty,
        signal.capacity_gap,
    )
    if any(not isfinite(value) for value in values):
        raise ValueError("research signal values must be finite")
    total = (
        policy.fitness_weight * signal.fitness
        + policy.robustness_weight * signal.robustness
        + policy.novelty_weight * signal.novelty
        + policy.uncertainty_weight * signal.uncertainty
        + policy.capacity_weight * signal.capacity_gap
    )
    return total / (
        policy.fitness_weight
        + policy.robustness_weight
        + policy.novelty_weight
        + policy.uncertainty_weight
        + policy.capacity_weight
    )


def prioritize(
    signals: list[ResearchSignal],
    *,
    policy: ResearchDirectorPolicy | None = None,
) -> tuple[ResearchRequest, ...]:
    """Convert research evidence into a deterministic, bounded research queue."""
    active_policy = policy or ResearchDirectorPolicy()
    scored = [(_priority(signal, active_policy), signal) for signal in signals]
    scored.sort(
        key=lambda item: (
            -item[0],
            item[1].reason.value,
            item[1].strategy_id or "",
        )
    )
    requests: list[ResearchRequest] = []
    seen: set[tuple[str | None, ResearchReason]] = set()
    for score, signal in scored:
        key = (signal.strategy_id, signal.reason)
        if key in seen:
            continue
        seen.add(key)
        # Lower queue priority means earlier execution; preserve deterministic ordering.
        priority = max(0, int(round((1.0 - max(0.0, min(1.0, score))) * 1000)))
        request_id = f"research:{signal.reason.value}:{signal.strategy_id or 'global'}"
        requests.append(
            ResearchRequest(
                request_id=request_id,
                source_strategy_id=signal.strategy_id,
                reason=signal.reason,
                priority=priority,
            )
        )
        if len(requests) >= active_policy.max_requests:
            break
    return tuple(requests)
