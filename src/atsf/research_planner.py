from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .genome import genome_distance
from .research_budget import ResearchAllocation, ResearchBudgetPolicy, allocate_budget
from .research_director import ResearchDirectorPolicy, ResearchSignal, prioritize
from .research_queue import ResearchReason, ResearchRequest
from .scheduler import GenerationResult


@dataclass(frozen=True)
class ResearchPlan:
    """Deterministic research decisions derived from one evaluated generation."""

    generation: int
    requests: tuple[ResearchRequest, ...]
    allocations: tuple[ResearchAllocation, ...]


def _novelty(candidate, population) -> float:
    peers = [other for other in population if other.strategy_id != candidate.strategy_id]
    if not peers:
        return 1.0
    distances = [genome_distance(candidate.strategy, other.strategy) for other in peers]
    return sum(distances) / len(distances)


def build_research_plan(
    result: GenerationResult,
    *,
    director_policy: ResearchDirectorPolicy | None = None,
    budget_policy: ResearchBudgetPolicy | None = None,
    population=None,
    additional_signals: tuple[ResearchSignal, ...] = (),
) -> ResearchPlan:
    """Translate generation and cross-cutting portfolio evidence into research work."""
    if population is None:
        population = result.survivors
    population = tuple(population)
    by_id = {candidate.strategy_id: candidate for candidate in population}
    signals: list[ResearchSignal] = []

    for evaluation in result.evaluations:
        candidate = by_id.get(evaluation.candidate_id)
        novelty = _novelty(candidate, population) if candidate is not None else 0.0
        fitness = max(0.0, evaluation.fitness.score) if isfinite(evaluation.fitness.score) else 0.0
        robustness = 1.0 if evaluation.robustness.passed else 0.0
        uncertainty = 1.0 - evaluation.monte_carlo.pass_rate
        if not isfinite(uncertainty):
            uncertainty = 1.0
        capacity_gap = 1.0 if candidate is None else 0.0

        if not evaluation.validation_passed or not evaluation.promotion.eligible:
            reason = ResearchReason.DEGRADED
        elif evaluation.monte_carlo.simulations == 0:
            reason = ResearchReason.HALTED
        elif novelty < 0.25:
            reason = ResearchReason.DIVERSIFICATION
        else:
            reason = ResearchReason.CAPACITY

        signals.append(
            ResearchSignal(
                strategy_id=evaluation.candidate_id,
                reason=reason,
                fitness=fitness,
                robustness=robustness,
                novelty=max(0.0, novelty),
                uncertainty=max(0.0, min(1.0, uncertainty)),
                capacity_gap=capacity_gap,
            )
        )

    signals.extend(additional_signals)
    requests = prioritize(signals, policy=director_policy)
    request_ids = [request.request_id for request in requests]
    signal_by_key = {(signal.strategy_id, signal.reason): signal for signal in signals}
    request_signals = [
        signal_by_key[(request.source_strategy_id, request.reason)] for request in requests
    ]
    allocations = allocate_budget(
        request_signals,
        request_ids=request_ids,
        policy=budget_policy,
    )
    return ResearchPlan(
        generation=result.generation,
        requests=requests,
        allocations=allocations,
    )