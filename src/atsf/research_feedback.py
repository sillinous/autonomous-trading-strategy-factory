from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .genome import genome_distance
from .research_director import ResearchSignal
from .research_queue import ResearchReason


@dataclass(frozen=True)
class ResearchFeedbackResult:
    generation: int
    signals: tuple[ResearchSignal, ...]


def _unit(value: float, *, default: float = 0.0) -> float:
    if not isfinite(value):
        return default
    return min(1.0, max(0.0, value))


def _fitness_score(evaluation) -> float:
    score = getattr(getattr(evaluation, "fitness", None), "score", 0.0)
    if not isfinite(score):
        return 0.0
    return _unit(max(0.0, score))


def _novelty(candidate, population: tuple) -> float:
    if candidate is None:
        return 0.0
    peers = [other for other in population if other.strategy_id != candidate.strategy_id]
    if not peers:
        return 1.0
    distances = [genome_distance(candidate.strategy, other.strategy) for other in peers]
    return _unit(sum(distances) / len(distances))


def _signal_for(evaluation, *, candidate=None, population: tuple = ()) -> ResearchSignal:
    candidate_id = str(evaluation.candidate_id)
    validation_passed = bool(evaluation.validation_passed)
    promotion = getattr(evaluation, "promotion", None)
    eligible = bool(getattr(promotion, "eligible", False))
    monte_carlo = getattr(evaluation, "monte_carlo", None)
    simulations = int(getattr(monte_carlo, "simulations", 0))
    pass_rate = float(getattr(monte_carlo, "pass_rate", 0.0))

    if not validation_passed or not eligible:
        reason = ResearchReason.DEGRADED
    elif simulations <= 0:
        reason = ResearchReason.HALTED
    else:
        reason = ResearchReason.CAPACITY

    uncertainty = 1.0 if simulations <= 0 else _unit(1.0 - pass_rate, default=1.0)
    robustness = 1.0 if bool(getattr(getattr(evaluation, "robustness", None), "passed", False)) else 0.0
    capacity_gap = 1.0 if eligible else 0.0

    return ResearchSignal(
        strategy_id=candidate_id,
        reason=reason,
        fitness=_fitness_score(evaluation),
        robustness=robustness,
        novelty=_novelty(candidate, population),
        uncertainty=uncertainty,
        capacity_gap=capacity_gap,
    )


def build_research_feedback(
    evaluations,
    *,
    generation: int = 0,
    population=(),
) -> ResearchFeedbackResult:
    """Convert deterministic evaluation outcomes into the next research signals."""
    if not isinstance(generation, int) or generation < 0:
        raise ValueError("generation must be a nonnegative integer")
    population = tuple(population)
    ordered = sorted(tuple(evaluations), key=lambda evaluation: str(evaluation.candidate_id))
    signals = tuple(
        _signal_for(
            evaluation,
            candidate=next((item for item in population if item.strategy_id == evaluation.candidate_id), None),
            population=population,
        )
        for evaluation in ordered
    )
    return ResearchFeedbackResult(generation=generation, signals=signals)
