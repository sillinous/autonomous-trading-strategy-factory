from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from random import Random
from statistics import median
from typing import Callable

from .generator import mutate_indicator_period, mutate_threshold
from .population import strategy_id
from .strategy import StrategySpec


@dataclass(frozen=True)
class PerturbationResult:
    samples: int
    seed: int
    pass_rate: float
    worst_score: float
    median_score: float
    strategy_ids: tuple[str, ...]


def evaluate_parameter_perturbations(
    strategy: StrategySpec,
    evaluator: Callable[[StrategySpec], float],
    samples: int = 20,
    seed: int = 0,
) -> PerturbationResult:
    """Evaluate nearby constrained strategies using deterministic mutations."""
    if samples <= 0:
        raise ValueError("samples must be positive")
    if not strategy.indicators:
        raise ValueError("strategy must contain indicators")

    rng = Random(seed)
    mutations = (mutate_indicator_period, mutate_threshold)
    scores: list[float] = []
    ids: list[str] = []
    for _ in range(samples):
        mutation = rng.choice(mutations)
        try:
            candidate = mutation(strategy, rng)
        except (TypeError, ValueError):
            candidate = mutate_indicator_period(strategy, rng)
        score = float(evaluator(candidate))
        scores.append(score)
        ids.append(strategy_id(candidate))

    finite_scores = [score for score in scores if isfinite(score)]
    if not finite_scores:
        raise ValueError("evaluator produced no finite scores")
    return PerturbationResult(
        samples=samples,
        seed=seed,
        pass_rate=sum(isfinite(score) and score >= 0 for score in scores) / samples,
        worst_score=min(finite_scores),
        median_score=float(median(finite_scores)),
        strategy_ids=tuple(ids),
    )
