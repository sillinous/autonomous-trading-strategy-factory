from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .fitness import FitnessResult


@dataclass(frozen=True)
class RankedCandidate:
    candidate_id: str
    fitness_score: float
    robustness_score: float
    diversity_score: float
    final_score: float


def rank_candidates(
    candidates: list[tuple[str, FitnessResult, float, float]],
    *,
    fitness_weight: float = 0.6,
    robustness_weight: float = 0.25,
    diversity_weight: float = 0.15,
) -> tuple[RankedCandidate, ...]:
    """Rank eligible candidates using performance, robustness and diversity evidence."""
    weights = (fitness_weight, robustness_weight, diversity_weight)
    if any(w < 0 or not isfinite(w) for w in weights) or sum(weights) <= 0:
        raise ValueError("ranking weights must be finite, non-negative, and non-zero")
    ranked: list[RankedCandidate] = []
    total = sum(weights)
    for candidate_id, fitness, robustness, diversity in candidates:
        if not fitness.eligible:
            continue
        values = (fitness.score, robustness, diversity)
        if not all(isfinite(value) for value in values):
            continue
        score = sum(weight * value for weight, value in zip(weights, values, strict=True)) / total
        ranked.append(RankedCandidate(candidate_id, fitness.score, robustness, diversity, score))
    return tuple(sorted(ranked, key=lambda item: (-item.final_score, -item.fitness_score, item.candidate_id)))
