from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .genome import genome_distance
from .orchestrator import CandidateEvaluation
from .population import Candidate


@dataclass(frozen=True)
class SelectionPolicy:
    """Deterministic selection policy for evaluated strategy populations."""

    population_size: int
    require_promotion: bool = True
    max_drawdown: float = 1.0
    min_robustness_equity_ratio: float = 0.0
    min_monte_carlo_pass_rate: float = 0.0
    preserve_diversity: bool = True
    min_genome_distance: float = 0.0

    def __post_init__(self) -> None:
        if self.population_size <= 0:
            raise ValueError("population_size must be positive")
        if not 0 <= self.max_drawdown <= 1:
            raise ValueError("max_drawdown must be between 0 and 1")
        if not 0 <= self.min_robustness_equity_ratio <= 1:
            raise ValueError("min_robustness_equity_ratio must be between 0 and 1")
        if not 0 <= self.min_monte_carlo_pass_rate <= 1:
            raise ValueError("min_monte_carlo_pass_rate must be between 0 and 1")
        if not 0 <= self.min_genome_distance <= 1:
            raise ValueError("min_genome_distance must be between 0 and 1")


def _robustness_ratio(evaluation: CandidateEvaluation) -> float:
    baseline = float(evaluation.robustness.baseline.equity.iloc[-1])
    if baseline <= 0 or not isfinite(baseline):
        return 0.0
    stressed = [float(result.equity.iloc[-1]) for _, result in evaluation.robustness.scenarios]
    finite = [value for value in stressed if isfinite(value)]
    if not finite:
        return 0.0
    return min(finite) / baseline


def _eligible(evaluation: CandidateEvaluation, policy: SelectionPolicy) -> bool:
    if policy.require_promotion and not evaluation.promotion.eligible:
        return False
    if not evaluation.validation_passed:
        return False
    if evaluation.backtest.max_drawdown > policy.max_drawdown:
        return False
    if _robustness_ratio(evaluation) < policy.min_robustness_equity_ratio:
        return False
    if evaluation.monte_carlo.pass_rate < policy.min_monte_carlo_pass_rate:
        return False
    return isfinite(evaluation.fitness.score)


def _rank_key(evaluation: CandidateEvaluation) -> tuple:
    """Stable scalar ordering; Pareto filtering is performed before this key."""
    robustness = _robustness_ratio(evaluation)
    return (
        float(evaluation.fitness.score),
        float(evaluation.monte_carlo.pass_rate),
        robustness,
        float(evaluation.regime.score),
        evaluation.candidate_id,
    )


def _dominates(left: CandidateEvaluation, right: CandidateEvaluation) -> bool:
    """Return whether left is no worse on all selection objectives and better on one."""
    left_values = (
        float(left.fitness.score),
        -float(left.backtest.max_drawdown),
        float(left.monte_carlo.pass_rate),
        _robustness_ratio(left),
        float(left.regime.score),
    )
    right_values = (
        float(right.fitness.score),
        -float(right.backtest.max_drawdown),
        float(right.monte_carlo.pass_rate),
        _robustness_ratio(right),
        float(right.regime.score),
    )
    return all(a >= b for a, b in zip(left_values, right_values)) and any(
        a > b for a, b in zip(left_values, right_values)
    )


def pareto_front(evaluations: list[CandidateEvaluation]) -> list[CandidateEvaluation]:
    """Return the non-dominated evaluations in deterministic order."""
    valid = [evaluation for evaluation in evaluations if isfinite(evaluation.fitness.score)]
    front = [
        evaluation
        for evaluation in valid
        if not any(
            other.candidate_id != evaluation.candidate_id
            and _dominates(other, evaluation)
            for other in valid
        )
    ]
    return sorted(front, key=_rank_key, reverse=True)


def _select_diverse(
    front: list[CandidateEvaluation],
    by_id: dict[str, Candidate],
    population_size: int,
    minimum_distance: float,
) -> list[str]:
    """Select a quality-ranked max-min diverse subset of the Pareto front."""
    if not front or population_size <= 0:
        return []

    selected: list[str] = [front[0].candidate_id]
    while len(selected) < population_size:
        options = [item for item in front if item.candidate_id not in selected]
        if not options:
            break

        scored: list[tuple[float, tuple, str]] = []
        for evaluation in options:
            candidate = by_id[evaluation.candidate_id]
            distances = [
                genome_distance(candidate.strategy, by_id[selected_id].strategy)
                for selected_id in selected
            ]
            minimum = min(distances)
            scored.append((minimum, _rank_key(evaluation), evaluation.candidate_id))

        admissible = [item for item in scored if item[0] >= minimum_distance]
        if not admissible:
            break
        _, _, candidate_id = max(admissible, key=lambda item: (item[0], item[1], item[2]))
        selected.append(candidate_id)

    return selected


def select_population(
    candidates: list[Candidate],
    evaluations: list[CandidateEvaluation],
    policy: SelectionPolicy,
) -> list[Candidate]:
    """Select a deterministic, promotion-aware population from evaluated candidates.

    Selection is deliberately separate from variation. The evaluator remains the
    source of truth for fitness and robustness evidence; this function only chooses
    which already-evaluated candidates survive to become parents.
    """
    if not candidates:
        raise ValueError("candidates must not be empty")
    if len(evaluations) != len(candidates):
        raise ValueError("candidates and evaluations must have equal length")
    by_id = {candidate.strategy_id: candidate for candidate in candidates}
    if len(by_id) != len(candidates):
        raise ValueError("candidate strategy IDs must be unique")
    if {evaluation.candidate_id for evaluation in evaluations} != set(by_id):
        raise ValueError("evaluations must exactly match candidate IDs")

    eligible = [evaluation for evaluation in evaluations if _eligible(evaluation, policy)]
    if not eligible:
        raise ValueError("no candidates satisfy selection gates")

    front = pareto_front(eligible)
    selected_ids: list[str] = []
    selected_set: set[str] = set()

    def add(evaluation: CandidateEvaluation) -> None:
        if evaluation.candidate_id not in selected_set and len(selected_ids) < policy.population_size:
            selected_set.add(evaluation.candidate_id)
            selected_ids.append(evaluation.candidate_id)

    if policy.preserve_diversity:
        selected_ids.extend(
            _select_diverse(
                front,
                by_id,
                policy.population_size,
                policy.min_genome_distance,
            )
        )
        selected_set.update(selected_ids)
    else:
        for evaluation in front:
            add(evaluation)

    # Diversity is a preference, not a reason to fail a valid research cycle when
    # the eligible pool cannot satisfy the configured distance threshold.
    remaining = sorted(
        (evaluation for evaluation in eligible if evaluation.candidate_id not in selected_set),
        key=_rank_key,
        reverse=True,
    )
    for evaluation in remaining:
        add(evaluation)

    if len(selected_ids) < policy.population_size:
        raise ValueError("fewer eligible candidates than requested population size")
    return [by_id[candidate_id] for candidate_id in selected_ids]
