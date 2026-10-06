from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
import pandas as pd

from .genome import genome_distance
from .orchestrator import CandidateEvaluation
from .population import Candidate
from .statistics import OverfittingReport, deflated_sharpe_ratio, probability_of_backtest_overfitting


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
    min_deflated_sharpe: float = 0.95
    max_pbo: float = 0.5
    pbo_blocks: int = 10

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
        if not 0 <= self.min_deflated_sharpe < 1:
            raise ValueError("min_deflated_sharpe must be in [0, 1)")
        if not 0 < self.max_pbo <= 1:
            raise ValueError("max_pbo must be in (0, 1]")
        if self.pbo_blocks < 2 or self.pbo_blocks % 2:
            raise ValueError("pbo_blocks must be an even integer >= 2")


def _oos_returns(evaluation: CandidateEvaluation) -> pd.Series | None:
    walk_forward = getattr(evaluation, "walk_forward", None)
    returns = getattr(walk_forward, "oos_returns", None)
    return returns if isinstance(returns, pd.Series) and len(returns) >= 3 else None


def population_trial_statistics(evaluations: list[CandidateEvaluation]) -> tuple[int, float | None]:
    """Number of trials and cross-sectional variance of per-period OOS Sharpes."""
    sharpes = []
    for evaluation in evaluations:
        returns = _oos_returns(evaluation)
        if returns is not None and float(returns.std(ddof=1)) > 0:
            sharpes.append(float(returns.mean() / returns.std(ddof=1)))
    variance = float(np.var(sharpes, ddof=1)) if len(sharpes) >= 2 else None
    return len(evaluations), variance


def population_overfitting(evaluations: list[CandidateEvaluation], n_blocks: int = 10) -> OverfittingReport | None:
    """PBO across the evaluated population's aligned OOS return streams, if computable."""
    series = {evaluation.candidate_id: _oos_returns(evaluation) for evaluation in evaluations}
    series = {key: value for key, value in series.items() if value is not None}
    if len(series) < 2:
        return None
    frame = pd.concat(series, axis=1, join="inner").dropna()
    frame = frame.T.drop_duplicates().T  # identical return streams are a single trial
    if frame.shape[1] < 2 or len(frame) < n_blocks * 2:
        return None
    return probability_of_backtest_overfitting(frame, n_blocks)


def _robustness_ratio(evaluation: CandidateEvaluation) -> float:
    baseline = float(evaluation.robustness.baseline.equity.iloc[-1])
    if baseline <= 0 or not isfinite(baseline):
        return 0.0
    stressed = [float(result.equity.iloc[-1]) for _, result in evaluation.robustness.scenarios]
    finite = [value for value in stressed if isfinite(value)]
    if not finite:
        return 0.0
    return min(finite) / baseline


def _eligible(evaluation: CandidateEvaluation, policy: SelectionPolicy, n_trials: int = 1,
              trial_variance: float | None = None) -> bool:
    if policy.require_promotion and not evaluation.promotion.eligible:
        return False
    if not evaluation.validation_passed:
        return False
    if policy.min_deflated_sharpe > 0:
        returns = _oos_returns(evaluation)
        if returns is None:
            return False
        if deflated_sharpe_ratio(returns, n_trials, trial_variance).probability < policy.min_deflated_sharpe:
            return False
    # drawdowns are reported as negative fractions; compare magnitudes
    if abs(float(evaluation.backtest.max_drawdown)) > policy.max_drawdown:
        return False
    if _robustness_ratio(evaluation) < policy.min_robustness_equity_ratio:
        return False
    if evaluation.monte_carlo.pass_rate < policy.min_monte_carlo_pass_rate:
        return False
    return isfinite(evaluation.fitness.score)


def _rank_key(evaluation: CandidateEvaluation) -> tuple:
    """Stable scalar ordering used after eligibility and diversity decisions."""
    return (
        float(evaluation.fitness.score),
        float(evaluation.monte_carlo.pass_rate),
        _robustness_ratio(evaluation),
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
    evaluations: list[CandidateEvaluation],
    by_id: dict[str, Candidate],
    population_size: int,
    minimum_distance: float,
) -> list[str]:
    """Select a quality-aware max-min diverse subset of the eligible pool.

    Diversity is evaluated across every eligible candidate, not just the Pareto
    front. This prevents a high-quality but dominated strategy family from being
    erased solely because it is dominated on current performance objectives.
    """
    ordered = sorted(evaluations, key=_rank_key, reverse=True)
    if not ordered or population_size <= 0:
        return []

    selected: list[str] = [ordered[0].candidate_id]
    while len(selected) < population_size:
        options = [item for item in ordered if item.candidate_id not in selected]
        if not options:
            break

        scored: list[tuple[float, tuple, str]] = []
        for evaluation in options:
            candidate = by_id[evaluation.candidate_id]
            minimum = min(
                genome_distance(candidate.strategy, by_id[selected_id].strategy)
                for selected_id in selected
            )
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
    *,
    prior_trials: int = 0,
) -> list[Candidate]:
    """Select a deterministic, promotion-aware population from evaluated candidates.

    Selection is deliberately separate from variation. Pareto analysis remains
    available as an objective-quality diagnostic, while survivor selection can
    preserve structurally diverse strategies across the complete eligible pool.
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

    overfitting = population_overfitting(evaluations, policy.pbo_blocks)
    if overfitting is not None and overfitting.pbo > policy.max_pbo:
        raise ValueError(
            f"population is overfit: PBO {overfitting.pbo:.2f} exceeds {policy.max_pbo:.2f}"
        )
    if prior_trials < 0:
        raise ValueError("prior_trials must be non-negative")
    n_trials, trial_variance = population_trial_statistics(evaluations)
    # Every variant evaluated in earlier generations counts toward the search budget.
    n_trials += prior_trials
    eligible = [
        evaluation for evaluation in evaluations
        if _eligible(evaluation, policy, n_trials, trial_variance)
    ]
    if not eligible:
        raise ValueError("no candidates satisfy selection gates")

    selected_ids: list[str] = []
    selected_set: set[str] = set()

    def add(evaluation: CandidateEvaluation) -> None:
        if evaluation.candidate_id not in selected_set and len(selected_ids) < policy.population_size:
            selected_set.add(evaluation.candidate_id)
            selected_ids.append(evaluation.candidate_id)

    if policy.preserve_diversity:
        selected_ids.extend(
            _select_diverse(
                eligible,
                by_id,
                policy.population_size,
                policy.min_genome_distance,
            )
        )
        selected_set.update(selected_ids)
    else:
        for evaluation in pareto_front(eligible):
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
