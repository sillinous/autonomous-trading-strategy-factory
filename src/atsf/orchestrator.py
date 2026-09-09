from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .backtest import BacktestConfig, BacktestResult
from .evaluation import WalkForwardEvaluation, evaluate_walk_forward
from .experiment import ExperimentResult, ExperimentSpec
from .fitness import FitnessPolicy, FitnessResult
from .perturbation import PerturbationResult, evaluate_parameter_perturbations
from .population import Candidate
from .promotion import PromotionDecision, PromotionPolicy, research_to_paper
from .robustness import (
    MonteCarloResult,
    RegimeStabilityResult,
    RobustnessResult,
    monte_carlo_trade_bootstrap,
    regime_returns,
    score_regime_stability,
    test_robustness,
)
from .signals import strategy_signals
from .validation import ValidationPolicy


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate_id: str
    experiment: ExperimentResult
    backtest: BacktestResult
    validation_passed: bool
    fitness: FitnessResult
    walk_forward: WalkForwardEvaluation
    monte_carlo: MonteCarloResult
    perturbation: PerturbationResult
    regime: RegimeStabilityResult
    robustness: RobustnessResult
    promotion: PromotionDecision


def evaluate_candidate(
    candidate: Candidate,
    data: pd.DataFrame,
    dataset_id: str,
    dataset_version: str,
    seed: int = 0,
    backtest_config: BacktestConfig | None = None,
    validation_policy: ValidationPolicy | None = None,
    fitness_policy: FitnessPolicy | None = None,
    promotion_policy: PromotionPolicy | None = None,
    benchmark: pd.Series | None = None,
    perturbation_samples: int = 20,
) -> CandidateEvaluation:
    """Run one candidate through deterministic validation and promotion gates."""
    if candidate.strategy.side.value != "long":
        raise NotImplementedError("short-side execution is not implemented yet")
    if len(data) < 10:
        raise ValueError("data must contain at least 10 rows")

    train_size = max(2, int(len(data) * 0.6))
    validation_size = max(2, int(len(data) * 0.2))
    test_size = len(data) - train_size - validation_size
    if test_size < 2:
        raise ValueError("data is too short for train/validation/OOS evaluation")

    walk_forward = evaluate_walk_forward(
        data,
        candidate.strategy,
        train_size=train_size,
        validation_size=validation_size,
        test_size=test_size,
        backtest_config=backtest_config,
        validation_policy=validation_policy,
        fitness_policy=fitness_policy,
    )

    if walk_forward.oos_trade_returns:
        monte_carlo = monte_carlo_trade_bootstrap(
            walk_forward.oos_trade_returns, seed=seed
        )
    else:
        monte_carlo = MonteCarloResult(
            simulations=0,
            seed=seed,
            median_return=-1.0,
            worst_return=-1.0,
            lower_percentile_return=-1.0,
            pass_rate=0.0,
        )

    def perturbation_score(strategy) -> float:
        result = evaluate_walk_forward(
            data,
            strategy,
            train_size=train_size,
            validation_size=validation_size,
            test_size=test_size,
            backtest_config=backtest_config,
            validation_policy=validation_policy,
            fitness_policy=fitness_policy,
        )
        return result.oos_sharpe

    perturbation = evaluate_parameter_perturbations(
        candidate.strategy,
        perturbation_score,
        samples=perturbation_samples,
        seed=seed,
    )

    oos_equity = walk_forward.oos_equity
    if oos_equity is None:
        raise ValueError("walk-forward evaluation did not produce OOS equity")
    benchmark_series = benchmark if benchmark is not None else data["close"]
    benchmark_series = benchmark_series.reindex(oos_equity.index)
    if benchmark_series.isna().any():
        raise ValueError("benchmark does not cover all OOS timestamps")
    regime = score_regime_stability(regime_returns(oos_equity, benchmark_series))

    robustness = test_robustness(data, candidate, config=backtest_config)
    promotion = research_to_paper(
        walk_forward,
        monte_carlo,
        perturbation,
        regime,
        robustness,
        promotion_policy,
    )
    fitness = FitnessResult(
        score=walk_forward.oos_sharpe,
        eligible=promotion.eligible,
        reasons=promotion.reasons,
    )
    experiment_id = ExperimentSpec(
        candidate.strategy, dataset_id, dataset_version, seed
    ).experiment_id
    experiment = ExperimentResult(
        experiment_id=experiment_id,
        status=promotion.stage,
        score=fitness.score,
        reason="; ".join(fitness.reasons) if fitness.reasons else None,
    )
    first_train = walk_forward.windows[0]
    return CandidateEvaluation(
        candidate_id=candidate.strategy_id,
        experiment=experiment,
        backtest=first_train.backtest,
        validation_passed=walk_forward.passed,
        fitness=fitness,
        walk_forward=walk_forward,
        monte_carlo=monte_carlo,
        perturbation=perturbation,
        regime=regime,
        robustness=robustness,
        promotion=promotion,
    )
