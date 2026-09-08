from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .backtest import BacktestConfig, BacktestResult
from .evaluation import WalkForwardEvaluation, evaluate_walk_forward
from .experiment import ExperimentResult, ExperimentSpec
from .fitness import FitnessPolicy, FitnessResult
from .population import Candidate
from .validation import ValidationPolicy


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate_id: str
    experiment: ExperimentResult
    backtest: BacktestResult
    validation_passed: bool
    fitness: FitnessResult
    walk_forward: WalkForwardEvaluation


def evaluate_candidate(
    candidate: Candidate,
    data: pd.DataFrame,
    dataset_id: str,
    dataset_version: str,
    seed: int = 0,
    backtest_config: BacktestConfig | None = None,
    validation_policy: ValidationPolicy | None = None,
    fitness_policy: FitnessPolicy | None = None,
) -> CandidateEvaluation:
    """Run one candidate through the authoritative train/validation/OOS pipeline."""
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
    first_train = walk_forward.windows[0]
    fitness = FitnessResult(
        score=walk_forward.oos_sharpe,
        eligible=walk_forward.passed,
        reasons=() if walk_forward.passed else ("walk-forward/OOS promotion gates failed",),
    )
    experiment_id = ExperimentSpec(
        candidate.strategy, dataset_id, dataset_version, seed
    ).experiment_id
    experiment = ExperimentResult(
        experiment_id=experiment_id,
        status="passed" if walk_forward.passed else "rejected",
        score=fitness.score,
        reason="; ".join(fitness.reasons) if fitness.reasons else None,
    )
    return CandidateEvaluation(
        candidate_id=candidate.strategy_id,
        experiment=experiment,
        backtest=first_train.backtest,
        validation_passed=walk_forward.passed,
        fitness=fitness,
        walk_forward=walk_forward,
    )
