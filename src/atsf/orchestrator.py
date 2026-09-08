from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .backtest import BacktestConfig, BacktestResult, run_long_signal_backtest
from .experiment import ExperimentResult, ExperimentSpec
from .fitness import FitnessPolicy, FitnessResult, score_strategy
from .population import Candidate
from .signals import strategy_signals
from .splits import chronological_split
from .validation import ValidationPolicy, validate_equity


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate_id: str
    experiment: ExperimentResult
    backtest: BacktestResult
    validation_passed: bool
    fitness: FitnessResult


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
    """Run one candidate through the deterministic research pipeline."""
    split = chronological_split(data)
    entry, exit_ = strategy_signals(split.train, candidate.strategy)
    signal = entry.copy()
    in_position = False
    for index in signal.index:
        if exit_.loc[index]:
            in_position = False
        if entry.loc[index]:
            in_position = True
        signal.loc[index] = in_position

    backtest = run_long_signal_backtest(split.train, signal, backtest_config)
    validation = validate_equity(backtest.equity, validation_policy)
    fitness = score_strategy(validation.sharpe, validation.drawdown, fitness_policy)
    experiment = ExperimentResult(
        experiment_id=ExperimentSpec(
            candidate.strategy, dataset_id, dataset_version, seed
        ).experiment_id,
        status="passed" if fitness.eligible else "rejected",
        score=fitness.score,
        reason="; ".join(fitness.reasons) if fitness.reasons else None,
    )
    return CandidateEvaluation(
        candidate_id=candidate.strategy_id,
        experiment=experiment,
        backtest=backtest,
        validation_passed=validation.passed,
        fitness=fitness,
    )
