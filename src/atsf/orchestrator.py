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


def _position_signal(entry: pd.Series, exit_: pd.Series) -> pd.Series:
    """Turn entry/exit events into a held-position signal without look-ahead."""
    if not entry.index.equals(exit_.index):
        raise ValueError("entry and exit indexes must match")
    active = False
    values: list[bool] = []
    for timestamp in entry.index:
        if bool(exit_.loc[timestamp]):
            active = False
        if bool(entry.loc[timestamp]):
            active = True
        values.append(active)
    return pd.Series(values, index=entry.index, dtype=bool)


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
    """Run one long candidate through the deterministic research pipeline."""
    if candidate.strategy.side.value != "long":
        raise NotImplementedError("short-side execution is not implemented yet")

    split = chronological_split(data)
    entry, exit_ = strategy_signals(split.train, candidate.strategy)
    signal = _position_signal(entry, exit_)
    backtest = run_long_signal_backtest(
        split.train, signal, candidate.strategy, backtest_config
    )
    validation = validate_equity(backtest.equity, validation_policy)
    fitness = score_strategy(validation.sharpe, validation.drawdown, fitness_policy)
    experiment_id = ExperimentSpec(
        candidate.strategy, dataset_id, dataset_version, seed
    ).experiment_id
    experiment = ExperimentResult(
        experiment_id=experiment_id,
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
