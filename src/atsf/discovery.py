from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .backtest import BacktestConfig, BacktestResult, run_long_signal_backtest
from .generator import StrategyCandidate
from .signals import strategy_signals
from .validation import ValidationPolicy, ValidationResult, validate_equity


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate_id: str
    strategy_name: str
    backtest: BacktestResult
    validation: ValidationResult


@dataclass(frozen=True)
class DiscoveryResult:
    evaluations: tuple[CandidateEvaluation, ...]
    accepted: tuple[CandidateEvaluation, ...]


def evaluate_candidates(
    data: pd.DataFrame,
    candidates: tuple[StrategyCandidate, ...],
    backtest_config: BacktestConfig | None = None,
    validation_policy: ValidationPolicy | None = None,
) -> DiscoveryResult:
    """Backtest and validate generated candidates without executing live orders."""
    evaluations: list[CandidateEvaluation] = []
    for candidate in candidates:
        entry, _ = strategy_signals(data, candidate.strategy)
        result = run_long_signal_backtest(
            data, entry, candidate.strategy, backtest_config
        )
        validation = validate_equity(result.equity, validation_policy)
        evaluations.append(
            CandidateEvaluation(
                candidate.candidate_id,
                candidate.strategy.name,
                result,
                validation,
            )
        )
    accepted = tuple(item for item in evaluations if item.validation.passed)
    return DiscoveryResult(tuple(evaluations), accepted)
