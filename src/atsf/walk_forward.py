from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .backtest import BacktestConfig, BacktestResult, run_long_signal_backtest
from .generator import StrategyCandidate
from .signals import strategy_signals
from .validation import ValidationPolicy, ValidationResult, validate_equity


@dataclass(frozen=True)
class WalkForwardFold:
    train_start: object
    train_end: object
    test_start: object
    test_end: object
    result: BacktestResult
    validation: ValidationResult


@dataclass(frozen=True)
class WalkForwardResult:
    candidate_id: str
    folds: tuple[WalkForwardFold, ...]
    passed: bool
    reasons: tuple[str, ...]


def walk_forward_validate(
    data: pd.DataFrame,
    candidate: StrategyCandidate,
    train_size: int,
    test_size: int,
    step: int | None = None,
    backtest_config: BacktestConfig | None = None,
    validation_policy: ValidationPolicy | None = None,
) -> WalkForwardResult:
    """Evaluate a fixed strategy across sequential out-of-sample windows."""
    if train_size <= 0 or test_size <= 0:
        raise ValueError("train_size and test_size must be positive")
    step = step or test_size
    if step <= 0:
        raise ValueError("step must be positive")
    if len(data) < train_size + test_size:
        raise ValueError("insufficient data for one walk-forward fold")

    entry, _ = strategy_signals(data, candidate.strategy)
    folds: list[WalkForwardFold] = []
    start = 0
    while start + train_size + test_size <= len(data):
        test_start = start + train_size
        test_end = test_start + test_size
        test_data = data.iloc[test_start:test_end]
        test_signal = entry.iloc[test_start:test_end]
        result = run_long_signal_backtest(
            test_data, test_signal, candidate.strategy, backtest_config
        )
        validation = validate_equity(result.equity, validation_policy)
        folds.append(
            WalkForwardFold(
                data.index[start],
                data.index[test_start - 1],
                data.index[test_start],
                data.index[test_end - 1],
                result,
                validation,
            )
        )
        start += step

    failures = [f"fold {i} failed validation" for i, fold in enumerate(folds) if not fold.validation.passed]
    return WalkForwardResult(
        candidate.candidate_id,
        tuple(folds),
        not failures,
        tuple(failures),
    )
