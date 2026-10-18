from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

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


def walk_forward_validate(data: pd.DataFrame, candidate: StrategyCandidate, train_size: int, test_size: int, step: int | None = None, backtest_config: BacktestConfig | None = None, validation_policy: ValidationPolicy | None = None) -> WalkForwardResult:
    """Evaluate a fixed strategy on sequential OOS windows, including a final partial window."""
    if train_size <= 0 or test_size <= 0:
        raise ValueError("train_size and test_size must be positive")
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("walk-forward data index must be a DatetimeIndex")
    if not data.index.is_monotonic_increasing or data.index.has_duplicates:
        raise ValueError("walk-forward timestamps must be sorted and unique")
    if "close" not in data.columns:
        raise ValueError("walk-forward data must contain a close column")
    try:
        close = pd.to_numeric(data["close"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("walk-forward close prices must be numeric") from exc
    if close.isna().any() or not close.map(isfinite).all() or (close <= 0).any():
        raise ValueError("walk-forward close prices must be finite and positive")
    step = test_size if step is None else step
    if step <= 0:
        raise ValueError("step must be positive")
    if len(data) < train_size + test_size:
        raise ValueError("insufficient data for one walk-forward fold")
    folds: list[WalkForwardFold] = []
    start = 0
    while start + train_size < len(data):
        train_end = start + train_size
        test_end = min(train_end + test_size, len(data))
        context = data.iloc[start:test_end]
        context_entry, _ = strategy_signals(context, candidate.strategy)
        test_data = data.iloc[train_end:test_end]
        actual_test_size = len(test_data)
        if actual_test_size == 0:
            break
        test_signal = context_entry.iloc[-actual_test_size:]
        result = run_long_signal_backtest(test_data, test_signal, candidate.strategy, backtest_config)
        validation = validate_equity(result.equity, validation_policy)
        folds.append(WalkForwardFold(data.index[start], data.index[train_end - 1], data.index[train_end], data.index[test_end - 1], result, validation))
        start += step
    failures = [f"fold {i} failed validation" for i, fold in enumerate(folds) if not fold.validation.passed]
    return WalkForwardResult(candidate.candidate_id, tuple(folds), not failures, tuple(failures))
