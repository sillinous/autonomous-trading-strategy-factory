from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .backtest import BacktestConfig, BacktestResult, run_long_signal_backtest
from .fitness import FitnessPolicy, FitnessResult, score_strategy
from .signals import strategy_signals
from .splits import WalkForwardWindow, walk_forward_windows
from .strategy import StrategySpec
from .validation import ValidationPolicy, ValidationResult, validate_equity


@dataclass(frozen=True)
class SegmentEvaluation:
    backtest: BacktestResult
    validation: ValidationResult
    fitness: FitnessResult


@dataclass(frozen=True)
class WalkForwardEvaluation:
    windows: tuple[SegmentEvaluation, ...]
    passed: bool
    oos_return: float
    oos_sharpe: float
    oos_drawdown: float


def _position_signal(entry: pd.Series, exit_: pd.Series) -> pd.Series:
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


def _evaluate_segment(
    data: pd.DataFrame,
    strategy: StrategySpec,
    backtest_config: BacktestConfig | None,
    validation_policy: ValidationPolicy | None,
    fitness_policy: FitnessPolicy | None,
) -> SegmentEvaluation:
    entry, exit_ = strategy_signals(data, strategy)
    signal = _position_signal(entry, exit_)
    backtest = run_long_signal_backtest(data, signal, strategy, backtest_config)
    validation = validate_equity(backtest.equity, validation_policy)
    fitness = score_strategy(validation.sharpe, validation.drawdown, fitness_policy)
    return SegmentEvaluation(backtest, validation, fitness)


def evaluate_walk_forward(
    data: pd.DataFrame,
    strategy: StrategySpec,
    train_size: int,
    validation_size: int,
    test_size: int,
    step_size: int | None = None,
    backtest_config: BacktestConfig | None = None,
    validation_policy: ValidationPolicy | None = None,
    fitness_policy: FitnessPolicy | None = None,
) -> WalkForwardEvaluation:
    """Evaluate fixed strategy parameters on chronological train/validation/OOS windows."""
    windows: list[WalkForwardWindow] = walk_forward_windows(
        data, train_size, validation_size, test_size, step_size
    )
    if not windows:
        raise ValueError("window sizes produce no complete walk-forward windows")

    evaluations: list[SegmentEvaluation] = []
    oos_returns: list[pd.Series] = []
    all_passed = True
    for window in windows:
        train = _evaluate_segment(
            window.train, strategy, backtest_config, validation_policy, fitness_policy
        )
        validation = _evaluate_segment(
            window.validation, strategy, backtest_config, validation_policy, fitness_policy
        )
        test = _evaluate_segment(
            window.test, strategy, backtest_config, validation_policy, fitness_policy
        )
        evaluations.extend((train, validation, test))
        all_passed = (
            all_passed
            and train.fitness.eligible
            and validation.fitness.eligible
            and test.fitness.eligible
        )
        oos_returns.append(test.backtest.equity.pct_change().fillna(0.0))

    combined_oos = pd.concat(oos_returns).sort_index()
    oos_equity = (1.0 + combined_oos).cumprod()
    oos_validation = validate_equity(oos_equity, validation_policy)
    return WalkForwardEvaluation(
        windows=tuple(evaluations),
        passed=all_passed and oos_validation.passed,
        oos_return=float(oos_equity.iloc[-1] - 1.0),
        oos_sharpe=oos_validation.sharpe,
        oos_drawdown=oos_validation.drawdown,
    )
