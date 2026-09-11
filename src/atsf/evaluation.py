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
    windows: tuple[SegmentEvaluation, ...] = ()
    passed: bool = False
    oos_return: float = 0.0
    oos_sharpe: float = 0.0
    oos_drawdown: float = 0.0
    oos_equity: pd.Series | None = None
    oos_returns: pd.Series | None = None
    oos_trade_returns: tuple[float, ...] = ()
    folds: tuple[SegmentEvaluation, ...] | None = None

    def __post_init__(self) -> None:
        if self.folds is not None:
            if self.windows and tuple(self.windows) != tuple(self.folds):
                raise ValueError("windows and folds must match")
            object.__setattr__(self, "windows", tuple(self.folds))
        else:
            object.__setattr__(self, "folds", tuple(self.windows))


def _position_signal(entry: pd.Series, exit_: pd.Series) -> pd.Series:
    if not entry.index.equals(exit_.index):
        raise ValueError("entry and exit indexes must match")
    active = False
    values: list[bool] = []
    for timestamp in entry.index:
        if bool(exit_.loc[timestamp]): active = False
        if bool(entry.loc[timestamp]): active = True
        values.append(active)
    return pd.Series(values, index=entry.index, dtype=bool)


def _evaluate_segment(data: pd.DataFrame, strategy: StrategySpec, backtest_config: BacktestConfig | None,
                      validation_policy: ValidationPolicy | None, fitness_policy: FitnessPolicy | None) -> SegmentEvaluation:
    entry, exit_ = strategy_signals(data, strategy)
    backtest = run_long_signal_backtest(data, _position_signal(entry, exit_), strategy, backtest_config)
    validation = validate_equity(backtest.equity, validation_policy)
    return SegmentEvaluation(backtest, validation, score_strategy(validation.sharpe, validation.drawdown, fitness_policy))


def evaluate_walk_forward(data: pd.DataFrame, strategy: StrategySpec, train_size: int, validation_size: int,
                          test_size: int, step_size: int | None = None, backtest_config: BacktestConfig | None = None,
                          validation_policy: ValidationPolicy | None = None, fitness_policy: FitnessPolicy | None = None) -> WalkForwardEvaluation:
    windows: list[WalkForwardWindow] = walk_forward_windows(data, train_size, validation_size, test_size, step_size)
    if not windows: raise ValueError("window sizes produce no complete walk-forward windows")
    evaluations: list[SegmentEvaluation] = []
    oos_returns: list[pd.Series] = []
    oos_trade_returns: list[float] = []
    all_passed = True
    for window in windows:
        train = _evaluate_segment(window.train, strategy, backtest_config, validation_policy, fitness_policy)
        validation = _evaluate_segment(window.validation, strategy, backtest_config, validation_policy, fitness_policy)
        test = _evaluate_segment(window.test, strategy, backtest_config, validation_policy, fitness_policy)
        evaluations.extend((train, validation, test))
        all_passed = all_passed and train.fitness.eligible and validation.fitness.eligible and test.fitness.eligible
        oos_returns.append(test.backtest.equity.pct_change().fillna(0.0))
        oos_trade_returns.extend(test.backtest.trade_returns)
    combined_oos = pd.concat(oos_returns).sort_index()
    combined_oos = combined_oos[~combined_oos.index.duplicated(keep="first")]
    oos_equity = (1.0 + combined_oos).cumprod()
    oos_validation = validate_equity(oos_equity, validation_policy)
    return WalkForwardEvaluation(tuple(evaluations), all_passed and oos_validation.passed,
                                 float(oos_equity.iloc[-1] - 1.0), oos_validation.sharpe, oos_validation.drawdown,
                                 oos_equity, combined_oos, tuple(oos_trade_returns))
