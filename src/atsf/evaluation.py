from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .backtest import BacktestConfig, BacktestResult, run_long_signal_backtest
from .fitness import FitnessPolicy, FitnessResult, score_strategy
from .signals import position_state, strategy_signals
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
    train_passed: bool | None = None


_position_signal = position_state


def _evaluate_segment(data: pd.DataFrame, strategy: StrategySpec, backtest_config: BacktestConfig | None,
                      validation_policy: ValidationPolicy | None, fitness_policy: FitnessPolicy | None,
                      position: pd.Series | None = None) -> SegmentEvaluation:
    """Evaluate one segment. ``position`` should come from indicators warmed on prior history."""
    if position is None:
        entry, exit_ = strategy_signals(data, strategy)
        position = position_state(entry, exit_)
    backtest = run_long_signal_backtest(data, position, strategy, backtest_config)
    validation = validate_equity(backtest.equity, validation_policy)
    return SegmentEvaluation(backtest, validation, score_strategy(validation.sharpe, validation.drawdown, fitness_policy))


def _warm_positions(full_entry: pd.Series, full_exit: pd.Series, segment: pd.DataFrame) -> pd.Series:
    """Slice causally computed full-history signals to a segment, starting flat."""
    return position_state(full_entry.loc[segment.index], full_exit.loc[segment.index])


def evaluate_walk_forward(data: pd.DataFrame, strategy: StrategySpec, train_size: int, validation_size: int,
                          test_size: int, step_size: int | None = None, backtest_config: BacktestConfig | None = None,
                          validation_policy: ValidationPolicy | None = None, fitness_policy: FitnessPolicy | None = None,
                          *, require_train_pass: bool = False,
                          min_window_pass_rate: float = 1.0,
                          gate_validation: bool = True) -> WalkForwardEvaluation:
    """Rolling train/validation/test evaluation with indicators warmed on prior history.

    ``passed`` requires every validation and test segment, and the stitched OOS curve, to
    clear the gates (or, with ``min_window_pass_rate`` < 1, at least that share of
    windows, since requiring every one-year window to clear a Sharpe floor rejects any
    strategy with a single losing year). When windows roll by exactly one validation
    length, each validation segment is the previous window's test segment, so
    ``gate_validation=False`` avoids counting the same year twice. Train segments are reported (``train_passed``) but only gate when
    ``require_train_pass`` is set: in-sample results say little about edge, and a hard
    in-sample gate mostly rejects long-only strategies for living through a crash.
    """
    if not 0 < min_window_pass_rate <= 1:
        raise ValueError("min_window_pass_rate must be in (0, 1]")
    windows: list[WalkForwardWindow] = walk_forward_windows(data, train_size, validation_size, test_size, step_size)
    if not windows: raise ValueError("window sizes produce no complete walk-forward windows")
    evaluations: list[SegmentEvaluation] = []
    oos_returns: list[pd.Series] = []
    oos_trade_returns: list[float] = []
    window_results: list[bool] = []
    train_passed = True
    # Indicators are causal, so computing them once on the full history and slicing is
    # equivalent to warming each segment on everything before it, without look-ahead.
    full_entry, full_exit = strategy_signals(data, strategy)
    for window in windows:
        train, validation, test = (
            _evaluate_segment(segment, strategy, backtest_config, validation_policy, fitness_policy,
                              _warm_positions(full_entry, full_exit, segment))
            for segment in (window.train, window.validation, window.test)
        )
        evaluations.extend((train, validation, test))
        train_passed = train_passed and train.fitness.eligible
        window_ok = test.fitness.eligible and (validation.fitness.eligible or not gate_validation)
        if require_train_pass:
            window_ok = window_ok and train.fitness.eligible
        window_results.append(window_ok)
        oos_returns.append(test.backtest.equity.pct_change().fillna(0.0))
        oos_trade_returns.extend(test.backtest.trade_returns)
    combined_oos = pd.concat(oos_returns).sort_index()
    combined_oos = combined_oos[~combined_oos.index.duplicated(keep="first")]
    oos_equity = (1.0 + combined_oos).cumprod()
    oos_validation = validate_equity(oos_equity, validation_policy)
    all_passed = sum(window_results) / len(window_results) >= min_window_pass_rate - 1e-12
    return WalkForwardEvaluation(tuple(evaluations), all_passed and oos_validation.passed,
                                 float(oos_equity.iloc[-1] - 1.0), oos_validation.sharpe, oos_validation.drawdown,
                                 oos_equity, combined_oos, tuple(oos_trade_returns), train_passed=train_passed)
