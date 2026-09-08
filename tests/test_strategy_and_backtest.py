import pandas as pd
import pytest

from atsf.backtest import BacktestConfig, run_long_signal_backtest
from atsf.metrics import max_drawdown, sharpe_ratio
from atsf.strategy import (
    Comparator,
    Condition,
    Indicator,
    PositionSizing,
    RiskLimits,
    Signal,
    StrategySpec,
)


def make_strategy() -> StrategySpec:
    return StrategySpec(
        name="test",
        universe=["TEST"],
        indicators=[Indicator(name="sma", period=20)],
        entry=Signal(all=[Condition(left="close", comparator=Comparator.GT, right="sma")]),
        exit=Signal(all=[Condition(left="close", comparator=Comparator.LT, right="sma")]),
        position_sizing=PositionSizing(method="fixed_fraction", value=1, max_position=1),
        risk=RiskLimits(max_position=1),
    )


def test_strategy_rejects_inconsistent_position_limit():
    with pytest.raises(ValueError):
        StrategySpec(
            name="bad",
            universe=["TEST"],
            entry=Signal(all=[Condition(left="close", comparator=Comparator.GT, right=1)]),
            exit=Signal(all=[Condition(left="close", comparator=Comparator.LT, right=1)]),
            position_sizing=PositionSizing(method="fixed_fraction", value=0.5, max_position=0.5),
            risk=RiskLimits(max_position=1),
        )


def test_backtest_applies_next_bar_execution_and_costs():
    index = pd.date_range("2024-01-01", periods=5, freq="D")
    data = pd.DataFrame({"close": [100, 110, 105, 115, 120]}, index=index)
    signal = pd.Series([False, True, True, False, False], index=index)
    result = run_long_signal_backtest(
        data, signal, make_strategy(), BacktestConfig(initial_cash=100_000, commission_bps=1, slippage_bps=2)
    )
    assert result.equity.iloc[-1] > 100_000
    assert result.max_drawdown <= 0
    assert not result.trades.empty


def test_metrics_are_defined_for_flat_equity():
    equity = pd.Series([100, 100, 100])
    assert sharpe_ratio(equity) == 0
    assert max_drawdown(equity) == 0
