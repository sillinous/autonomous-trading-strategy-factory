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


def test_position_does_not_earn_return_before_next_close_fill():
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    data = pd.DataFrame({"close": [100, 200, 300]}, index=index)
    signal = pd.Series([False, True, True], index=index)
    result = run_long_signal_backtest(
        data, signal, make_strategy(), BacktestConfig(commission_bps=0, slippage_bps=0)
    )

    # Signal from day 2 fills at day 3 close; the 100 -> 200 move is not earned.
    assert result.trades.iloc[0]["price"] == 300
    assert result.returns.iloc[1] == pytest.approx(0.0)
    assert result.returns.iloc[2] == pytest.approx(0.0)
    assert result.total_return == pytest.approx(0.0)


def test_exit_bar_return_is_earned_until_the_close_fill():
    index = pd.date_range("2024-01-01", periods=4, freq="D")
    data = pd.DataFrame({"close": [100, 110, 120, 130]}, index=index)
    signal = pd.Series([False, True, False, False], index=index)
    result = run_long_signal_backtest(
        data, signal, make_strategy(), BacktestConfig(commission_bps=0, slippage_bps=0)
    )

    # Entry fills at 120; exposure earns only the following 120 -> 130 interval.
    assert result.trades["price"].tolist() == [120.0, 130.0]
    assert result.returns.iloc[1] == pytest.approx(0.0)
    assert result.returns.iloc[2] == pytest.approx(0.0)
    assert result.returns.iloc[3] == pytest.approx(130 / 120 - 1)


def test_open_position_is_liquidated_at_window_end_with_exit_friction():
    index = pd.date_range("2024-01-01", periods=4, freq="D")
    data = pd.DataFrame({"close": [100, 100, 110, 120]}, index=index)
    signal = pd.Series([False, True, True, True], index=index)
    result = run_long_signal_backtest(
        data, signal, make_strategy(), BacktestConfig(commission_bps=1, slippage_bps=2)
    )

    assert result.trades["action"].tolist() == ["buy", "sell"]
    assert result.trades.iloc[-1]["timestamp"] == index[-1]
    assert len(result.trade_returns) == 1
    assert result.trade_returns[0] == pytest.approx(120 / 110 - 1 - 0.0006)
    assert result.returns.iloc[-1] < data["close"].pct_change().fillna(0).iloc[-1]


def test_position_closed_on_final_bar_is_not_double_counted():
    index = pd.date_range("2024-01-01", periods=4, freq="D")
    data = pd.DataFrame({"close": [100, 100, 110, 120]}, index=index)
    signal = pd.Series([False, True, False, False], index=index)
    result = run_long_signal_backtest(data, signal, make_strategy())

    assert result.trades["action"].tolist() == ["buy", "sell"]
    assert len(result.trade_returns) == 1
    assert result.trades.iloc[-1]["timestamp"] == index[-1]


def test_flat_signal_has_no_trades_or_trade_returns():
    index = pd.date_range("2024-01-01", periods=4, freq="D")
    data = pd.DataFrame({"close": [100, 101, 99, 102]}, index=index)
    signal = pd.Series([False, False, False, False], index=index)
    result = run_long_signal_backtest(data, signal, make_strategy())

    assert result.trades.empty
    assert result.trade_returns == ()


def test_metrics_are_defined_for_flat_equity():
    equity = pd.Series([100, 100, 100])
    assert sharpe_ratio(equity) == 0
    assert max_drawdown(equity) == 0
